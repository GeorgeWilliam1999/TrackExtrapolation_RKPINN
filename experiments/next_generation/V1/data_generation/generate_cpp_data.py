#!/usr/bin/env python3
"""
Generate Training Data for Track Extrapolation using Real Field Map

Uses RK4 integrator with REAL FIELD MAP INTERPOLATION (twodip.rtf) to generate
high-precision ground truth extrapolations. This ensures consistency between
training data and PINN physics loss.

IMPORTANT: This script uses the UNIFIED field module to ensure all code
(data generation, PINN training, validation) uses the same magnetic field.

Field Map: twodip.rtf
- 81 × 81 × 146 grid points
- x: [-4000, 4000] mm, y: [-4000, 4000] mm, z: [-500, 14000] mm
- Grid spacing: 100mm
- Trilinear interpolation error: O(h²) ≈ 10⁻⁵ T

Author: G. Scriven
Date: 2026-01-19 (Fixed to use real field interpolation)
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import Tuple
import sys

# Add utils to path - USE THE UNIFIED FIELD MODULE
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from magnetic_field import get_field_numpy, C_LIGHT, InterpolatedFieldNumpy


class RK4IntegratorRealField:
    """
    RK4 integrator using the REAL interpolated field map.
    
    This is a standalone version optimized for batch data generation.
    Uses the unified field module for consistency with PINN training.
    """
    
    def __init__(self, field, step_size: float = 5.0):
        self.field = field
        self.step_size = step_size
        self.c_light = C_LIGHT
    
    def derivatives(self, state: np.ndarray, z: float) -> np.ndarray:
        """Compute state derivatives using Lorentz force."""
        x, y, tx, ty, qop = state
        
        # Get field at current position from REAL FIELD MAP
        Bx, By, Bz = self.field(x, y, z)
        
        # Lorentz force equations
        kappa = self.c_light * qop
        N = np.sqrt(1 + tx**2 + ty**2)
        
        dx_dz = tx
        dy_dz = ty
        dtx_dz = kappa * N * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = kappa * N * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        dqop_dz = 0.0
        
        return np.array([dx_dz, dy_dz, dtx_dz, dty_dz, dqop_dz])
    
    def rk4_step(self, state: np.ndarray, z: float, dz: float) -> np.ndarray:
        """Single RK4 integration step."""
        k1 = self.derivatives(state, z)
        k2 = self.derivatives(state + 0.5 * dz * k1, z + 0.5 * dz)
        k3 = self.derivatives(state + 0.5 * dz * k2, z + 0.5 * dz)
        k4 = self.derivatives(state + dz * k3, z + dz)
        return state + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def propagate(self, state: np.ndarray, z_start: float, z_end: float) -> np.ndarray:
        """Propagate track from z_start to z_end."""
        current_state = state.copy()
        current_z = z_start
        dz = z_end - z_start
        step = self.step_size if dz > 0 else -self.step_size
        
        with np.errstate(over='ignore', invalid='ignore'):
            while abs(current_z - z_end) > abs(step):
                current_state = self.rk4_step(current_state, current_z, step)
                current_z += step
                
                if not np.all(np.isfinite(current_state)):
                    return current_state
                
                # Acceptance cuts
                if abs(current_state[0]) > 5000 or abs(current_state[1]) > 5000:
                    return np.full(5, np.nan)
                if abs(current_state[2]) > 2.0 or abs(current_state[3]) > 2.0:
                    return np.full(5, np.nan)
            
            # Final fractional step
            remaining = z_end - current_z
            if abs(remaining) > 1e-6:
                current_state = self.rk4_step(current_state, current_z, remaining)
        
        return current_state


def generate_training_data(
    n_tracks: int,
    z_start: float,
    z_end: float,
    p_min: float,
    p_max: float,
    seed: int,
    step_size: float = 5.0,
    polarity: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data using RK4 with REAL FIELD MAP interpolation.
    
    Args:
        n_tracks: Number of tracks to generate
        z_start: Initial z position (mm)
        z_end: Final z position (mm)
        p_min: Minimum momentum (GeV)
        p_max: Maximum momentum (GeV)
        seed: Random seed
        step_size: RK4 step size (mm), default 5mm for high accuracy
        polarity: Magnet polarity (+1 MagUp, -1 MagDown)
        
    Returns:
        X: Input states [n_tracks, 6] = [x, y, tx, ty, q/p, dz] at z_start
        Y: Output states [n_tracks, 4] = [x, y, tx, ty] at z_end
        P: Momenta [n_tracks] in GeV
    """
    np.random.seed(seed)
    
    # Load REAL FIELD MAP (this is the key difference!)
    print("Loading REAL field map for interpolation...")
    field = get_field_numpy(use_interpolated=True, polarity=polarity)
    
    if isinstance(field, InterpolatedFieldNumpy):
        print("✓ Using INTERPOLATED field map (twodip.rtf)")
    else:
        print("⚠️ WARNING: Falling back to Gaussian approximation!")
    
    integrator = RK4IntegratorRealField(field, step_size=step_size)
    
    # Allocate arrays
    X = np.zeros((n_tracks, 6))
    Y = np.zeros((n_tracks, 4))
    P = np.zeros(n_tracks)
    
    dz = z_end - z_start
    valid_count = 0
    attempt = 0
    max_attempts = n_tracks * 3  # Allow retries for failed tracks
    
    while valid_count < n_tracks and attempt < max_attempts:
        attempt += 1
        
        # Random momentum (log-uniform for better coverage)
        p_gev = np.exp(np.random.uniform(np.log(p_min), np.log(p_max)))
        charge = np.random.choice([-1, 1])
        qop = charge / (p_gev * 1000.0)  # 1/MeV
        
        # Random initial state
        x = np.random.uniform(-300, 300)
        y = np.random.uniform(-250, 250)
        tx = np.random.uniform(-0.15, 0.15)
        ty = np.random.uniform(-0.15, 0.15)
        
        state_initial = np.array([x, y, tx, ty, qop])
        
        # Propagate using RK4 with REAL FIELD
        state_final = integrator.propagate(state_initial, z_start, z_end)
        
        # Skip if propagation failed
        if not np.all(np.isfinite(state_final)):
            continue
        
        # Store result
        X[valid_count] = [x, y, tx, ty, qop, dz]
        Y[valid_count] = state_final[:4]  # x, y, tx, ty only
        P[valid_count] = p_gev
        valid_count += 1
        
        # Progress indicator
        if valid_count % 1000 == 0:
            print(f"  Generated {valid_count}/{n_tracks} tracks...")
    
    if valid_count < n_tracks:
        print(f"⚠️ Only generated {valid_count}/{n_tracks} valid tracks")
        X = X[:valid_count]
        Y = Y[:valid_count]
        P = P[:valid_count]
    
    return X, Y, P


def save_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    output_file: Path,
    metadata: dict
):
    """Save dataset with metadata."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_file,
        X=X,
        Y=Y,
        P=P,
        metadata=json.dumps(metadata)
    )
    
    file_size_mb = output_file.stat().st_size / (1024**2)
    print(f"✓ Saved: {output_file} ({file_size_mb:.2f} MB, {len(P)} tracks)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate track extrapolation data using REAL FIELD MAP interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses the REAL LHCb field map (twodip.rtf) with trilinear interpolation.
This ensures consistency between training data and PINN physics loss.

Examples:
  # Generate 10k tracks
  python generate_cpp_data.py --n-tracks 10000 --output data/batch_0.npz --seed 0
  
  # HTCondor batch job
  python generate_cpp_data.py --n-tracks 10000 --output data/batch_$(Process).npz --seed $(Process)
        """
    )
    
    # For backward compatibility, accept but ignore --extrapolator
    parser.add_argument('--extrapolator', type=str, default='RK4_RealField',
                       help='(Ignored) Now uses RK4 with real field interpolation')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (e.g., data/batch_0.npz)')
    
    parser.add_argument('--n-tracks', type=int, default=10000,
                       help='Number of tracks to generate (default: 10000)')
    
    parser.add_argument('--z-start', type=float, default=4000.0,
                       help='Initial z position in mm (default: 4000, VELO exit)')
    
    parser.add_argument('--z-end', type=float, default=12000.0,
                       help='Final z position in mm (default: 12000, T-station)')
    
    parser.add_argument('--p-min', type=float, default=0.5,
                       help='Minimum momentum in GeV (default: 0.5)')
    
    parser.add_argument('--p-max', type=float, default=100.0,
                       help='Maximum momentum in GeV (default: 100.0)')
    
    parser.add_argument('--step-size', type=float, default=5.0,
                       help='RK4 step size in mm (default: 5.0 for high accuracy)')
    
    parser.add_argument('--polarity', type=int, default=-1, choices=[-1, 1],
                       help='Magnet polarity: -1=MagDown (default), +1=MagUp')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training Data Generation with REAL FIELD MAP Interpolation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Output: {args.output}")
    print(f"  Tracks: {args.n_tracks}")
    print(f"  z range: {args.z_start} → {args.z_end} mm (Δz = {args.z_end - args.z_start} mm)")
    print(f"  p range: {args.p_min} - {args.p_max} GeV")
    print(f"  RK4 step: {args.step_size} mm")
    print(f"  Polarity: {'MagDown' if args.polarity < 0 else 'MagUp'}")
    print(f"  Seed: {args.seed}")
    print()
    
    # Generate data with REAL FIELD
    X, Y, P = generate_training_data(
        n_tracks=args.n_tracks,
        z_start=args.z_start,
        z_end=args.z_end,
        p_min=args.p_min,
        p_max=args.p_max,
        seed=args.seed,
        step_size=args.step_size,
        polarity=args.polarity
    )
    
    # Save with metadata
    metadata = {
        'field_model': 'InterpolatedFieldNumpy (twodip.rtf)',
        'field_map': 'twodip.rtf',
        'interpolation': 'trilinear',
        'interpolation_error': 'O(h^2) ~ 1e-5 T',
        'integrator': 'RK4',
        'step_size_mm': args.step_size,
        'n_tracks': len(P),
        'z_start_mm': args.z_start,
        'z_end_mm': args.z_end,
        'p_min_gev': args.p_min,
        'p_max_gev': args.p_max,
        'polarity': args.polarity,
        'seed': args.seed,
        'c_light': C_LIGHT,
        'generated_by': 'generate_cpp_data.py (FIXED: real field interpolation)',
        'date': '2026-01-19'
    }
    
    save_dataset(X, Y, P, Path(args.output), metadata)
    
    print()
    print(f"Dataset statistics:")
    print(f"  Tracks: {len(P)}")
    print(f"  Momentum: {P.min():.2f} - {P.max():.2f} GeV (mean: {P.mean():.2f})")
    print(f"  X deflection: mean={np.mean(Y[:,0] - X[:,0]):.1f}mm, std={np.std(Y[:,0] - X[:,0]):.1f}mm")
    print()
    print("=" * 70)
    print("✓ DONE - Data generated with REAL FIELD MAP interpolation")
    print("=" * 70)


if __name__ == '__main__':
    main()
