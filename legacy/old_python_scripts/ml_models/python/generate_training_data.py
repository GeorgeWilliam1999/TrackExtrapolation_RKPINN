#!/usr/bin/env python3
"""
Generate and save training data for track extrapolation.

This script generates large datasets using parallel processing for speed.
Data is saved to .npy files for reuse.

Usage:
    python generate_training_data.py --samples 100000 --output data/
    
Author: G. Scriven
Date: 2026-01-12
"""

import numpy as np
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time


class LHCbMagneticField:
    """Simplified LHCb dipole magnetic field model."""
    def __init__(self, polarity: int = 1):
        self.polarity = polarity
        self.B0 = 1.0
        self.z_center = 5250.0
        self.z_halfwidth = 2500.0
        
    def get_field(self, x: float, y: float, z: float):
        z_rel = (z - self.z_center) / self.z_halfwidth
        By_profile = np.exp(-0.5 * z_rel**2)
        r_trans = np.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        By = self.polarity * self.B0 * By_profile * fringe_factor
        Bx = -0.01 * By * (x / 1000.0)
        Bz = 0.0
        return (Bx, By, Bz)


class RKIntegrator:
    """Fast RK4 integrator for ground truth."""
    def __init__(self, field, step_size: float = 10.0):
        self.field = field
        self.step_size = step_size
        self.c_light = 299.792458
        
    def derivatives(self, z: float, state: np.ndarray):
        x, y, tx, ty, qop = state
        Bx, By, Bz = self.field.get_field(x, y, z)
        factor = qop * self.c_light * 1e-3
        norm = np.sqrt(1.0 + tx**2 + ty**2)
        
        dtx_dz = factor * norm * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = factor * norm * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        
        return np.array([tx, ty, dtx_dz, dty_dz, 0.0])
    
    def rk4_step(self, z: float, state: np.ndarray, h: float):
        k1 = self.derivatives(z, state)
        k2 = self.derivatives(z + h/2, state + h*k1/2)
        k3 = self.derivatives(z + h/2, state + h*k2/2)
        k4 = self.derivatives(z + h, state + h*k3)
        return state + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def propagate(self, state_in: np.ndarray, z_in: float, z_out: float):
        state = state_in.copy()
        z = z_in
        dz = z_out - z_in
        
        h = self.step_size if dz > 0 else -self.step_size
        n_steps = int(np.ceil(abs(dz) / self.step_size))
        h = dz / n_steps
        
        for _ in range(n_steps):
            state = self.rk4_step(z, state, h)
            z += h
        
        return state


def generate_sample(args):
    """Generate a single training sample (for parallel processing)."""
    seed, z_in, z_out = args
    np.random.seed(seed)
    
    field = LHCbMagneticField(polarity=1)
    integrator = RKIntegrator(field, step_size=10.0)
    
    # Sample parameters
    x0 = np.random.uniform(-900, 900)
    y0 = np.random.uniform(-750, 750)
    tx0 = np.random.uniform(-0.3, 0.3)
    ty0 = np.random.uniform(-0.25, 0.25)
    
    # Log-uniform momentum (2-100 GeV)
    log_p = np.random.uniform(np.log(2.0), np.log(100))
    p_gev = np.exp(log_p)
    charge = np.random.choice([-1, 1])
    qop = charge / (p_gev * 1000.0)
    
    state_in = np.array([x0, y0, tx0, ty0, qop])
    
    try:
        state_out = integrator.propagate(state_in, z_in, z_out)
        
        # Check for valid result
        if np.any(np.isnan(state_out)) or np.any(np.isinf(state_out)):
            return None
        
        dz = z_out - z_in
        X = [x0, y0, tx0, ty0, qop, dz]
        Y = [state_out[0], state_out[1], state_out[2], state_out[3]]
        P = p_gev
        
        return (X, Y, P)
    except:
        return None


def generate_training_data(num_samples: int, n_workers: int = None, 
                          z_in: float = 3000.0, z_out: float = 7000.0):
    """Generate training data in parallel."""
    
    if n_workers is None:
        n_workers = min(cpu_count(), 32)  # Limit to 32 cores
    
    print(f"Generating {num_samples} samples using {n_workers} workers...")
    
    # Create arguments for parallel processing
    seeds = np.random.randint(0, 2**31, size=num_samples)
    args_list = [(seed, z_in, z_out) for seed in seeds]
    
    # Generate in parallel
    start = time.time()
    with Pool(n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_sample, args_list, chunksize=100)):
            if result is not None:
                results.append(result)
            
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (num_samples - i - 1) / rate
                print(f"  Progress: {i+1}/{num_samples} ({rate:.1f} samples/s, {remaining:.0f}s remaining)")
    
    elapsed = time.time() - start
    print(f"Generated {len(results)} valid samples in {elapsed:.1f}s ({len(results)/elapsed:.1f} samples/s)")
    
    # Convert to arrays
    X_list = [r[0] for r in results]
    Y_list = [r[1] for r in results]
    P_list = [r[2] for r in results]
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    P = np.array(P_list, dtype=np.float32)
    
    return X, Y, P


def main():
    parser = argparse.ArgumentParser(description='Generate training data for track extrapolation')
    parser.add_argument('--samples', type=int, default=100000, help='Number of samples to generate')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='../data/', help='Output directory')
    parser.add_argument('--name', type=str, default='train', help='Dataset name prefix')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRAINING DATA GENERATION")
    print("="*70)
    print(f"Samples: {args.samples}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Output: {output_dir}")
    print()
    
    # Generate data
    X, Y, P = generate_training_data(args.samples, n_workers=args.workers)
    
    # Save to disk
    print(f"\nSaving data to {output_dir}...")
    np.save(output_dir / f'X_{args.name}.npy', X)
    np.save(output_dir / f'Y_{args.name}.npy', Y)
    np.save(output_dir / f'P_{args.name}.npy', P)
    
    print(f"✓ Saved X_{args.name}.npy: {X.shape}")
    print(f"✓ Saved Y_{args.name}.npy: {Y.shape}")
    print(f"✓ Saved P_{args.name}.npy: {P.shape}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Momentum range: {P.min():.2f} - {P.max():.2f} GeV")
    print(f"  Mean momentum: {P.mean():.2f} GeV")
    print(f"  X range: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}] mm")
    print(f"  Y range: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}] mm")
    print(f"  File size: ~{(X.nbytes + Y.nbytes + P.nbytes) / 1024**2:.1f} MB")
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
