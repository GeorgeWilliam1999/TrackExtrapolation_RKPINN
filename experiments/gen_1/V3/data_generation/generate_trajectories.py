#!/usr/bin/env python3
"""
V3 Trajectory Generation: Generate full trajectories with 5mm resolution.

This script generates high-resolution trajectories that can be reused
to extract millions of variable dz training samples.

Key Points:
- Generate trajectories over the full detector range (0 → 15000 mm)
- Use 5mm RK4 steps for high accuracy
- Save full trajectory (not just endpoints)
- One trajectory → thousands of training samples via segment extraction

Usage:
    # Generate 10k trajectories (recommended for 100M samples)
    python generate_trajectories.py \
        --n_trajectories 10000 \
        --z_start 0 \
        --z_end 15000 \
        --step_size 5 \
        --output trajectories_10k.npz

Author: G. Scriven
Date: January 2026
"""

import numpy as np
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List, Optional
import sys
import time

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from rk4_propagator import RK4Integrator, generate_random_track
from magnetic_field import get_field_numpy


# Global variables for worker process
_WORKER_FIELD = None
_WORKER_INTEGRATOR = None


def _init_worker(use_real_field: bool, polarity: int, step_size: float):
    """Initialize field and integrator once per worker process."""
    global _WORKER_FIELD, _WORKER_INTEGRATOR
    _WORKER_FIELD = get_field_numpy(use_interpolated=use_real_field, polarity=polarity)
    _WORKER_INTEGRATOR = RK4Integrator(field=_WORKER_FIELD, step_size=step_size)


def generate_single_trajectory(args: Tuple) -> Tuple[np.ndarray, float]:
    """
    Generate a single full trajectory.
    
    Args:
        args: (track_id, z_start, z_end, p_range)
        
    Returns:
        trajectory: Array of shape (n_steps, 6) = [z, x, y, tx, ty, qop]
        momentum: Track momentum in GeV
    """
    global _WORKER_INTEGRATOR
    
    track_id, z_start, z_end, p_range = args
    integrator = _WORKER_INTEGRATOR
    
    # Generate random initial state
    state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
    
    # Propagate with full trajectory
    trajectory = integrator.propagate(state_initial, z_start, z_end, save_trajectory=True)
    
    # Validate
    max_retries = 3
    retry = 0
    while (len(trajectory) < 10 or not np.all(np.isfinite(trajectory))) and retry < max_retries:
        state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
        trajectory = integrator.propagate(state_initial, z_start, z_end, save_trajectory=True)
        retry += 1
    
    if len(trajectory) < 10 or not np.all(np.isfinite(trajectory)):
        # Return empty on failure
        return np.zeros((0, 6)), 0.0
    
    return trajectory.astype(np.float32), momentum


def generate_trajectories(
    n_trajectories: int,
    z_start: float = 0.0,
    z_end: float = 15000.0,
    p_range: Tuple[float, float] = (0.5, 100.0),
    step_size: float = 5.0,
    polarity: int = 1,
    n_workers: int = 8,
    use_real_field: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate full trajectories in parallel.
    
    Args:
        n_trajectories: Number of trajectories to generate
        z_start: Starting z position (mm)
        z_end: Ending z position (mm)
        p_range: Momentum range (GeV)
        step_size: RK4 step size (mm)
        polarity: Magnet polarity
        n_workers: Number of parallel workers
        use_real_field: Use interpolated field map
        
    Returns:
        trajectories: Object array of trajectories
        momenta: Array of momenta
    """
    field_type = "REAL FIELD MAP" if use_real_field else "GAUSSIAN APPROX"
    
    print(f"\n{'='*70}")
    print(f"V3 Trajectory Generation")
    print(f"{'='*70}")
    print(f"  Trajectories: {n_trajectories:,}")
    print(f"  z range: {z_start:.0f} → {z_end:.0f} mm")
    print(f"  Momentum: {p_range[0]:.1f} - {p_range[1]:.1f} GeV")
    print(f"  RK4 step: {step_size} mm")
    print(f"  Expected points per trajectory: ~{int((z_end - z_start) / step_size)}")
    print(f"  Field: {field_type}, polarity={'MagUp' if polarity > 0 else 'MagDown'}")
    print(f"  Workers: {n_workers}")
    print(f"{'='*70}\n")
    
    # Prepare arguments
    args_list = [(i, z_start, z_end, p_range) for i in range(n_trajectories)]
    
    # Generate in parallel
    start_time = time.time()
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(use_real_field, polarity, step_size)) as pool:
        results = list(pool.imap(generate_single_trajectory, args_list, chunksize=10))
    
    elapsed = time.time() - start_time
    
    # Unpack results
    trajectories = [r[0] for r in results]
    momenta = np.array([r[1] for r in results])
    
    # Filter failed trajectories
    valid_mask = momenta > 0
    n_failed = np.sum(~valid_mask)
    
    if n_failed > 0:
        print(f"  WARNING: {n_failed:,}/{n_trajectories:,} trajectories failed")
        trajectories = [t for t, v in zip(trajectories, valid_mask) if v]
        momenta = momenta[valid_mask]
    
    # Convert to object array for saving
    T_array = np.empty(len(trajectories), dtype=object)
    for i, t in enumerate(trajectories):
        T_array[i] = t
    
    # Statistics
    lengths = [len(t) for t in trajectories]
    print(f"\n✓ Generated {len(trajectories):,} trajectories in {elapsed:.1f}s")
    print(f"  Points per trajectory: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    print(f"  Total data points: {sum(lengths):,}")
    print(f"  Generation rate: {n_trajectories/elapsed:.0f} traj/s")
    
    return T_array, momenta


def main():
    parser = argparse.ArgumentParser(
        description="Generate full trajectories for V3 training"
    )
    parser.add_argument('--n_trajectories', type=int, default=10000,
                       help='Number of trajectories (default: 10000)')
    parser.add_argument('--z_start', type=float, default=0.0,
                       help='Starting z position in mm (default: 0)')
    parser.add_argument('--z_end', type=float, default=15000.0,
                       help='Ending z position in mm (default: 15000)')
    parser.add_argument('--p_min', type=float, default=0.5,
                       help='Minimum momentum in GeV (default: 0.5)')
    parser.add_argument('--p_max', type=float, default=100.0,
                       help='Maximum momentum in GeV (default: 100)')
    parser.add_argument('--step_size', type=float, default=5.0,
                       help='RK4 step size in mm (default: 5)')
    parser.add_argument('--polarity', type=int, choices=[-1, 1], default=1,
                       help='Magnet polarity: +1=MagUp, -1=MagDown (default: 1)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='trajectories.npz',
                       help='Output file (default: trajectories.npz)')
    parser.add_argument('--use_gaussian_field', action='store_true',
                       help='Use Gaussian field approximation instead of real field map')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Generate
    T, P = generate_trajectories(
        n_trajectories=args.n_trajectories,
        z_start=args.z_start,
        z_end=args.z_end,
        p_range=(args.p_min, args.p_max),
        step_size=args.step_size,
        polarity=args.polarity,
        n_workers=args.workers,
        use_real_field=not args.use_gaussian_field
    )
    
    # Save
    print(f"\nSaving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, T=T, P=P)
    
    print(f"  File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    print("\n" + "="*70)
    print("TRAJECTORY GENERATION COMPLETE")
    print("="*70)
    print(f"\nNext step: Extract training samples:")
    print(f"  python extract_segments.py --input {args.output} --n_samples 100000000 --output training_mlp_v3.npz")


if __name__ == "__main__":
    main()
