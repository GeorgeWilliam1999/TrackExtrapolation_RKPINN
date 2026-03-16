#!/usr/bin/env python3
"""
V3 Data Generation: Variable Step Size Training Data

Generates track extrapolation training data with VARIABLE dz values.
This is the key difference from V1/V2 which used fixed dz=8000.

Key Features:
- Variable dz: uniform distribution in [dz_min, dz_max]
- Variable z_start: allows different starting positions
- Full momentum range: 0.5-100 GeV
- Uses real field map (twodip.rtf) for accuracy

Usage:
    # Generate 10M samples for testing
    python generate_variable_dz.py --n_samples 10000000 --output data/test_v3_10M.npz
    
    # Generate 100M samples for production training
    python generate_variable_dz.py --n_samples 100000000 --output data/training_v3_100M.npz
    
    # Custom dz range
    python generate_variable_dz.py --dz_min 1000 --dz_max 10000 --n_samples 50000000

Author: G. Scriven
Date: January 2026
"""

import numpy as np
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Optional
import sys
import time
import json

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'utils'))
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


def generate_single_track_variable_dz(args: Tuple) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate single track propagation with variable dz.
    
    Args:
        args: (track_id, z_start_range, dz_range, p_range)
        
    Returns:
        input_state: [x, y, tx, ty, qop, dz] at z_start
        output_state: [x, y, tx, ty] at z_end
        momentum: Track momentum in GeV
        z_start: Starting z position
        z_end: Ending z position
    """
    global _WORKER_INTEGRATOR
    
    track_id, z_start_range, dz_range, p_range = args
    
    # Use pre-initialized integrator from worker process
    integrator = _WORKER_INTEGRATOR
    
    # Sample random z_start and dz
    z_start = np.random.uniform(z_start_range[0], z_start_range[1])
    dz = np.random.uniform(dz_range[0], dz_range[1])
    z_end = z_start + dz
    
    # Generate random initial state
    state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
    
    # Propagate to z_end
    state_final = integrator.propagate(state_initial, z_start, z_end, save_trajectory=False)
    
    # Validate output (filter NaN/Inf from numerical issues)
    max_retries = 3
    retry_count = 0
    while not np.all(np.isfinite(state_final)) and retry_count < max_retries:
        # Retry with different random parameters
        z_start = np.random.uniform(z_start_range[0], z_start_range[1])
        dz = np.random.uniform(dz_range[0], dz_range[1])
        z_end = z_start + dz
        state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
        state_final = integrator.propagate(state_initial, z_start, z_end, save_trajectory=False)
        retry_count += 1
    
    # If still fails, return zeros (will be filtered later)
    if not np.all(np.isfinite(state_final)):
        return np.zeros(6), np.zeros(4), 0.0, 0.0, 0.0
    
    # Prepare input: [x, y, tx, ty, qop, dz]
    input_state = np.array([
        state_initial[0],  # x
        state_initial[1],  # y
        state_initial[2],  # tx
        state_initial[3],  # ty
        state_initial[4],  # qop
        dz                 # Variable delta z!
    ])
    
    # Prepare output: [x, y, tx, ty] (qop is conserved)
    output_state = np.array([
        state_final[0],  # x
        state_final[1],  # y
        state_final[2],  # tx
        state_final[3],  # ty
    ])
    
    return input_state, output_state, momentum, z_start, z_end


def generate_dataset_v3(
    n_tracks: int,
    z_start_range: Tuple[float, float] = (0.0, 4000.0),
    dz_range: Tuple[float, float] = (500.0, 12000.0),
    p_range: Tuple[float, float] = (0.5, 100.0),
    step_size: float = 5.0,
    polarity: int = 1,
    n_workers: int = 8,
    use_real_field: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate V3 dataset with variable dz in parallel.
    
    Args:
        n_tracks: Number of tracks to generate
        z_start_range: Range for z_start as (min, max) in mm
        dz_range: Range for dz (propagation distance) as (min, max) in mm
        p_range: Momentum range (GeV) as (min, max)
        step_size: RK4 step size in mm
        polarity: Magnet polarity (+1 MagUp, -1 MagDown)
        n_workers: Number of parallel workers
        use_real_field: If True, use interpolated field map
        
    Returns:
        X: Input states [n_tracks, 6] = [x, y, tx, ty, qop, dz]
        Y: Output states [n_tracks, 4] = [x, y, tx, ty]
        P: Momenta [n_tracks] in GeV
        Z_start: Starting z positions [n_tracks]
        Z_end: Ending z positions [n_tracks]
    """
    field_type = "REAL FIELD MAP" if use_real_field else "GAUSSIAN APPROX"
    
    print(f"\n{'='*70}")
    print(f"V3 Data Generation: Variable dz Training Data")
    print(f"{'='*70}")
    print(f"  Tracks to generate: {n_tracks:,}")
    print(f"  z_start range: {z_start_range[0]:.0f} - {z_start_range[1]:.0f} mm")
    print(f"  dz range: {dz_range[0]:.0f} - {dz_range[1]:.0f} mm")
    print(f"  Momentum range: {p_range[0]:.1f} - {p_range[1]:.1f} GeV")
    print(f"  RK4 step size: {step_size} mm")
    print(f"  Field: {field_type}, polarity={'MagUp' if polarity > 0 else 'MagDown'}")
    print(f"  Workers: {n_workers}")
    print(f"{'='*70}\n")
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, z_start_range, dz_range, p_range)
        for i in range(n_tracks)
    ]
    
    # Generate in parallel
    start_time = time.time()
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(use_real_field, polarity, step_size)) as pool:
        # Use chunksize for better performance with large datasets
        chunksize = max(1000, n_tracks // (n_workers * 100))
        results = pool.map(generate_single_track_variable_dz, args_list, chunksize=chunksize)
    
    elapsed = time.time() - start_time
    
    # Unpack results
    X = np.array([r[0] for r in results])
    Y = np.array([r[1] for r in results])
    P = np.array([r[2] for r in results])
    Z_start = np.array([r[3] for r in results])
    Z_end = np.array([r[4] for r in results])
    
    # Filter out failed tracks (momentum = 0 indicates failure)
    valid_mask = P > 0
    n_failed = np.sum(~valid_mask)
    
    if n_failed > 0:
        print(f"  WARNING: {n_failed:,}/{n_tracks:,} tracks failed (numerical instability)")
        X = X[valid_mask]
        Y = Y[valid_mask]
        P = P[valid_mask]
        Z_start = Z_start[valid_mask]
        Z_end = Z_end[valid_mask]
    
    n_valid = len(P)
    print(f"\n  Generated {n_valid:,} valid tracks in {elapsed:.1f}s")
    print(f"  Rate: {n_valid/elapsed:.0f} tracks/s")
    
    # Print statistics
    print(f"\n  Input Statistics (X):")
    names = ['x', 'y', 'tx', 'ty', 'qop', 'dz']
    for i, name in enumerate(names):
        print(f"    {name:>4}: mean={X[:,i].mean():12.4e}, std={X[:,i].std():12.4e}, "
              f"min={X[:,i].min():12.4e}, max={X[:,i].max():12.4e}")
    
    print(f"\n  Output Statistics (Y):")
    names_out = ['x', 'y', 'tx', 'ty']
    for i, name in enumerate(names_out):
        print(f"    {name:>4}: mean={Y[:,i].mean():12.4e}, std={Y[:,i].std():12.4e}, "
              f"min={Y[:,i].min():12.4e}, max={Y[:,i].max():12.4e}")
    
    print(f"\n  Momentum: mean={P.mean():.2f} GeV, std={P.std():.2f} GeV")
    print(f"  dz: mean={X[:,5].mean():.0f} mm, std={X[:,5].std():.0f} mm")
    
    return X, Y, P, Z_start, Z_end


def save_dataset(
    output_path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    Z_start: np.ndarray,
    Z_end: np.ndarray,
    metadata: dict
):
    """Save dataset to NPZ file with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main data
    np.savez_compressed(
        output_path,
        X=X.astype(np.float32),  # Input features
        Y=Y.astype(np.float32),  # Output features
        P=P.astype(np.float32),  # Momentum
        z_start=Z_start.astype(np.float32),  # Starting z
        z_end=Z_end.astype(np.float32),      # Ending z
    )
    
    # Save metadata alongside
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved to: {output_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate V3 training data with variable dz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    parser.add_argument('--n_samples', type=int, default=10_000_000,
                        help='Number of track samples to generate')
    parser.add_argument('--output', type=str, default='data/training_v3.npz',
                        help='Output file path')
    
    # z range parameters
    parser.add_argument('--z_start_min', type=float, default=0.0,
                        help='Minimum z_start in mm')
    parser.add_argument('--z_start_max', type=float, default=4000.0,
                        help='Maximum z_start in mm')
    parser.add_argument('--dz_min', type=float, default=500.0,
                        help='Minimum propagation distance dz in mm')
    parser.add_argument('--dz_max', type=float, default=12000.0,
                        help='Maximum propagation distance dz in mm')
    
    # Physics parameters
    parser.add_argument('--p_min', type=float, default=0.5,
                        help='Minimum momentum in GeV')
    parser.add_argument('--p_max', type=float, default=100.0,
                        help='Maximum momentum in GeV')
    parser.add_argument('--step_size', type=float, default=5.0,
                        help='RK4 integration step size in mm')
    parser.add_argument('--polarity', type=int, default=1, choices=[1, -1],
                        help='Magnet polarity: 1=MagUp, -1=MagDown')
    
    # Field model
    parser.add_argument('--gaussian_field', action='store_true',
                        help='Use Gaussian approximation instead of real field map')
    
    # Parallelization
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel workers')
    
    # Batch processing (for very large datasets)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Process in batches (for memory efficiency)')
    
    args = parser.parse_args()
    
    # Generate dataset
    z_start_range = (args.z_start_min, args.z_start_max)
    dz_range = (args.dz_min, args.dz_max)
    p_range = (args.p_min, args.p_max)
    
    if args.batch_size and args.n_samples > args.batch_size:
        # Generate in batches for very large datasets
        n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
        print(f"\nGenerating {args.n_samples:,} samples in {n_batches} batches...")
        
        all_X, all_Y, all_P, all_Zs, all_Ze = [], [], [], [], []
        
        for batch_idx in range(n_batches):
            n_this_batch = min(args.batch_size, args.n_samples - batch_idx * args.batch_size)
            print(f"\n--- Batch {batch_idx + 1}/{n_batches} ({n_this_batch:,} samples) ---")
            
            X, Y, P, Z_start, Z_end = generate_dataset_v3(
                n_tracks=n_this_batch,
                z_start_range=z_start_range,
                dz_range=dz_range,
                p_range=p_range,
                step_size=args.step_size,
                polarity=args.polarity,
                n_workers=args.n_workers,
                use_real_field=not args.gaussian_field
            )
            
            all_X.append(X)
            all_Y.append(Y)
            all_P.append(P)
            all_Zs.append(Z_start)
            all_Ze.append(Z_end)
        
        X = np.concatenate(all_X)
        Y = np.concatenate(all_Y)
        P = np.concatenate(all_P)
        Z_start = np.concatenate(all_Zs)
        Z_end = np.concatenate(all_Ze)
        
    else:
        X, Y, P, Z_start, Z_end = generate_dataset_v3(
            n_tracks=args.n_samples,
            z_start_range=z_start_range,
            dz_range=dz_range,
            p_range=p_range,
            step_size=args.step_size,
            polarity=args.polarity,
            n_workers=args.n_workers,
            use_real_field=not args.gaussian_field
        )
    
    # Metadata
    metadata = {
        'version': 'V3',
        'description': 'Variable dz training data for ML track extrapolation',
        'n_samples': len(X),
        'z_start_range': list(z_start_range),
        'dz_range': list(dz_range),
        'p_range': list(p_range),
        'step_size': args.step_size,
        'polarity': args.polarity,
        'field_model': 'gaussian' if args.gaussian_field else 'real_fieldmap',
        'input_features': ['x', 'y', 'tx', 'ty', 'qop', 'dz'],
        'output_features': ['x', 'y', 'tx', 'ty'],
        'input_stats': {
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist(),
        },
        'output_stats': {
            'mean': Y.mean(axis=0).tolist(),
            'std': Y.std(axis=0).tolist(),
        },
    }
    
    # Save
    save_dataset(Path(args.output), X, Y, P, Z_start, Z_end, metadata)
    
    print(f"\n{'='*70}")
    print("V3 Data Generation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
