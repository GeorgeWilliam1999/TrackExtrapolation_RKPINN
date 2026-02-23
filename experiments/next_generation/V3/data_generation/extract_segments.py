#!/usr/bin/env python3
"""
V3 Segment Extraction: Extract variable dz training samples from full trajectories.

This is the key data generation script for V3:
1. Load full trajectories (generated with 5mm RK4 steps)
2. Extract random segments with variable dz
3. Optionally include supervised collocation points for PINN

From ONE trajectory, we can extract O(N²) training samples!
This is much more efficient than regenerating data for each dz.

Usage:
    # MLP training data (endpoint only)
    python extract_segments.py \
        --input trajectories_10k.npz \
        --n_samples 100000000 \
        --dz_min 500 --dz_max 12000 \
        --output training_mlp_v3.npz

    # PINN training data (with supervised collocation)
    python extract_segments.py \
        --input trajectories_10k.npz \
        --n_samples 10000000 \
        --dz_min 500 --dz_max 12000 \
        --collocation_points 10 \
        --output training_pinn_v3.npz

Author: G. Scriven
Date: January 2026
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Optional
import time


def load_trajectories(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectories from NPZ file.
    
    Returns:
        trajectories: Object array of shape (n_traj,) containing arrays of shape (n_steps, 6)
                      Each trajectory point is [z, x, y, tx, ty, qop]
        momenta: Array of shape (n_traj,) with momentum for each track
    """
    data = np.load(path, allow_pickle=True)
    trajectories = data['T']
    momenta = data['P'] if 'P' in data else np.ones(len(trajectories))
    return trajectories, momenta


def extract_mlp_samples(
    trajectories: np.ndarray,
    n_samples: int,
    dz_min: float = 500.0,
    dz_max: float = 12000.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MLP training samples (endpoint only).
    
    Args:
        trajectories: Array of trajectories, each shape (n_steps, 6) = [z, x, y, tx, ty, qop]
        n_samples: Number of samples to extract
        dz_min: Minimum propagation distance
        dz_max: Maximum propagation distance
        seed: Random seed
        
    Returns:
        X: Input features [n_samples, 6] = [x, y, tx, ty, qop, dz]
        Y: Output features [n_samples, 4] = [x, y, tx, ty]
    """
    np.random.seed(seed)
    
    # Pre-compute trajectory lengths and step sizes
    traj_lengths = np.array([len(t) for t in trajectories])
    n_traj = len(trajectories)
    
    # Get typical step size (assume uniform, get from first trajectory)
    step_size = trajectories[0][1, 0] - trajectories[0][0, 0]  # z[1] - z[0]
    
    # Min/max steps for dz range
    min_steps = max(1, int(dz_min / step_size))
    max_steps = int(dz_max / step_size)
    
    print(f"Step size: {step_size:.1f} mm")
    print(f"dz range: [{dz_min}, {dz_max}] mm → [{min_steps}, {max_steps}] steps")
    
    # Allocate output arrays
    X = np.zeros((n_samples, 6), dtype=np.float32)
    Y = np.zeros((n_samples, 4), dtype=np.float32)
    
    valid_count = 0
    attempts = 0
    max_attempts = n_samples * 3
    
    print(f"\nExtracting {n_samples:,} samples...")
    report_interval = n_samples // 10
    
    while valid_count < n_samples and attempts < max_attempts:
        # Random trajectory
        traj_idx = np.random.randint(n_traj)
        traj = trajectories[traj_idx]
        n_points = len(traj)
        
        # Need at least min_steps points
        if n_points < min_steps + 1:
            attempts += 1
            continue
        
        # Random start index (leave room for at least min_steps)
        max_start = n_points - min_steps - 1
        if max_start < 0:
            attempts += 1
            continue
        start_idx = np.random.randint(0, max_start + 1)
        
        # Random end index (at least min_steps away, at most max_steps)
        min_end = start_idx + min_steps
        max_end = min(n_points - 1, start_idx + max_steps)
        if min_end > max_end:
            attempts += 1
            continue
        end_idx = np.random.randint(min_end, max_end + 1)
        
        # Extract states
        # Trajectory format: [z, x, y, tx, ty, qop]
        state_start = traj[start_idx]  # [z, x, y, tx, ty, qop]
        state_end = traj[end_idx]
        
        z_start = state_start[0]
        z_end = state_end[0]
        dz = z_end - z_start
        qop = state_start[5]
        
        # Build input: [x, y, tx, ty, qop, dz]
        X[valid_count, 0] = state_start[1]  # x
        X[valid_count, 1] = state_start[2]  # y
        X[valid_count, 2] = state_start[3]  # tx
        X[valid_count, 3] = state_start[4]  # ty
        X[valid_count, 4] = qop
        X[valid_count, 5] = dz
        
        # Build output: [x, y, tx, ty]
        Y[valid_count, 0] = state_end[1]  # x
        Y[valid_count, 1] = state_end[2]  # y
        Y[valid_count, 2] = state_end[3]  # tx
        Y[valid_count, 3] = state_end[4]  # ty
        
        valid_count += 1
        attempts += 1
        
        if valid_count % report_interval == 0:
            print(f"  {valid_count:,}/{n_samples:,} ({100*valid_count/n_samples:.0f}%)")
    
    if valid_count < n_samples:
        print(f"WARNING: Only extracted {valid_count:,}/{n_samples:,} samples")
        X = X[:valid_count]
        Y = Y[:valid_count]
    
    return X, Y


def extract_pinn_samples(
    trajectories: np.ndarray,
    n_samples: int,
    n_collocation: int = 10,
    dz_min: float = 500.0,
    dz_max: float = 12000.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract PINN training samples with supervised collocation.
    
    This is the key innovation: we have ground truth at collocation points!
    
    Args:
        trajectories: Array of trajectories
        n_samples: Number of samples to extract
        n_collocation: Number of collocation points per sample
        dz_min: Minimum propagation distance
        dz_max: Maximum propagation distance
        seed: Random seed
        
    Returns:
        X: Input features [n_samples, 6] = [x, y, tx, ty, qop, dz]
        Y: Output features [n_samples, 4] = [x, y, tx, ty] at endpoint
        z_frac: Collocation z_frac values [n_samples, n_collocation]
        Y_col: TRUE states at collocation [n_samples, n_collocation, 4]
    """
    np.random.seed(seed)
    
    # Pre-compute trajectory info
    n_traj = len(trajectories)
    step_size = trajectories[0][1, 0] - trajectories[0][0, 0]
    
    min_steps = max(1, int(dz_min / step_size))
    max_steps = int(dz_max / step_size)
    
    # Need enough steps for meaningful collocation
    min_steps = max(min_steps, n_collocation + 1)
    
    print(f"Step size: {step_size:.1f} mm")
    print(f"dz range: [{dz_min}, {dz_max}] mm → [{min_steps}, {max_steps}] steps")
    print(f"Collocation points per sample: {n_collocation}")
    
    # Allocate output arrays
    X = np.zeros((n_samples, 6), dtype=np.float32)
    Y = np.zeros((n_samples, 4), dtype=np.float32)
    z_frac = np.zeros((n_samples, n_collocation), dtype=np.float32)
    Y_col = np.zeros((n_samples, n_collocation, 4), dtype=np.float32)
    
    valid_count = 0
    attempts = 0
    max_attempts = n_samples * 3
    
    print(f"\nExtracting {n_samples:,} PINN samples with {n_collocation} collocation points each...")
    report_interval = max(1, n_samples // 10)
    
    while valid_count < n_samples and attempts < max_attempts:
        # Random trajectory
        traj_idx = np.random.randint(n_traj)
        traj = trajectories[traj_idx]
        n_points = len(traj)
        
        if n_points < min_steps + 1:
            attempts += 1
            continue
        
        # Random start/end
        max_start = n_points - min_steps - 1
        if max_start < 0:
            attempts += 1
            continue
        start_idx = np.random.randint(0, max_start + 1)
        
        min_end = start_idx + min_steps
        max_end = min(n_points - 1, start_idx + max_steps)
        if min_end > max_end:
            attempts += 1
            continue
        end_idx = np.random.randint(min_end, max_end + 1)
        
        # Extract segment info
        state_start = traj[start_idx]
        state_end = traj[end_idx]
        dz = state_end[0] - state_start[0]
        qop = state_start[5]
        
        # Build input: [x, y, tx, ty, qop, dz]
        X[valid_count, 0] = state_start[1]
        X[valid_count, 1] = state_start[2]
        X[valid_count, 2] = state_start[3]
        X[valid_count, 3] = state_start[4]
        X[valid_count, 4] = qop
        X[valid_count, 5] = dz
        
        # Build output: [x, y, tx, ty] at endpoint
        Y[valid_count, 0] = state_end[1]
        Y[valid_count, 1] = state_end[2]
        Y[valid_count, 2] = state_end[3]
        Y[valid_count, 3] = state_end[4]
        
        # Select collocation points: uniformly spaced in z_frac
        # Exclude 0 and 1 (IC and endpoint handled separately)
        col_fracs = np.linspace(0.1, 0.9, n_collocation)
        z_frac[valid_count] = col_fracs
        
        # Get TRUE states at collocation points from trajectory
        segment_length = end_idx - start_idx
        for i, frac in enumerate(col_fracs):
            # Map z_frac to trajectory index
            col_idx = start_idx + int(frac * segment_length)
            col_idx = min(col_idx, end_idx)  # Safety clamp
            col_state = traj[col_idx]
            Y_col[valid_count, i, 0] = col_state[1]  # x
            Y_col[valid_count, i, 1] = col_state[2]  # y
            Y_col[valid_count, i, 2] = col_state[3]  # tx
            Y_col[valid_count, i, 3] = col_state[4]  # ty
        
        valid_count += 1
        attempts += 1
        
        if valid_count % report_interval == 0:
            print(f"  {valid_count:,}/{n_samples:,} ({100*valid_count/n_samples:.0f}%)")
    
    if valid_count < n_samples:
        print(f"WARNING: Only extracted {valid_count:,}/{n_samples:,} samples")
        X = X[:valid_count]
        Y = Y[:valid_count]
        z_frac = z_frac[:valid_count]
        Y_col = Y_col[:valid_count]
    
    return X, Y, z_frac, Y_col


def compute_statistics(X: np.ndarray, Y: np.ndarray):
    """Compute and print dataset statistics."""
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    feature_names_in = ['x', 'y', 'tx', 'ty', 'qop', 'dz']
    feature_names_out = ['x', 'y', 'tx', 'ty']
    
    print("\nInput Features:")
    print(f"  {'Feature':<8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-"*56)
    for i, name in enumerate(feature_names_in):
        print(f"  {name:<8} {X[:,i].mean():>12.4f} {X[:,i].std():>12.4f} "
              f"{X[:,i].min():>12.4f} {X[:,i].max():>12.4f}")
    
    print("\nOutput Features:")
    print(f"  {'Feature':<8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-"*56)
    for i, name in enumerate(feature_names_out):
        print(f"  {name:<8} {Y[:,i].mean():>12.4f} {Y[:,i].std():>12.4f} "
              f"{Y[:,i].min():>12.4f} {Y[:,i].max():>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract variable dz training samples from trajectories"
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input trajectories file (NPZ)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output training data file (NPZ)')
    parser.add_argument('--n_samples', type=int, default=10000000,
                       help='Number of samples to extract (default: 10M)')
    parser.add_argument('--dz_min', type=float, default=500.0,
                       help='Minimum dz in mm (default: 500)')
    parser.add_argument('--dz_max', type=float, default=12000.0,
                       help='Maximum dz in mm (default: 12000)')
    parser.add_argument('--collocation_points', type=int, default=0,
                       help='Number of collocation points for PINN (0 = MLP mode)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("V3 Segment Extraction")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.n_samples:,}")
    print(f"dz range: [{args.dz_min}, {args.dz_max}] mm")
    print(f"Mode: {'PINN (supervised collocation)' if args.collocation_points > 0 else 'MLP (endpoint only)'}")
    if args.collocation_points > 0:
        print(f"Collocation points: {args.collocation_points}")
    
    # Load trajectories
    print(f"\nLoading trajectories from {args.input}...")
    start_time = time.time()
    trajectories, momenta = load_trajectories(args.input)
    print(f"  Loaded {len(trajectories):,} trajectories in {time.time()-start_time:.1f}s")
    
    # Extract samples
    start_time = time.time()
    
    if args.collocation_points > 0:
        # PINN mode with supervised collocation
        X, Y, z_frac, Y_col = extract_pinn_samples(
            trajectories,
            n_samples=args.n_samples,
            n_collocation=args.collocation_points,
            dz_min=args.dz_min,
            dz_max=args.dz_max,
            seed=args.seed
        )
        
        # Save with collocation data
        np.savez_compressed(
            args.output,
            X=X, Y=Y, z_frac=z_frac, Y_col=Y_col
        )
    else:
        # MLP mode (endpoint only)
        X, Y = extract_mlp_samples(
            trajectories,
            n_samples=args.n_samples,
            dz_min=args.dz_min,
            dz_max=args.dz_max,
            seed=args.seed
        )
        
        # Save without collocation
        np.savez_compressed(args.output, X=X, Y=Y)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Extracted {len(X):,} samples in {elapsed:.1f}s ({len(X)/elapsed:.0f} samples/s)")
    
    # Statistics
    compute_statistics(X, Y)
    
    # File info
    output_path = Path(args.output)
    print(f"\n✓ Saved to {args.output}")
    print(f"  File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
