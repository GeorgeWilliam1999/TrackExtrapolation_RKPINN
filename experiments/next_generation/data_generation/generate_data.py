#!/usr/bin/env python3
"""
Generate Training Data for Track Extrapolation

Uses pure Python RK4 integrator to generate ground truth extrapolations.
This script creates datasets for training ML models.

Supports two output modes:
1. Endpoint only: X (input state) → Y (final state) - compact for MLP/PINN training
2. Full trajectory: T (all states along path) - for analysis and visualization

Based on: legacy code but cleaned up for next-generation experiments
Author: G. Scriven
Date: 2025-01-14
Updated: 2026-01-19 (added trajectory saving option)
"""

import numpy as np
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List, Optional
import sys
import time

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils' / 'archived'))
from rk4_propagator import LHCbMagneticField, RK4Integrator, generate_random_track


def generate_single_track(args: Tuple) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Generate single track propagation (for parallel processing).
    
    Args:
        args: (track_id, z_start, z_end, p_range, step_size, polarity, save_trajectory)
        
    Returns:
        input_state: [x, y, tx, ty, qop, dz] at z_start
        output_state: [x, y, tx, ty] at z_end
        momentum: Track momentum in GeV
        trajectory: If save_trajectory=True, array of shape (n_steps, 6) = [z, x, y, tx, ty, qop]
                   If save_trajectory=False, None
    """
    track_id, z_start, z_end, p_range, step_size, polarity, save_trajectory = args
    
    # Setup field and integrator
    field = LHCbMagneticField()
    field.params.polarity = polarity
    integrator = RK4Integrator(field, step_size=step_size)
    
    # Generate random initial state
    state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
    
    # Propagate to z_end
    result = integrator.propagate(state_initial, z_start, z_end, save_trajectory=save_trajectory)
    
    if save_trajectory:
        trajectory = result
        if len(trajectory) < 2:
            # Track failed early - retry
            state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
            trajectory = integrator.propagate(state_initial, z_start, z_end, save_trajectory=True)
            if len(trajectory) < 2:
                return np.zeros(6), np.zeros(4), 0.0, None
        state_final = trajectory[-1, 1:]  # Last row, skip z column
    else:
        state_final = result
        trajectory = None
    
    # Validate output (filter NaN/Inf from numerical issues)
    if not np.all(np.isfinite(state_final)):
        # Retry with different random parameters
        state_initial, momentum = generate_random_track(p_range=p_range, z_start=z_start)
        result = integrator.propagate(state_initial, z_start, z_end, save_trajectory=save_trajectory)
        
        if save_trajectory:
            trajectory = result
            if len(trajectory) < 2:
                return np.zeros(6), np.zeros(4), 0.0, None
            state_final = trajectory[-1, 1:]
        else:
            state_final = result
            trajectory = None
        
        # If still fails, skip this track (return zeros to filter later)
        if not np.all(np.isfinite(state_final)):
            return np.zeros(6), np.zeros(4), 0.0, None
    
    # Prepare input: [x, y, tx, ty, qop, dz]
    dz = z_end - z_start
    input_state = np.array([
        state_initial[0],  # x
        state_initial[1],  # y
        state_initial[2],  # tx
        state_initial[3],  # ty
        state_initial[4],  # qop
        dz                 # delta z
    ])
    
    # Prepare output: [x, y, tx, ty] (only position and slopes, qop is conserved)
    output_state = np.array([
        state_final[0],  # x
        state_final[1],  # y
        state_final[2],  # tx
        state_final[3],  # ty
    ])
    
    return input_state, output_state, momentum, trajectory


def generate_dataset(n_tracks: int,
                     z_start: float = 4000.0,
                     z_end: float = 12000.0,
                     p_range: Tuple[float, float] = (0.5, 100.0),
                     step_size: float = 5.0,
                     polarity: int = 1,
                     n_workers: int = 8,
                     save_trajectories: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """
    Generate dataset of track propagations in parallel.
    
    Args:
        n_tracks: Number of tracks to generate
        z_start: Initial z position (mm)
        z_end: Final z position (mm)
        p_range: Momentum range (GeV) as (min, max)
        step_size: RK4 step size in mm (5mm recommended, 10mm for speed)
        polarity: Magnet polarity (+1 MagUp, -1 MagDown)
        n_workers: Number of parallel workers
        save_trajectories: If True, save full trajectory for each track
        
    Returns:
        X: Input states [n_tracks, 6] = [x, y, tx, ty, qop, dz]
        Y: Output states [n_tracks, 4] = [x, y, tx, ty]
        P: Momenta [n_tracks] in GeV
        T: List of trajectories (if save_trajectories=True), each is array of shape (n_steps, 6)
           where columns are [z, x, y, tx, ty, qop]. None if save_trajectories=False.
    """
    print(f"\nGenerating {n_tracks} tracks...")
    print(f"  z: {z_start}mm → {z_end}mm (Δz = {z_end - z_start}mm)")
    print(f"  Momentum range: {p_range[0]}-{p_range[1]} GeV")
    print(f"  RK4 step size: {step_size}mm")
    print(f"  Field polarity: {'MagUp' if polarity > 0 else 'MagDown'}")
    print(f"  Parallel workers: {n_workers}")
    print(f"  Save trajectories: {save_trajectories}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, z_start, z_end, p_range, step_size, polarity, save_trajectories)
        for i in range(n_tracks)
    ]
    
    # Generate in parallel
    start_time = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(generate_single_track, args_list)
    elapsed = time.time() - start_time
    
    # Unpack results
    X = np.array([r[0] for r in results])
    Y = np.array([r[1] for r in results])
    P = np.array([r[2] for r in results])
    
    if save_trajectories:
        T = [r[3] for r in results]
    else:
        T = None
    
    # Filter out failed tracks (momentum = 0 indicates failure)
    valid_mask = P > 0
    n_failed = np.sum(~valid_mask)
    if n_failed > 0:
        print(f"\n  WARNING: {n_failed}/{n_tracks} tracks failed (numerical instability)")
        print(f"  Keeping {np.sum(valid_mask)} valid tracks")
        X = X[valid_mask]
        Y = Y[valid_mask]
        P = P[valid_mask]
        if T is not None:
            T = [t for t, valid in zip(T, valid_mask) if valid]
    
    print(f"\n✓ Generated {len(P)} tracks in {elapsed:.1f}s ({n_tracks/elapsed:.0f} tracks/s)")
    print(f"\nDataset statistics:")
    print(f"  Input X shape: {X.shape}")
    print(f"  Output Y shape: {Y.shape}")
    print(f"  Momentum P shape: {P.shape}")
    if T is not None:
        n_steps_list = [len(t) for t in T if t is not None]
        print(f"  Trajectories: {len(T)} tracks, {np.mean(n_steps_list):.0f} avg steps")
    print(f"  Momentum range: {P.min():.2f} - {P.max():.2f} GeV")
    print(f"  Mean momentum: {P.mean():.2f} GeV")
    
    return X, Y, P, T


def save_dataset(X: np.ndarray, Y: np.ndarray, P: np.ndarray, 
                 output_dir: Path, name: str, 
                 trajectories: Optional[List[np.ndarray]] = None):
    """Save dataset to .npy files or .npz if name ends with .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If name ends with .npz, save as compressed archive (condor mode)
    if name.endswith('.npz'):
        output_path = output_dir / name
        
        if trajectories is not None:
            # Save trajectories as object array (variable length per track)
            # Convert list to object array for np.savez compatibility
            T_array = np.empty(len(trajectories), dtype=object)
            for i, t in enumerate(trajectories):
                T_array[i] = t
            np.savez_compressed(output_path, X=X, Y=Y, P=P, T=T_array)
        else:
            np.savez_compressed(output_path, X=X, Y=Y, P=P)
            
        print(f"\n✓ Saved dataset '{name}':")
        print(f"  Compressed archive: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024**2):.1f} MB")
        if trajectories is not None:
            print(f"  Includes full trajectories!")
    else:
        # Original mode: save separate .npy files
        X_path = output_dir / f"X_{name}.npy"
        Y_path = output_dir / f"Y_{name}.npy"
        P_path = output_dir / f"P_{name}.npy"
        
        np.save(X_path, X)
        np.save(Y_path, Y)
        np.save(P_path, P)
        
        total_size = X.nbytes + Y.nbytes + P.nbytes
        print(f"\n✓ Saved dataset '{name}':")
        print(f"  {X_path}")
        print(f"  {Y_path}")
        print(f"  {P_path}")
        
        if trajectories is not None:
            T_path = output_dir / f"T_{name}.npy"
            T_array = np.empty(len(trajectories), dtype=object)
            for i, t in enumerate(trajectories):
                T_array[i] = t
            np.save(T_path, T_array, allow_pickle=True)
            print(f"  {T_path}")
            
        print(f"  Total size: {total_size / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Generate track extrapolation training data")
    
    # Dataset parameters
    parser.add_argument('--n-tracks', type=int, default=10000,
                       help='Number of tracks to generate (default: 10000)')
    parser.add_argument('--name', type=str, default='train',
                       help='Dataset name (default: train)')
    parser.add_argument('--output-dir', type=str, default='../data_generation/datasets',
                       help='Output directory')
    
    # Physics parameters
    parser.add_argument('--z-start', type=float, default=4000.0,
                       help='Initial z position in mm (default: 4000)')
    parser.add_argument('--z-end', type=float, default=12000.0,
                       help='Final z position in mm (default: 12000)')
    parser.add_argument('--p-min', type=float, default=3.0,
                       help='Minimum momentum in GeV (default: 3.0 for stable 8km propagation)')
    parser.add_argument('--p-max', type=float, default=100.0,
                       help='Maximum momentum in GeV (default: 100)')
    parser.add_argument('--step-size', type=float, default=5.0,
                       help='RK4 step size in mm (default: 5.0 mm for fast generation)')
    parser.add_argument('--polarity', type=int, choices=[-1, 1], default=-1,
                       help='Magnet polarity: +1=MagUp, -1=MagDown (default: -1 from fit)')
    
    # Trajectory saving
    parser.add_argument('--save-trajectories', action='store_true',
                       help='Save full trajectories (all intermediate states)')
    
    # Computational parameters
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("Track Extrapolation Data Generation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Dataset: {args.name}")
    print(f"  Output: {args.output_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Save trajectories: {args.save_trajectories}")
    
    # Generate dataset
    X, Y, P, T = generate_dataset(
        n_tracks=args.n_tracks,
        z_start=args.z_start,
        z_end=args.z_end,
        p_range=(args.p_min, args.p_max),
        step_size=args.step_size,
        polarity=args.polarity,
        n_workers=args.workers,
        save_trajectories=args.save_trajectories
    )
    
    # Save
    output_dir = Path(args.output_dir)
    save_dataset(X, Y, P, output_dir, args.name, trajectories=T)
    
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Generate validation set: --name val --n-tracks 5000")
    print("  2. Generate test set: --name test --n-tracks 5000")
    print("  3. Train model: python ../training/train_model.py")


if __name__ == "__main__":
    main()
