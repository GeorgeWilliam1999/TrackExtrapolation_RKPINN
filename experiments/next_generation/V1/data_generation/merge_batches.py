#!/usr/bin/env python3
"""
Merge multiple batch .npz files into single dataset

This script combines output from condor job array into one consolidated dataset.
Useful after running submit_condor.sub which generates batch_000.npz, batch_001.npz, etc.

Supports both endpoint-only data (X, Y, P) and full trajectory data (X, Y, P, T).

Usage:
    python merge_batches.py --input "data/batch_*.npz" --output full_dataset_1M.npz
    python merge_batches.py --input data/*.npz --output merged.npz --verify

Author: G. Scriven
Date: 2025-01-14
Updated: 2026-01-19 (added trajectory support)
"""

import numpy as np
from pathlib import Path
import argparse
import glob
from typing import List, Tuple, Optional


def load_batch(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load single batch file, handling both with and without trajectories."""
    data = np.load(filepath, allow_pickle=True)
    X = data['X']
    Y = data['Y']
    P = data['P']
    T = data['T'] if 'T' in data else None
    return X, Y, P, T


def merge_batches(input_pattern: str, verify: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """
    Merge all batches matching pattern.
    
    Args:
        input_pattern: Glob pattern like "data/batch_*.npz"
        verify: If True, check for NaN/Inf values
        
    Returns:
        Combined (X, Y, P, T) where T is list of trajectories or None
    """
    # Find all matching files
    batch_files = sorted(glob.glob(input_pattern))
    
    if not batch_files:
        raise FileNotFoundError(f"No files found matching: {input_pattern}")
    
    print(f"Found {len(batch_files)} batch files")
    print(f"  First: {Path(batch_files[0]).name}")
    print(f"  Last: {Path(batch_files[-1]).name}")
    print()
    
    # Load all batches
    X_list, Y_list, P_list = [], [], []
    T_list = []
    has_trajectories = None
    
    for i, filepath in enumerate(batch_files):
        print(f"Loading batch {i+1}/{len(batch_files)}: {Path(filepath).name}", end='\r')
        X_batch, Y_batch, P_batch, T_batch = load_batch(Path(filepath))
        
        # Check trajectory consistency
        batch_has_traj = T_batch is not None
        if has_trajectories is None:
            has_trajectories = batch_has_traj
        elif has_trajectories != batch_has_traj:
            print(f"\nWARNING: Inconsistent trajectory data in {filepath}")
        
        if verify:
            # Check for invalid values
            if np.any(~np.isfinite(X_batch)):
                print(f"\nWARNING: Non-finite values in {filepath} X array")
            if np.any(~np.isfinite(Y_batch)):
                print(f"\nWARNING: Non-finite values in {filepath} Y array")
            if np.any(~np.isfinite(P_batch)):
                print(f"\nWARNING: Non-finite values in {filepath} P array")
        
        X_list.append(X_batch)
        Y_list.append(Y_batch)
        P_list.append(P_batch)
        
        if T_batch is not None:
            # T_batch is object array, convert to list
            T_list.extend(list(T_batch))
    
    print()  # Newline after progress
    
    # Concatenate all batches
    print("Concatenating arrays...")
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    P = np.concatenate(P_list, axis=0)
    
    print(f"Total dataset size:")
    print(f"  X: {X.shape} ({X.nbytes / (1024**2):.1f} MB)")
    print(f"  Y: {Y.shape} ({Y.nbytes / (1024**2):.1f} MB)")
    print(f"  P: {P.shape} ({P.nbytes / (1024**2):.1f} MB)")
    print(f"  Total: {(X.nbytes + Y.nbytes + P.nbytes) / (1024**2):.1f} MB")
    
    if has_trajectories and T_list:
        n_steps = [len(t) for t in T_list if t is not None]
        print(f"  Trajectories: {len(T_list)} tracks, {np.mean(n_steps):.0f} avg steps")
        return X, Y, P, T_list
    
    return X, Y, P, None


def save_merged(X: np.ndarray, Y: np.ndarray, P: np.ndarray, output_path: Path,
                trajectories: Optional[List[np.ndarray]] = None):
    """Save merged dataset as compressed .npz."""
    print(f"\nSaving to {output_path}...")
    
    if trajectories is not None:
        # Convert list to object array
        T_array = np.empty(len(trajectories), dtype=object)
        for i, t in enumerate(trajectories):
            T_array[i] = t
        np.savez_compressed(output_path, X=X, Y=Y, P=P, T=T_array)
    else:
        np.savez_compressed(output_path, X=X, Y=Y, P=P)
    
    file_size = output_path.stat().st_size / (1024**2)
    compression_ratio = (X.nbytes + Y.nbytes + P.nbytes) / output_path.stat().st_size
    
    print(f"Saved successfully!")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    if trajectories is not None:
        print(f"  Includes {len(trajectories)} full trajectories")


def main():
    parser = argparse.ArgumentParser(
        description='Merge batch .npz files from condor jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all batches from condor run
  python merge_batches.py --input "data/batch_*.npz" --output full_dataset.npz
  
  # Merge with verification
  python merge_batches.py --input "data/*.npz" --output merged.npz --verify
  
  # Merge specific range
  python merge_batches.py --input "data/batch_{000..099}.npz" --output dataset_1M.npz
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Glob pattern for input files (e.g., "data/batch_*.npz")')
    parser.add_argument('--output', type=str, required=True,
                       help='Output merged file path (e.g., full_dataset.npz)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify data integrity (check for NaN/Inf)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Batch Dataset Merger")
    print("=" * 70)
    print()
    
    # Merge batches
    X, Y, P, T = merge_batches(args.input, verify=args.verify)
    
    # Save merged dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_merged(X, Y, P, output_path, trajectories=T)
    
    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
