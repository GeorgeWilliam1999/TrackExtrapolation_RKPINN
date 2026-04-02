#!/usr/bin/env python3
"""
Merge multiple batch .npz files into a single dataset.

Combines output from condor job array (batch_000.npz, batch_001.npz, ...)
into one consolidated dataset.

Usage:
    python merge_batches.py --input "datasets/batch_*.npz" --output datasets/train_50M.npz
    python merge_batches.py --input "datasets/*.npz" --output merged.npz --verify

Author: G. Scriven
Date: March 2026
"""

import numpy as np
from pathlib import Path
import argparse
import glob


def merge_batches(input_pattern: str, verify: bool = False):
    """Merge all batches matching pattern."""
    batch_files = sorted(glob.glob(input_pattern))

    if not batch_files:
        raise FileNotFoundError(f"No files found matching: {input_pattern}")

    print(f"Found {len(batch_files)} batch files")
    print(f"  First: {Path(batch_files[0]).name}")
    print(f"  Last:  {Path(batch_files[-1]).name}\n")

    X_list, Y_list, P_list, Z_list = [], [], [], []

    for i, filepath in enumerate(batch_files):
        print(f"  Loading {i+1}/{len(batch_files)}: {Path(filepath).name}", end='\r')
        data = np.load(filepath, allow_pickle=True)
        X_list.append(data['X'])
        Y_list.append(data['Y'])
        P_list.append(data['P'])
        if 'Z' in data:
            Z_list.append(data['Z'])

        if verify:
            if np.any(~np.isfinite(data['X'])):
                print(f"\n  WARNING: Non-finite values in {filepath} X array")
            if np.any(~np.isfinite(data['Y'])):
                print(f"\n  WARNING: Non-finite values in {filepath} Y array")

    print()

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    P = np.concatenate(P_list, axis=0)
    Z = np.concatenate(Z_list, axis=0) if Z_list else None

    print(f"Total dataset:")
    print(f"  X: {X.shape} ({X.nbytes / (1024**2):.1f} MB)")
    print(f"  Y: {Y.shape} ({Y.nbytes / (1024**2):.1f} MB)")
    print(f"  P: {P.shape} ({P.nbytes / (1024**2):.1f} MB)")
    if Z is not None:
        print(f"  Z: {Z.shape} ({Z.nbytes / (1024**2):.1f} MB)")
    total_mb = (X.nbytes + Y.nbytes + P.nbytes) / (1024**2)
    if Z is not None:
        total_mb += Z.nbytes / (1024**2)
    print(f"  Total: {total_mb:.1f} MB (uncompressed)")

    return X, Y, P, Z


def main():
    parser = argparse.ArgumentParser(description='Merge batch .npz files')
    parser.add_argument('--input', type=str, required=True,
                        help='Glob pattern (e.g., "datasets/batch_*.npz")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file (e.g., datasets/train_50M.npz)')
    parser.add_argument('--verify', action='store_true',
                        help='Check for NaN/Inf values')

    args = parser.parse_args()

    print("=" * 70)
    print("Batch Dataset Merger")
    print("=" * 70 + "\n")

    X, Y, P, Z = merge_batches(args.input, verify=args.verify)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    if Z is not None:
        np.savez_compressed(output_path, X=X, Y=Y, P=P, Z=Z)
    else:
        np.savez_compressed(output_path, X=X, Y=Y, P=P)

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Size: {size_mb:.1f} MB (compressed)")
    print(f"\n{'='*70}\nDONE!\n{'='*70}")


if __name__ == '__main__':
    main()
