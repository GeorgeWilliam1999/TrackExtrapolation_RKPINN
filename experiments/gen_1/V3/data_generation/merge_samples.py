#!/usr/bin/env python3
"""
Merge MLP sample chunks from distributed extraction.

Usage:
    python merge_samples.py --input_dir data/chunks --pattern "mlp_samples_*.npz" --output data/training_mlp_v3.npz
"""

import numpy as np
import argparse
from pathlib import Path


def merge_samples(input_dir: str, output_path: str, pattern: str = "mlp_samples_*.npz"):
    """Merge sample chunks into single file."""
    
    input_path = Path(input_dir)
    files = sorted(input_path.glob(pattern))
    
    print(f"Found {len(files)} sample chunks")
    
    all_X = []
    all_Y = []
    
    for f in files:
        data = np.load(f)
        X = data['X']
        Y = data['Y']
        
        all_X.append(X)
        all_Y.append(Y)
        
        print(f"  Loaded {f.name}: {len(X):,} samples")
    
    # Concatenate
    X_merged = np.concatenate(all_X, axis=0)
    Y_merged = np.concatenate(all_Y, axis=0)
    
    print(f"\nTotal samples: {len(X_merged):,}")
    print(f"Saving to {output_path}...")
    
    np.savez_compressed(output_path, X=X_merged, Y=Y_merged)
    
    print(f"Done! File size: {Path(output_path).stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/chunks')
    parser.add_argument('--pattern', type=str, default='mlp_samples_*.npz')
    parser.add_argument('--output', type=str, default='data/training_mlp_v3.npz')
    args = parser.parse_args()
    
    merge_samples(args.input_dir, args.output, args.pattern)
