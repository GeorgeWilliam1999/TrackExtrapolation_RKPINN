#!/usr/bin/env python3
"""
Merge trajectory chunks from distributed generation.

Usage:
    python merge_trajectories.py --input_dir data/chunks --output data/trajectories_10k.npz
"""

import numpy as np
import argparse
from pathlib import Path
import glob


def merge_trajectories(input_dir: str, output_path: str, pattern: str = "trajectories_*.npz"):
    """Merge trajectory chunks into single file."""
    
    input_path = Path(input_dir)
    files = sorted(input_path.glob(pattern))
    
    print(f"Found {len(files)} trajectory chunks")
    
    all_trajectories = []
    all_momenta = []
    
    for f in files:
        data = np.load(f, allow_pickle=True)
        T = data['T']
        P = data['P']
        
        all_trajectories.extend(T)
        all_momenta.extend(P)
        
        print(f"  Loaded {f.name}: {len(T)} trajectories")
    
    # Convert to arrays
    T_merged = np.empty(len(all_trajectories), dtype=object)
    for i, t in enumerate(all_trajectories):
        T_merged[i] = t
    P_merged = np.array(all_momenta)
    
    print(f"\nTotal trajectories: {len(T_merged)}")
    print(f"Saving to {output_path}...")
    
    np.savez_compressed(output_path, T=T_merged, P=P_merged)
    
    print(f"Done! File size: {Path(output_path).stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/chunks')
    parser.add_argument('--output', type=str, default='data/trajectories_10k.npz')
    args = parser.parse_args()
    
    merge_trajectories(args.input_dir, args.output)
