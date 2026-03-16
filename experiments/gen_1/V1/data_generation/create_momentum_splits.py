#!/usr/bin/env python3
"""
Create momentum-split training datasets from the main training file.

Splits the 50M track dataset into three momentum ranges:
- Low momentum:  0.5 - 5 GeV (most challenging, multiple scattering dominant)
- Mid momentum:  5 - 20 GeV (typical LHCb tracks)
- High momentum: 20 - 100 GeV (less bending, easier extrapolation)
"""

import numpy as np
from pathlib import Path

def create_momentum_splits(input_file: str, output_dir: str, max_per_split: int = 10_000_000):
    """
    Split training data by momentum range.
    
    Args:
        input_file: Path to training_50M.npz
        output_dir: Directory to save split files
        max_per_split: Maximum samples per momentum range (to keep balanced)
    """
    print(f"Loading data from {input_file}...")
    data = np.load(input_file)
    X = data['X']
    Y = data['Y']
    P = data['P']
    
    print(f"  Total samples: {len(P):,}")
    print(f"  Momentum range: {P.min():.2f} - {P.max():.2f} GeV")
    
    # Define momentum ranges
    splits = {
        'low_p': (0.5, 5.0),    # Multiple scattering dominant
        'mid_p': (5.0, 20.0),   # Typical LHCb range
        'high_p': (20.0, 100.0) # High momentum, small bending
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, (p_min, p_max) in splits.items():
        print(f"\nProcessing {name} ({p_min}-{p_max} GeV)...")
        
        # Select tracks in momentum range
        mask = (P >= p_min) & (P < p_max)
        n_available = mask.sum()
        print(f"  Available samples: {n_available:,}")
        
        # Subsample if too many
        if n_available > max_per_split:
            indices = np.where(mask)[0]
            np.random.seed(42)  # Reproducible
            selected = np.random.choice(indices, max_per_split, replace=False)
            selected.sort()
        else:
            selected = np.where(mask)[0]
        
        X_split = X[selected]
        Y_split = Y[selected]
        P_split = P[selected]
        
        # Save
        output_file = output_path / f"training_{name}.npz"
        np.savez_compressed(
            output_file,
            X=X_split,
            Y=Y_split,
            P=P_split
        )
        
        file_size = output_file.stat().st_size / 1e6
        print(f"  Saved {len(selected):,} samples to {output_file}")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Momentum stats: mean={P_split.mean():.2f}, std={P_split.std():.2f} GeV")

    print("\n=== Summary ===")
    for name, (p_min, p_max) in splits.items():
        f = output_path / f"training_{name}.npz"
        if f.exists():
            d = np.load(f)
            print(f"{name}: {len(d['P']):,} samples, p ∈ [{p_min}, {p_max}) GeV")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create momentum-split training data")
    parser.add_argument("--input", default="data/training_50M.npz",
                        help="Input training file")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for split files")
    parser.add_argument("--max-samples", type=int, default=10_000_000,
                        help="Maximum samples per split")
    args = parser.parse_args()
    
    create_momentum_splits(args.input, args.output_dir, args.max_samples)
