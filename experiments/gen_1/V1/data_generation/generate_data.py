#!/usr/bin/env python3
"""
Generate Training Data for Track Extrapolation (V1 — Variable dz)

Uses pure Python RK4 integrator with the real LHCb field map (twodip.rtf)
to generate ground truth extrapolations.

KEY IMPROVEMENT over original V1: Variable dz (step size) per track.
The old V1 used fixed dz=8000mm, which caused normalization to fail
at any other step size (std(dz) ≈ 0 → division by zero).

Data format:
  X [N, 6]: Input  — [x, y, tx, ty, qop, dz]
  Y [N, 4]: Output — [x_out, y_out, tx_out, ty_out]
  P [N]:    Momentum in GeV
  Z [N, 2]: z_start, z_end for each track

Usage:
    python generate_data.py --n-tracks 100000 --output-dir datasets/
    python generate_data.py --n-tracks 50000000 --workers 16 --name train_50M

Author: G. Scriven
Date: March 2026
"""

import numpy as np
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Optional
import sys
import time
import os

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from rk4_propagator import RK4Integrator
from magnetic_field import get_field_numpy


# ============================================================================
# Global worker state (initialized once per worker process)
# ============================================================================

_WORKER_INTEGRATOR = None


def _init_worker(polarity: int, step_size: float):
    """Initialize field and integrator once per worker process."""
    global _WORKER_INTEGRATOR
    # Reseed numpy with OS entropy so forked workers don't share random state
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))
    field = get_field_numpy(use_interpolated=True, polarity=polarity)
    _WORKER_INTEGRATOR = RK4Integrator(field=field, step_size=step_size)


# ============================================================================
# Track generation
# ============================================================================

def generate_single_track(args: Tuple) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Generate a single track propagation with variable z_start/z_end.

    Args:
        args: (track_id, z_start, z_end, p_range, x_range, y_range, tx_range, ty_range)

    Returns:
        input_state:  [x, y, tx, ty, qop, dz]
        output_state: [x_out, y_out, tx_out, ty_out]
        momentum:     GeV
        z_info:       [z_start, z_end]
    """
    global _WORKER_INTEGRATOR
    integrator = _WORKER_INTEGRATOR

    (track_id, z_start, z_end, p_range, x_range, y_range, tx_range, ty_range) = args

    # Log-uniform momentum
    log_p = np.random.uniform(np.log(p_range[0]), np.log(p_range[1]))
    momentum = np.exp(log_p)
    charge = np.random.choice([-1, +1])
    qop = charge / (momentum * 1000.0)  # 1/MeV

    # Random position and slopes
    x  = np.random.uniform(*x_range)
    y  = np.random.uniform(*y_range)
    tx = np.random.uniform(*tx_range)
    ty = np.random.uniform(*ty_range)

    state_in = np.array([x, y, tx, ty, qop])
    dz = z_end - z_start

    # Propagate
    state_out = integrator.propagate(state_in, z_start, z_end)

    # Validate
    if not np.all(np.isfinite(state_out)):
        return np.zeros(6), np.zeros(4), 0.0, np.zeros(2)

    input_state = np.array([x, y, tx, ty, qop, dz])
    output_state = np.array([state_out[0], state_out[1], state_out[2], state_out[3]])
    z_info = np.array([z_start, z_end])

    return input_state, output_state, momentum, z_info


def _sample_z_pairs(n: int, z_min: float, z_max: float,
                    dz_min: float, dz_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (z_start, z_end) pairs with variable dz.

    Strategy: sample z_start uniformly, then sample dz uniformly,
    clamp z_end to stay within grid bounds.
    """
    z_starts = np.random.uniform(z_min, z_max - dz_min, size=n)
    dzs = np.random.uniform(dz_min, dz_max, size=n)
    z_ends = z_starts + dzs

    # Clamp z_end to grid boundary
    z_ends = np.clip(z_ends, z_min + dz_min, z_max)
    # Recompute dz after clamping
    dzs = z_ends - z_starts

    # Re-filter: ensure dz >= dz_min
    valid = dzs >= dz_min
    return z_starts[valid], z_ends[valid]


def generate_dataset(n_tracks: int,
                     z_min: float = 0.0,
                     z_max: float = 14000.0,
                     dz_min: float = 100.0,
                     dz_max: float = 10000.0,
                     p_range: Tuple[float, float] = (1.0, 100.0),
                     x_range: Tuple[float, float] = (-300.0, 300.0),
                     y_range: Tuple[float, float] = (-250.0, 250.0),
                     tx_range: Tuple[float, float] = (-0.3, 0.3),
                     ty_range: Tuple[float, float] = (-0.25, 0.25),
                     step_size: float = 5.0,
                     polarity: int = -1,
                     n_workers: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dataset of track propagations in parallel with VARIABLE dz.

    Returns:
        X: Input states  [N, 6] = [x, y, tx, ty, qop, dz]
        Y: Output states [N, 4] = [x_out, y_out, tx_out, ty_out]
        P: Momenta [N] in GeV
        Z: z positions [N, 2] = [z_start, z_end]
    """
    print(f"\n{'='*70}")
    print(f"Track Extrapolation Data Generation (Variable dz)")
    print(f"{'='*70}")
    print(f"  Tracks requested: {n_tracks:,}")
    print(f"  z range: [{z_min}, {z_max}] mm")
    print(f"  dz range: [{dz_min}, {dz_max}] mm")
    print(f"  Momentum: [{p_range[0]}, {p_range[1]}] GeV (log-uniform)")
    print(f"  x range: {x_range} mm, y range: {y_range} mm")
    print(f"  tx range: {tx_range}, ty range: {ty_range}")
    print(f"  RK4 step: {step_size} mm")
    print(f"  Polarity: {'MagDown' if polarity < 0 else 'MagUp'}")
    print(f"  Workers: {n_workers}")
    print(f"  Field: twodip.rtf (real, interpolated)")

    # Sample z pairs
    # Over-sample to account for filtering
    n_oversample = int(n_tracks * 1.05)
    z_starts, z_ends = _sample_z_pairs(n_oversample, z_min, z_max, dz_min, dz_max)
    n_available = len(z_starts)
    if n_available < n_tracks:
        # If too many got filtered, resample
        while n_available < n_tracks:
            extra_starts, extra_ends = _sample_z_pairs(
                n_tracks - n_available, z_min, z_max, dz_min, dz_max)
            z_starts = np.concatenate([z_starts, extra_starts])
            z_ends = np.concatenate([z_ends, extra_ends])
            n_available = len(z_starts)
    z_starts = z_starts[:n_tracks]
    z_ends = z_ends[:n_tracks]

    print(f"\n  dz distribution: mean={np.mean(z_ends - z_starts):.0f}mm, "
          f"std={np.std(z_ends - z_starts):.0f}mm, "
          f"range=[{np.min(z_ends - z_starts):.0f}, {np.max(z_ends - z_starts):.0f}]mm")

    # Build argument list
    args_list = [
        (i, z_starts[i], z_ends[i], p_range, x_range, y_range, tx_range, ty_range)
        for i in range(n_tracks)
    ]

    # Generate in parallel
    start_time = time.time()
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(polarity, step_size)) as pool:
        chunk = max(100, n_tracks // (n_workers * 20))
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_single_track, args_list, chunksize=chunk)):
            results.append(result)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_tracks - i - 1) / rate
                print(f"  Progress: {i+1:,}/{n_tracks:,} "
                      f"({rate:.0f} tracks/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time

    # Unpack
    X = np.array([r[0] for r in results], dtype=np.float32)
    Y = np.array([r[1] for r in results], dtype=np.float32)
    P = np.array([r[2] for r in results], dtype=np.float32)
    Z = np.array([r[3] for r in results], dtype=np.float32)

    # Filter out failed tracks (P == 0)
    valid = P > 0
    n_failed = np.sum(~valid)
    if n_failed > 0:
        print(f"\n  WARNING: {n_failed:,}/{n_tracks:,} tracks failed ({100*n_failed/n_tracks:.2f}%)")
        X, Y, P, Z = X[valid], Y[valid], P[valid], Z[valid]

    print(f"\n  Generated {len(P):,} valid tracks in {elapsed:.1f}s "
          f"({n_tracks/elapsed:.0f} tracks/s)")
    print(f"\n  Dataset shapes:")
    print(f"    X (input):  {X.shape}")
    print(f"    Y (output): {Y.shape}")
    print(f"    P (mom.):   {P.shape}")
    print(f"    Z (z pos.): {Z.shape}")
    print(f"  Momentum: [{P.min():.2f}, {P.max():.2f}] GeV (mean={P.mean():.2f})")
    print(f"  dz: [{X[:,5].min():.0f}, {X[:,5].max():.0f}] mm "
          f"(mean={X[:,5].mean():.0f}, std={X[:,5].std():.0f})")

    return X, Y, P, Z


def save_dataset(X, Y, P, Z, output_dir: Path, name: str):
    """Save dataset as compressed NPZ archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not name.endswith('.npz'):
        name += '.npz'

    output_path = output_dir / name
    np.savez_compressed(output_path, X=X, Y=Y, P=P, Z=Z)

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"\n  Saved: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate track extrapolation training data (variable dz)")

    # Dataset
    parser.add_argument('--n-tracks', type=int, default=10000,
                        help='Number of tracks (default: 10000)')
    parser.add_argument('--name', type=str, default='train',
                        help='Dataset name (default: train)')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='Output directory (default: datasets/)')

    # Physics: z range
    parser.add_argument('--z-min', type=float, default=0.0,
                        help='Minimum z for sampling (default: 0 mm)')
    parser.add_argument('--z-max', type=float, default=14000.0,
                        help='Maximum z for sampling (default: 14000 mm)')
    parser.add_argument('--dz-min', type=float, default=100.0,
                        help='Minimum step size (default: 100 mm)')
    parser.add_argument('--dz-max', type=float, default=10000.0,
                        help='Maximum step size (default: 10000 mm)')

    # Physics: track parameters
    parser.add_argument('--p-min', type=float, default=1.0,
                        help='Min momentum in GeV (default: 1.0)')
    parser.add_argument('--p-max', type=float, default=100.0,
                        help='Max momentum in GeV (default: 100)')
    parser.add_argument('--step-size', type=float, default=5.0,
                        help='RK4 integration step in mm (default: 5.0)')
    parser.add_argument('--polarity', type=int, choices=[-1, 1], default=-1,
                        help='Magnet polarity (default: -1 = MagDown)')

    # Computation
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    np.random.seed(args.seed)

    X, Y, P, Z = generate_dataset(
        n_tracks=args.n_tracks,
        z_min=args.z_min,
        z_max=args.z_max,
        dz_min=args.dz_min,
        dz_max=args.dz_max,
        p_range=(args.p_min, args.p_max),
        step_size=args.step_size,
        polarity=args.polarity,
        n_workers=args.workers,
    )

    save_dataset(X, Y, P, Z, Path(args.output_dir), args.name)

    print(f"\n{'='*70}")
    print("DATA GENERATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
