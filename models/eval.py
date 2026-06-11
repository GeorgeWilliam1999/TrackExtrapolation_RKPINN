#!/usr/bin/env python3
"""eval.py — R1 standalone evaluation script.

Re-evaluates any saved gen-3 checkpoint using the R1 robust-metric suite
(median, p68, p95, p99 of |Δx|, |Δy|, |Δslope|) and a log-cosh loss.
Results are written to <exp_dir>/eval_R1_<date>.json and printed to stdout.

Usage
-----
# Evaluate a single checkpoint against its own saved test-split indices:
    python models/eval.py --checkpoint trained_models/pinn_v2_small_v1

# Evaluate all checkpoints in trained_models/ at once:
    python models/eval.py --all

# Override the data path (if the default 10M file moved):
    python models/eval.py --checkpoint ... --data-path /path/to/train_10M_gen3.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
sys.path.insert(0, str(_REPO_ROOT / "core"))

# Big data / checkpoints live in the lab, not in this repo.
_LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))

from architectures import create_model  # noqa: E402
from train import validate, log_cosh_loss  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(exp_dir: Path, device: torch.device):
    """Load model + config from a gen-3 experiment directory."""
    ckpt_path = exp_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {exp_dir}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt["config"]

    # Reconstruct the architecture
    mt = config["model_type"]
    if mt == "mlp":
        model = create_model(
            "mlp",
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            dropout=config.get("dropout", 0.0),
        )
    elif mt == "pinn_v2":
        model = create_model(
            "pinn_v2",
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            dropout=config.get("dropout", 0.0),
            lambda_pde=config.get("lambda_pde", 0.1),
            lambda_ic=config.get("lambda_ic", 0.1),
            n_collocation=config.get("n_collocation", 2),
        )
    elif mt == "neural_rk4":
        model = create_model(
            "neural_rk4",
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            n_rk_steps=config.get("n_rk_steps", 1),
            correction_scale_init=config.get("correction_scale_init", 1e-3),
        )
    else:
        raise ValueError(f"Unknown model_type {mt!r}")

    norm_path = exp_dir / "normalization.json"
    if norm_path.exists():
        model.load_normalization(str(norm_path))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


def _load_test_data(exp_dir: Path, config: dict, data_path_override: str | None):
    """Load the frozen test-split for this experiment."""
    data_path = Path(data_path_override or config.get("data_path",
        str(_LAB / "data" / "train_10M_gen3.npz")))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"  Loading {data_path} ...")
    with np.load(data_path, allow_pickle=False) as d:
        X = d["X"].astype(np.float32)
        Y = d["Y"].astype(np.float32)

    idx_path = exp_dir / "test_indices.npy"
    if idx_path.exists():
        idx = np.load(idx_path)
        print(f"  Using saved test indices: {len(idx):,} tracks")
        X = X[idx]
        Y = Y[idx]
    else:
        # Fall back: use last 10% of data (deterministic but not the exact split)
        n = X.shape[0]
        n_test = max(10_000, int(n * 0.10))
        X = X[-n_test:]
        Y = Y[-n_test:]
        print(f"  WARNING: no test_indices.npy — using last {n_test:,} tracks as proxy")

    return X, Y


def _make_loader(X, Y, batch_size=4096):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    exp_dir: Path,
    device: torch.device,
    data_path_override: str | None = None,
) -> dict:
    """Return the full R1 metric dict for one checkpoint directory."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_dir.name}")

    model, config = _load_checkpoint(exp_dir, device)
    n_params = model.count_parameters()
    print(f"  model_type={config['model_type']}  params={n_params:,}  hidden_dims={config['hidden_dims']}")

    X, Y = _load_test_data(exp_dir, config, data_path_override)
    loader = _make_loader(X, Y)
    metrics = validate(model, loader, device)

    # Stamp result
    result = {
        "experiment": exp_dir.name,
        "model_type": config["model_type"],
        "hidden_dims": config["hidden_dims"],
        "n_params": n_params,
        "n_test_tracks": len(X),
        "eval_date": datetime.now().strftime("%Y-%m-%d"),
        "metrics": metrics,
    }

    # Print summary
    print(f"  median |Δx|      = {metrics['median_dx_mm']*1e3:7.2f} µm")
    print(f"  p68    |Δx|      = {metrics['p68_dx_mm']*1e3:7.2f} µm")
    print(f"  p95    |Δx|      = {metrics['p95_dx_mm']*1e3:7.2f} µm")
    print(f"  p99    |Δx|      = {metrics['p99_dx_mm']*1e3:7.2f} µm")
    print(f"  median |Δpos|    = {metrics['median_pos_mm']*1e3:7.2f} µm")
    print(f"  p95    |Δpos|    = {metrics['p95_pos_mm']*1e3:7.2f} µm")
    print(f"  median |Δtx|     = {metrics['median_dtx_mrad']:7.3f} mrad")
    print(f"  pos rms          = {metrics['pos_rms_mm']*1e3:7.2f} µm  (legacy, not used for selection)")
    print(f"  log-cosh loss    = {metrics['loss']:.4e}")
    print(f"  mse loss         = {metrics['mse_loss']:.4e}  (ratio log-cosh/mse = {metrics['loss']/max(metrics['mse_loss'],1e-30):.1f}x)")

    # Save
    out_path = exp_dir / f"eval_R1_{result['eval_date']}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("R1 EVALUATION SUMMARY")
    print("=" * 100)
    header = f"{'Experiment':<35} {'Type':<12} {'Params':>8}  {'med|Δx|(µm)':>12}  {'p95|Δx|(µm)':>12}  {'med|Δpos|(µm)':>14}  {'med|Δtx|(mrad)':>15}"
    print(header)
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["metrics"]["median_dx_mm"]):
        m = r["metrics"]
        print(
            f"{r['experiment']:<35} {r['model_type']:<12} {r['n_params']:>8,}  "
            f"{m['median_dx_mm']*1e3:>12.2f}  {m['p95_dx_mm']*1e3:>12.1f}  "
            f"{m['median_pos_mm']*1e3:>14.2f}  {m['median_dtx_mrad']:>15.3f}"
        )
    print("=" * 100)

    # Warn if any model has mse_loss / log_cosh_loss ratio > 10x
    for r in results:
        m = r["metrics"]
        ratio = m["mse_loss"] / max(m["loss"], 1e-30)
        if ratio > 10:
            print(
                f"  WARNING [{r['experiment']}]: mse_loss/log_cosh = {ratio:.0f}x — "
                f"heavy-tail dominance confirmed (gen-3 §F2). median metric is the reliable number."
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="R1 evaluation of gen-3 checkpoints")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a single experiment directory (e.g. trained_models/pinn_v2_small_v1)")
    p.add_argument("--all", action="store_true",
                   help="Evaluate all subdirectories of trained_models/ (skips _archive_* and smoke_*)")
    p.add_argument("--data-path", type=str, default=None,
                   help="Override data file path")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (default: cuda if available)")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if args.all:
        base = _LAB / "trained_models"
        dirs = sorted(
            d for d in base.iterdir()
            if d.is_dir()
            and not d.name.startswith("_archive")
            and not d.name.startswith("smoke")
            and not d.name.startswith("timing")
            and (d / "best_model.pt").exists()
        )
        if not dirs:
            print(f"No valid checkpoints found under {base}")
            sys.exit(1)
        results = [evaluate_checkpoint(d, device, args.data_path) for d in dirs]
        _print_summary(results)
    elif args.checkpoint:
        exp_dir = Path(args.checkpoint)
        if not exp_dir.is_absolute():
            exp_dir = _LAB / exp_dir
        evaluate_checkpoint(exp_dir, device, args.data_path)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
