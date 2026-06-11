#!/usr/bin/env python3
"""R2 — A4 Jacobian gate measurement on replacement candidates.

Runs evaluate_a4() on:
  1. pinn_v2_small_v1   (current best true replacement)
  2. mlp_medium_v1_broken  (collapsed MLP, used as sanity lower bound)

Uses the existing RK4 reference Jacobians from:
    For_Allen/artifacts/phase1a/J_rk4_reference.npy
    For_Allen/artifacts/phase1a/X_a4.npy

Results written to:
    experiments/gen_3/results/R2_jacobian_2026-MM-DD.json
"""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FOR_ALLEN = REPO / "For_Allen"
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))

sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))
sys.path.insert(0, str(FOR_ALLEN / "src"))

from architectures import create_model  # noqa: E402
from for_allen.eval.jacobian import evaluate_a4, load_reference_jacobians  # noqa: E402


def _load_checkpoint(exp_dir: Path, device: torch.device):
    ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location=device)
    config = ckpt["config"]
    mt = config["model_type"]
    if mt == "mlp":
        model = create_model("mlp", hidden_dims=config["hidden_dims"],
                             activation=config["activation"],
                             dropout=config.get("dropout", 0.0))
    elif mt == "pinn_v2":
        model = create_model("pinn_v2", hidden_dims=config["hidden_dims"],
                             activation=config["activation"],
                             dropout=config.get("dropout", 0.0),
                             lambda_pde=config.get("lambda_pde", 0.1),
                             lambda_ic=config.get("lambda_ic", 0.1),
                             n_collocation=config.get("n_collocation", 2),
                             kick_scaled_head=config.get("kick_scaled_head", False),
                             pde_scale_mode=config.get("pde_scale_mode", "legacy"),
                             pde_ref_length=config.get("pde_ref_length", 5213.0))
    else:
        raise ValueError(f"Unknown model_type {mt!r}")
    norm_path = exp_dir / "normalization.json"
    if norm_path.exists():
        model.load_normalization(str(norm_path))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def main() -> None:
    artifacts = FOR_ALLEN / "artifacts" / "phase1a"
    j_ref_path = artifacts / "J_rk4_reference.npy"
    x_a4_path  = artifacts / "X_a4.npy"

    if not j_ref_path.exists() or not x_a4_path.exists():
        print(f"ERROR: reference artifacts not found under {artifacts}")
        print("  Expected: J_rk4_reference.npy  X_a4.npy")
        sys.exit(1)

    J_ref = load_reference_jacobians(j_ref_path)
    X_a4  = np.load(x_a4_path).astype(np.float64)
    n = min(J_ref.shape[0], X_a4.shape[0])
    J_ref = J_ref[:n]
    X_a4  = X_a4[:n]
    print(f"Reference Jacobians: {n} tracks  (from {j_ref_path})")

    candidates = [
        ("pinn_v2_small_v1",        LAB / "trained_models" / "pinn_v2_small_v1"),
        ("pinn_v2_kick_2M_cpu",     LAB / "trained_models" / "pinn_v2_kick_2M_cpu"),
        ("pinn_v2_kick_only_2M_cpu",LAB / "trained_models" / "pinn_v2_kick_only_2M_cpu"),
        ("pinn_v2_kick_10M",        LAB / "trained_models" / "pinn_v2_kick_10M"),
        ("pinn_v2_kick_only_10M",   LAB / "trained_models" / "pinn_v2_kick_only_10M"),
        ("pinn_v2_lam0p1_2M_cpu",   LAB / "trained_models" / "pinn_v2_lam0p1_2M_cpu"),
        ("pinn_v2_lam0_2M_cpu",     LAB / "trained_models" / "pinn_v2_lam0_2M_cpu"),
        ("pinn_v2_lam0p1_10M",      LAB / "trained_models" / "pinn_v2_lam0p1_10M"),
        ("pinn_v2_lam0_10M",        LAB / "trained_models" / "pinn_v2_lam0_10M"),
    ]

    results = {}
    for name, exp_dir in candidates:
        if not (exp_dir / "best_model.pt").exists():
            print(f"  SKIP {name} — no best_model.pt")
            continue
        print(f"\n{'='*60}\nEvaluating A4: {name}")
        model, _ = _load_checkpoint(exp_dir, torch.device("cpu"))
        report = evaluate_a4(model, X_a4, J_ref, model_name=name, verbose=True)
        results[name] = report.to_dict()

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("R2 A4 SUMMARY")
    print("=" * 80)
    hdr = f"{'Model':<30} {'frob_mean':>10} {'frob_p95':>10} {'off_frob_p95':>14} {'Verdict'}"
    print(hdr)
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<30} {r['frob_rel_mean']:>10.4f} {r['frob_rel_p95']:>10.4f} "
              f"{r['off_max_frob_p95']:>14.6f}  {r['verdict']}")
    print("=" * 80)

    # --- Decision text ---
    pinn_r = results.get("pinn_v2_small_v1", {})
    if pinn_r.get("passes"):
        print("\nDECISION: pinn_v2_small_v1 PASSES A4 → proceed to R3/R4 unchanged.")
    else:
        frob = pinn_r.get("frob_rel_mean", float("nan"))
        off  = pinn_r.get("off_max_frob_p95", float("nan"))
        if frob < 2 * 0.05:
            print(f"\nDECISION: pinn_v2_small_v1 FAILS A4 narrowly (frob={frob:.4f} off_frob={off:.4f}, < 2× gate).")
            print("  → Proceed but warm up R-X.1 (Jacobian co-supervision) as parallel track.")
        else:
            print(f"\nDECISION: pinn_v2_small_v1 FAILS A4 by >2× (frob={frob:.4f} off_frob={off:.4f}).")
            print("  → Activate R-X.2 (straight-line residual head) before further scaling.")

    # --- Persist ---
    out_path = REPO / "results" / f"R2_jacobian_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
