#!/usr/bin/env python3
"""A4 sanity check against the rebuilt PHYSICAL-kappa reference.

Drives the *existing* A4 gate (for_allen.eval.jacobian.evaluate_a4, the same
function run_r2_jacobian.py uses) and the same checkpoint loader, against the
new fp64 / kappa=1e-3 reference, on:

  * pinn_v2_small_v1          — known-good gen-3-era replacement candidate
  * pinn_v2_g4_lam0_2M_cpu    } the four wave-2 gen-4 models
  * pinn_v2_g4_lam0p1_2M_cpu  }  (undertrained — point is the gate now RUNS
  * pinn_v2_g4_lam0_10M       }   meaningfully at physical kappa, not that
  * pinn_v2_g4_lam0p1_10M     }   they pass)

Reports Frobenius rel-err (mean/median/p95) + off-diag metric + verdict, and
writes results/A4_sanity_<date>.json.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# run_r2_jacobian is the canonical A4 gate driver; importing it also wires up
# sys.path (models, core/utils, For_Allen/src) and defines the tree's paths.
import run_r2_jacobian as r2  # noqa: E402
from run_r2_jacobian import _load_checkpoint  # noqa: E402
from for_allen.eval.jacobian import evaluate_a4, load_reference_jacobians  # noqa: E402

FOR_ALLEN = r2.FOR_ALLEN
# checkpoints live at GEN3_ROOT/trained_models (lab) or LAB/trained_models (repo).
CKPT_ROOT = getattr(r2, "LAB", getattr(r2, "GEN3_ROOT", HERE)) / "trained_models"


def main() -> None:
    artifacts = FOR_ALLEN / "artifacts" / "phase1a"
    J_ref = load_reference_jacobians(artifacts / "J_rk4_reference.npy")
    X_a4 = np.load(artifacts / "X_a4.npy").astype(np.float64)
    n = min(J_ref.shape[0], X_a4.shape[0])
    J_ref, X_a4 = J_ref[:n], X_a4[:n]
    print(f"Reference: {n} states  (rebuilt physical-kappa fp64)  from {artifacts}")

    models = [
        ("pinn_v2_small_v1",        "gen-3-era"),
        ("pinn_v2_g4_lam0_2M_cpu",  "gen-4"),
        ("pinn_v2_g4_lam0p1_2M_cpu","gen-4"),
        ("pinn_v2_g4_lam0_10M",     "gen-4"),
        ("pinn_v2_g4_lam0p1_10M",   "gen-4"),
    ]

    results = {}
    for name, era in models:
        exp_dir = CKPT_ROOT / name
        if not (exp_dir / "best_model.pt").exists():
            print(f"  SKIP {name} — no best_model.pt")
            continue
        print(f"\n{'='*62}\n[{era}] {name}")
        model, _ = _load_checkpoint(exp_dir, torch.device("cpu"))
        report = evaluate_a4(model, X_a4, J_ref, model_name=name, verbose=False)
        d = report.to_dict()
        d["era"] = era
        results[name] = d
        print(report.summary())

    print("\n" + "=" * 86)
    print("A4 SANITY SUMMARY (vs rebuilt physical-kappa reference)")
    print("=" * 86)
    hdr = f"{'Model':<28}{'era':<11}{'frob_mean':>10}{'frob_med':>10}{'frob_p95':>10}{'off_p95':>10}  verdict"
    print(hdr); print("-" * 86)
    for name, r in results.items():
        print(f"{name:<28}{r['era']:<11}{r['frob_rel_mean']:>10.4f}{r['frob_rel_median']:>10.4f}"
              f"{r['frob_rel_p95']:>10.4f}{r['off_max_frob_p95']:>10.4f}  {r['verdict']}")
    print("=" * 86)

    out = HERE / "results" / f"A4_sanity_{date.today()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
