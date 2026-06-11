#!/usr/bin/env python3
"""P0.1 — verdict table: production polynomial vs NN candidates vs RK truth
on the fixed UT->T plane set (z 2665 -> 7826).

Inputs:
  utt_plane_ref.npz          (X_plane, Y_true)
  poly_pred_pol{m1,p1}.csv   (from extraputt_baseline, polarity -1/+1)

NN candidates evaluated in-process on the same plane states.
Momentum labels: p[GeV] = 0.299792458 / qop_corpus  (corpus qop = 299.792458 * q/p[1/MeV]).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))
from architectures import create_model  # noqa: E402

ZINI, ZFIN = 2665.0, 7826.0
DZ = ZFIN - ZINI


def load_nn(exp_dir: Path):
    ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    model = create_model(
        "pinn_v2", hidden_dims=cfg["hidden_dims"], activation=cfg["activation"],
        dropout=cfg.get("dropout", 0.0), lambda_pde=cfg.get("lambda_pde", 0.1),
        lambda_ic=cfg.get("lambda_ic", 0.1), n_collocation=cfg.get("n_collocation", 2),
        kick_scaled_head=cfg.get("kick_scaled_head", False),
        pde_scale_mode=cfg.get("pde_scale_mode", "legacy"),
        pde_ref_length=cfg.get("pde_ref_length", 5213.0))
    model.load_normalization(str(exp_dir / "normalization.json"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def metrics(pred_xy_txty: np.ndarray, truth: np.ndarray, qop: np.ndarray) -> dict:
    dx = np.abs(pred_xy_txty[:, 0] - truth[:, 0]) * 1e3  # um
    dy = np.abs(pred_xy_txty[:, 1] - truth[:, 1]) * 1e3
    dtx = np.abs(pred_xy_txty[:, 2] - truth[:, 2]) * 1e6  # urad
    q = np.abs(qop)
    edges = np.quantile(q, [0.25, 0.5, 0.75])
    bins = np.digitize(q, edges)
    by_q = [float(np.median(dx[bins == k])) for k in range(4)]
    p_edges_gev = [float(0.299792458 / e) for e in edges]  # high->low momentum
    return {
        "median_dx_um": float(np.median(dx)),
        "p68_dx_um": float(np.quantile(dx, 0.68)),
        "p95_dx_um": float(np.quantile(dx, 0.95)),
        "p99_dx_um": float(np.quantile(dx, 0.99)),
        "median_dy_um": float(np.median(dy)),
        "median_dtx_urad": float(np.median(dtx)),
        "median_dx_by_qop_quartile_um": by_q,
        "p_GeV_quartile_edges_high_to_low": p_edges_gev,
    }


def main() -> None:
    d = np.load(HERE / "utt_plane_ref.npz")
    X, Y = d["X_plane"], d["Y_true"]
    qop = X[:, 4]
    print(f"reference set: {X.shape[0]} tracks; p in "
          f"[{0.299792458/np.abs(qop).max():.2f}, {0.299792458/np.abs(qop).min():.1f}] GeV")

    results = {}

    # --- production polynomial (both polarity hypotheses) ---
    for tag in ["m1", "p1"]:
        f = HERE / f"poly_pred_pol{tag}.csv"
        if f.exists():
            P = np.loadtxt(f, delimiter=",", skiprows=1)
            results[f"extrapUTT_pol{tag}"] = metrics(P, Y, qop)

    # --- NN candidates on identical states ---
    cands = {
        "pinn_v2_small_v1 (deployed)": LAB / "trained_models" / "pinn_v2_small_v1",
        "pinn_v2_kick_10M": LAB / "trained_models" / "pinn_v2_kick_10M",
        "pinn_v2_lam0_2M_cpu": LAB / "trained_models" / "pinn_v2_lam0_2M_cpu",
        "pinn_v2_lam0p1_2M_cpu": LAB / "trained_models" / "pinn_v2_lam0p1_2M_cpu",
    }
    Xin = np.concatenate([X[:, :5],
                          np.full((X.shape[0], 1), ZINI),
                          np.full((X.shape[0], 1), DZ)], axis=1).astype(np.float32)
    for name, p in cands.items():
        if not (p / "best_model.pt").exists():
            print(f"  (skip {name}: no checkpoint)")
            continue
        m = load_nn(p)
        with torch.no_grad():
            pred = m(torch.from_numpy(Xin)).numpy()
        results[name] = metrics(pred, Y, qop)

    # --- straight line control (no field) ---
    sl = np.stack([X[:, 0] + X[:, 2] * DZ, X[:, 1] + X[:, 3] * DZ, X[:, 2], X[:, 3]], axis=1)
    results["straight_line"] = metrics(sl, Y, qop)

    hdr = f"{'Model':<28} {'med dx um':>10} {'p68':>9} {'p95':>10} {'p99':>10}  {'med dx by |q/p| quartile (hi->lo p)'}"
    print("\n" + hdr); print("-" * len(hdr))
    for k, v in results.items():
        bq = "/".join(f"{x:.0f}" for x in v["median_dx_by_qop_quartile_um"])
        print(f"{k:<28} {v['median_dx_um']:>10.1f} {v['p68_dx_um']:>9.1f} "
              f"{v['p95_dx_um']:>10.1f} {v['p99_dx_um']:>10.1f}  [{bq}]")

    out = HERE / "P0p1_baseline_verdict.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
