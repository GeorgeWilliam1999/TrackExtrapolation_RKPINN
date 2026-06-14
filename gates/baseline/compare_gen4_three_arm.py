#!/usr/bin/env python3
"""Gen-4 three-arm UT->T evaluation: NN vs production extrapUTT vs straight-line.

Common ground: the fixed UT->T plane (z 2665 -> 7826) with v8r1 physical-kappa RK
truth -- the same set on which the production polynomial scores 15 um median
(P0.1 / P0.0b, 2026-06-11).

INPUT DATA
  paper_p0/plane_ref_v8r1.npz        X_plane[N,5]=(x,y,tx,ty,qop), Y_true[N,5]
                                     (physical PV-pointing tracks, v8r1, kappa=1e-3;
                                      built by make_plane_ref_v8r1.py)
  paper_p0/plane_poly_v8r1_polm1.csv extrapUTT predictions on the same X (m_polarity=-1)
MODELS (lab trained_models/, trained on train_10M_gen4.npz)
  pinn_v2_g4_lam0_2M_cpu  pinn_v2_g4_lam0p1_2M_cpu
  pinn_v2_g4_lam0_10M     pinn_v2_g4_lam0p1_10M

OUTPUT
  paper_p0/gen4_three_arm.json   per-arm metrics (median/p68/p95/p99 |dx| um,
                                 |dtx| urad, median |dx| by |q/p| quartile)
  paper_p0/gen4_three_arm_arrays.npz   per-arm |dx| arrays + qop (for the notebook)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent
sys.path.insert(0, str(GEN3 / "models"))
from architectures import create_model  # noqa: E402

ZINI, ZFIN = 2665.0, 7826.0
DZ = ZFIN - ZINI
TM = GEN3 / "trained_models"
MODELS = [
    "pinn_v2_g4_lam0_2M_cpu", "pinn_v2_g4_lam0p1_2M_cpu",
    "pinn_v2_g4_lam0_10M", "pinn_v2_g4_lam0p1_10M",
]


def load_model(name: str):
    d = TM / name
    ckpt = torch.load(d / "best_model.pt", weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    model = create_model("pinn_v2", hidden_dims=cfg["hidden_dims"],
                         activation=cfg["activation"], dropout=cfg.get("dropout", 0.0),
                         lambda_pde=cfg.get("lambda_pde", 0.0),
                         lambda_ic=cfg.get("lambda_ic", 0.0),
                         n_collocation=cfg.get("n_collocation", 2),
                         kick_scaled_head=cfg.get("kick_scaled_head", False),
                         pde_scale_mode=cfg.get("pde_scale_mode", "legacy"),
                         pde_ref_length=cfg.get("pde_ref_length", 5213.0))
    if (d / "normalization.json").exists():
        model.load_normalization(str(d / "normalization.json"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def metrics(pred_xy_tx, Y, qop):
    dx = np.abs(pred_xy_tx[:, 0] - Y[:, 0]) * 1e3       # um
    dtx = np.abs(pred_xy_tx[:, 2] - Y[:, 2]) * 1e6      # urad
    q = np.abs(qop)
    edges = np.quantile(q, [0.25, 0.5, 0.75])
    b = np.digitize(q, edges)                            # 0 (high p) .. 3 (low p)
    byq = [float(np.median(dx[b == k])) for k in range(4)]
    return dict(median_dx_um=float(np.median(dx)),
                p68_dx_um=float(np.quantile(dx, 0.68)),
                p95_dx_um=float(np.quantile(dx, 0.95)),
                p99_dx_um=float(np.quantile(dx, 0.99)),
                median_dtx_urad=float(np.median(dtx)),
                median_dx_um_by_qop_quartile_hi2lo_p=byq), dx


def main() -> None:
    d = np.load(HERE / "plane_ref_v8r1.npz")
    X, Y = d["X_plane"], d["Y_true"]
    qop = X[:, 4]
    poly = np.loadtxt(HERE / "plane_poly_v8r1_polm1.csv", delimiter=",", skiprows=1)
    assert poly.shape[0] == X.shape[0], "poly/truth row mismatch"

    results, arrays = {}, {"qop": qop, "p_GeV": 0.299792458 / np.abs(qop)}

    # arm 1: incumbent polynomial
    results["extrapUTT (incumbent)"], arrays["extrapUTT"] = metrics(poly, Y, qop)
    # arm 2: straight line control
    sl = np.stack([X[:, 0] + X[:, 2] * DZ, X[:, 1] + X[:, 3] * DZ, X[:, 2], X[:, 3]], axis=1)
    results["straight_line"], arrays["straight_line"] = metrics(sl, Y, qop)
    # arm 3: the four gen-4 NNs
    Xin = np.concatenate([X[:, :5], np.full((X.shape[0], 1), ZINI),
                          np.full((X.shape[0], 1), DZ)], axis=1).astype(np.float32)
    with torch.no_grad():
        for name in MODELS:
            if not (TM / name / "best_model.pt").exists():
                print(f"SKIP {name}"); continue
            pred = load_model(name)(torch.from_numpy(Xin)).numpy()
            results[name], arrays[name] = metrics(pred, Y, qop)

    # report
    print(f"\nreference: {X.shape[0]} tracks, p in "
          f"[{arrays['p_GeV'].min():.1f}, {arrays['p_GeV'].max():.1f}] GeV   "
          f"(plane z {ZINI:.0f}->{ZFIN:.0f})\n")
    h = f"{'arm':<26}{'med um':>9}{'p68':>9}{'p95':>9}{'p99':>9}{'dtx urad':>10}   byQ hi->lo p"
    print(h); print("-" * len(h))
    for k, m in results.items():
        bq = "/".join(f"{v:.0f}" for v in m["median_dx_um_by_qop_quartile_hi2lo_p"])
        print(f"{k:<26}{m['median_dx_um']:>9.1f}{m['p68_dx_um']:>9.1f}"
              f"{m['p95_dx_um']:>9.1f}{m['p99_dx_um']:>9.1f}{m['median_dtx_urad']:>10.1f}   [{bq}]")

    (HERE / "gen4_three_arm.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(HERE / "gen4_three_arm_arrays.npz", **arrays)
    print(f"\nwrote gen4_three_arm.json + gen4_three_arm_arrays.npz")


if __name__ == "__main__":
    main()
