#!/usr/bin/env python3
"""R7 — UT->T propagation accuracy of candidate NNs.

Restricts each model's held-out test split to tracks whose start point sits
in the UT region (z_start in [2300, 3000] mm) and whose endpoint sits in the
SciFi T-stations (z_end in [7600, 9500] mm). This is the regime that the
Allen HLT1 Kalman filter actually exercises.

For every available candidate in trained_models/, reports:
  - n_test_utt
  - median / p68 / p95 / p99 / p99.9 / max |dx| in micrometres
  - median |dy|, |dtx|, |dty|
  - momentum dependence (median |dx| in 4 |q/p| quartiles)

Outputs:
    experiments/gen_3/results/R7_utt_eval_{date}.json
    stdout summary table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))

from architectures import create_model  # noqa: E402


# --- selection windows -----------------------------------------------------
UT_Z_MIN, UT_Z_MAX = 2300.0, 3000.0       # mm
T_Z_MIN,  T_Z_MAX  = 7600.0, 9500.0       # mm


def _load(exp_dir: Path, device):
    ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location=device)
    cfg = ckpt["config"]
    mt = cfg["model_type"]
    if mt == "mlp":
        model = create_model("mlp", hidden_dims=cfg["hidden_dims"],
                             activation=cfg["activation"],
                             dropout=cfg.get("dropout", 0.0),
                             engineered_features=cfg.get("engineered_features", False))
    elif mt == "pinn_v2":
        model = create_model("pinn_v2", hidden_dims=cfg["hidden_dims"],
                             activation=cfg["activation"],
                             dropout=cfg.get("dropout", 0.0),
                             lambda_pde=cfg.get("lambda_pde", 0.1),
                             lambda_ic=cfg.get("lambda_ic", 0.1),
                             n_collocation=cfg.get("n_collocation", 2),
                             kick_scaled_head=cfg.get("kick_scaled_head", False),
                             pde_scale_mode=cfg.get("pde_scale_mode", "legacy"),
                             pde_ref_length=cfg.get("pde_ref_length", 5213.0))
    else:
        raise ValueError(f"unknown model_type {mt!r}")
    norm = exp_dir / "normalization.json"
    if norm.exists():
        model.load_normalization(str(norm))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model, cfg


def _utt_metrics(model, X, Y, idx, device, batch=200_000):
    """Evaluate UT->T accuracy on (X[idx], Y[idx]) using mask on z_start/z_end."""
    Xt = X[idx]
    Yt = Y[idx]
    z0 = Xt[:, 5]
    zf = z0 + Xt[:, 6]
    m = (z0 >= UT_Z_MIN) & (z0 <= UT_Z_MAX) & (zf >= T_Z_MIN) & (zf <= T_Z_MAX) & (Xt[:, 6] > 0)
    if not m.any():
        return {"error": "no UT->T tracks in test split"}
    Xs = Xt[m].astype(np.float32)
    Ys = Yt[m].astype(np.float32)
    n = Xs.shape[0]

    preds = np.empty((n, 5), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.from_numpy(Xs[i:i + batch]).to(device)
            preds[i:i + batch] = model(xb).cpu().numpy()
    dt_s = time.time() - t0

    dx  = preds[:, 0] - Ys[:, 0]      # mm
    dy  = preds[:, 1] - Ys[:, 1]      # mm
    dtx = preds[:, 2] - Ys[:, 2]      # rad
    dty = preds[:, 3] - Ys[:, 3]      # rad
    abs_dx = np.abs(dx) * 1e3         # -> um

    # momentum-quartile slicing
    qop = np.abs(Xs[:, 4])
    q_edges = np.quantile(qop, [0.25, 0.50, 0.75])
    bins = np.digitize(qop, q_edges)  # 0..3
    median_um_by_q = [float(np.median(abs_dx[bins == k])) if (bins == k).any() else float("nan")
                      for k in range(4)]

    p_GeV = 1.0 / np.maximum(qop, 1e-12)
    return {
        "n_utt": int(n),
        "median_dx_um":  float(np.median(abs_dx)),
        "p68_dx_um":     float(np.quantile(abs_dx, 0.68)),
        "p95_dx_um":     float(np.quantile(abs_dx, 0.95)),
        "p99_dx_um":     float(np.quantile(abs_dx, 0.99)),
        "p99_9_dx_um":   float(np.quantile(abs_dx, 0.999)),
        "max_dx_um":     float(abs_dx.max()),
        "median_dy_um":  float(np.median(np.abs(dy)) * 1e3),
        "p95_dy_um":     float(np.quantile(np.abs(dy), 0.95) * 1e3),
        "median_dtx_urad": float(np.median(np.abs(dtx)) * 1e6),
        "median_dty_urad": float(np.median(np.abs(dty)) * 1e6),
        "median_dx_um_by_|q/p|_quartile": median_um_by_q,
        "qop_quartile_edges": [float(e) for e in q_edges],
        "p_GeV_range": [float(p_GeV.min()), float(p_GeV.max())],
        "throughput_tracks_per_s": float(n / max(dt_s, 1e-9)),
    }


def main() -> None:
    data = np.load(LAB / "data" / "train_10M_gen3.npz")
    X = data["X"]; Y = data["Y"]

    candidates = [
        ("pinn_v2_small_v1",        LAB / "trained_models" / "pinn_v2_small_v1"),    # deployed baseline
        ("pinn_v2_kick_2M_cpu",     LAB / "trained_models" / "pinn_v2_kick_2M_cpu"),
        ("pinn_v2_kick_only_2M_cpu",LAB / "trained_models" / "pinn_v2_kick_only_2M_cpu"),
        ("pinn_v2_kick_10M",        LAB / "trained_models" / "pinn_v2_kick_10M"),
        ("pinn_v2_kick_only_10M",   LAB / "trained_models" / "pinn_v2_kick_only_10M"),
        ("pinn_v2_lam0p1_2M_cpu",   LAB / "trained_models" / "pinn_v2_lam0p1_2M_cpu"),
        ("pinn_v2_lam0_2M_cpu",     LAB / "trained_models" / "pinn_v2_lam0_2M_cpu"),
        ("pinn_v2_lam0p1_10M",      LAB / "trained_models" / "pinn_v2_lam0p1_10M"),
        ("pinn_v2_lam0_10M",        LAB / "trained_models" / "pinn_v2_lam0_10M"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for name, d in candidates:
        if not (d / "best_model.pt").exists():
            print(f"SKIP {name}: no checkpoint")
            continue
        idx_path = d / "test_indices.npy"
        if not idx_path.exists():
            print(f"SKIP {name}: no test_indices.npy")
            continue
        idx = np.load(idx_path)
        print(f"\n== {name} ==  (test split {len(idx)})")
        model, cfg = _load(d, device)
        n_params = sum(p.numel() for p in model.parameters())
        m = _utt_metrics(model, X, Y, idx, device)
        m["params"] = int(n_params)
        m["hidden_dims"] = cfg.get("hidden_dims")
        results[name] = m
        if "error" in m:
            print(f"   {m['error']}")
            continue
        print(f"   n_utt={m['n_utt']}  params={n_params}")
        print(f"   |dx| um  median={m['median_dx_um']:8.2f}  p68={m['p68_dx_um']:8.2f}"
              f"  p95={m['p95_dx_um']:9.1f}  p99={m['p99_dx_um']:9.1f}  max={m['max_dx_um']:9.1f}")
        print(f"   |dy| um  median={m['median_dy_um']:8.2f}  p95={m['p95_dy_um']:9.1f}")
        print(f"   slopes urad  median |dtx|={m['median_dtx_urad']:7.1f}"
              f"  |dty|={m['median_dty_urad']:7.1f}")
        print(f"   median |dx| by |q/p| quartile (low->high): "
              f"{[f'{v:.1f}' for v in m['median_dx_um_by_|q/p|_quartile']]}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # summary
    print("\n" + "=" * 100)
    hdr = f"{'Model':<22} {'params':>7} {'n_utt':>6}  {'med um':>9} {'p68 um':>9} {'p95 um':>9} {'p99 um':>9}"
    print(hdr); print("-" * len(hdr))
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<22} {'-':>7} {'-':>6}  {m['error']}")
            continue
        print(f"{name:<22} {m['params']:>7d} {m['n_utt']:>6d}  "
              f"{m['median_dx_um']:>9.2f} {m['p68_dx_um']:>9.2f} "
              f"{m['p95_dx_um']:>9.1f} {m['p99_dx_um']:>9.1f}")
    print("=" * 100)

    out = REPO / "results" / f"R7_utt_eval_{datetime.now().strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
