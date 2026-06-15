#!/usr/bin/env python3
"""Wave-2 three-arm UT->T evaluation: wave-2 NNs vs production extrapUTT vs straight-line.

Extends gates/baseline/compare_gen4_three_arm.py with:
  * the Wave-2 residual/kick checkpoints (auto-discovered: trained_models/wave2_*),
  * a PHYSICAL-MOMENTUM-SPECTRUM-weighted metric (the plane ref is p log-uniform;
    the real LHCb spectrum is steeply falling, so low-p hard tracks are under-
    weighted in the flat metric). Weight w(p) ∝ p^-1 relative to log-uniform
    sampling == assuming physical dN/dp ∝ p^-2 (a steeply-falling model); we
    report weighted median + p95 alongside the flat ones, and the alpha sweep.
  * a second evaluation on the FROZEN UT->T pool (utt_pool_gen4_frozen.npz):
    NN vs straight-line (the >=10x sanity gate; incumbent not scored there -- the
    Python extrapUTT port is parser-only, the incumbent is scored on the plane ref).

Incumbent profile to beat (plane ref, P0.1): median 11 um, low-p quartile 475 um, p95 1.6 mm.

OUTPUT (TE_LAB/paper_p0/):
  wave2_three_arm.json         per-arm metrics on the plane ref
  wave2_three_arm_arrays.npz   per-arm |dx| arrays + qop + p (for the notebook)
  wave2_frozen_pool.json       per-arm metrics on the frozen UT->T pool
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
REF = LAB / "paper_p0"                 # where plane_ref_v8r1.npz + poly csv live
TM = LAB / "trained_models"
sys.path.insert(0, str(REPO / "models"))
from architectures import create_model  # noqa: E402

ZINI, ZFIN = 2665.0, 7826.0
DZ = ZFIN - ZINI
C_QP = 0.299792458

# wave-1 gen-4 (reference) + auto-discovered wave-2 runs
WAVE1 = ["pinn_v2_g4_lam0_10M", "pinn_v2_g4_lam0p1_10M"]


def discover_models():
    w2 = sorted(p.name for p in TM.glob("wave2_*") if (p / "best_model.pt").exists())
    w1 = [m for m in WAVE1 if (TM / m / "best_model.pt").exists()]
    return w1 + w2


def load_model(name: str):
    d = TM / name
    ckpt = torch.load(d / "best_model.pt", weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    model = create_model("pinn_v2", hidden_dims=cfg["hidden_dims"],
                         activation=cfg["activation"], dropout=cfg.get("dropout", 0.0),
                         lambda_pde=cfg.get("lambda_pde", 0.0), lambda_ic=cfg.get("lambda_ic", 0.0),
                         n_collocation=cfg.get("n_collocation", 2),
                         kick_scaled_head=cfg.get("kick_scaled_head", False),
                         pde_scale_mode=cfg.get("pde_scale_mode", "legacy"),
                         pde_ref_length=cfg.get("pde_ref_length", 5213.0))
    if (d / "normalization.json").exists():
        model.load_normalization(str(d / "normalization.json"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def wquantile(v, w, q):
    """Weighted quantile."""
    o = np.argsort(v); v = v[o]; w = w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return float(np.interp(q, c, v))


def metrics(pred, Y, qop, p, spec_alpha=1.0):
    dx = np.abs(pred[:, 0] - Y[:, 0]) * 1e3       # um
    dtx = np.abs(pred[:, 2] - Y[:, 2]) * 1e6      # urad
    q = np.abs(qop)
    edges = np.quantile(q, [0.25, 0.5, 0.75])
    b = np.digitize(q, edges)                      # 0 (high p) .. 3 (low p)
    byq = [float(np.median(dx[b == k])) for k in range(4)]
    # physical-spectrum weights (relative to log-uniform): w ∝ p^-alpha
    w = p ** (-spec_alpha); w /= w.sum()
    out = dict(median_dx_um=float(np.median(dx)),
               p68_dx_um=float(np.quantile(dx, 0.68)),
               p95_dx_um=float(np.quantile(dx, 0.95)),
               p99_dx_um=float(np.quantile(dx, 0.99)),
               median_dtx_urad=float(np.median(dtx)),
               median_dx_um_by_qop_quartile_hi2lo_p=byq,
               spec_weighted_median_dx_um=wquantile(dx, w, 0.50),
               spec_weighted_p95_dx_um=wquantile(dx, w, 0.95),
               spec_weighted_median_by_alpha={
                   f"{a}": wquantile(dx, (p ** (-a)) / (p ** (-a)).sum(), 0.50)
                   for a in (0.5, 1.0, 1.5)})
    return out, dx


def straight(X):
    return np.stack([X[:, 0] + X[:, 2] * DZ, X[:, 1] + X[:, 3] * DZ, X[:, 2], X[:, 3]], axis=1)


def run_plane_ref():
    d = np.load(REF / "plane_ref_v8r1.npz")
    X, Y = d["X_plane"], d["Y_true"]
    qop = X[:, 4]; p = C_QP / np.abs(qop)
    poly = np.loadtxt(REF / "plane_poly_v8r1_polm1.csv", delimiter=",", skiprows=1)
    assert poly.shape[0] == X.shape[0]

    results, arrays = {}, {"qop": qop, "p_GeV": p}
    results["extrapUTT (incumbent)"], arrays["extrapUTT"] = metrics(poly, Y, qop, p)
    sl = straight(X)
    results["straight_line"], arrays["straight_line"] = metrics(sl, Y, qop, p)

    Xin = np.concatenate([X[:, :5], np.full((X.shape[0], 1), ZINI),
                          np.full((X.shape[0], 1), DZ)], axis=1).astype(np.float32)
    with torch.no_grad():
        for name in discover_models():
            model, npar = load_model(name)
            pred = model(torch.from_numpy(Xin)).numpy()
            results[name], arrays[name] = metrics(pred, Y, qop, p)
            results[name]["params"] = int(npar)

    print(f"\n== PLANE REF (z {ZINI:.0f}->{ZFIN:.0f}, {X.shape[0]} tracks, p[{p.min():.0f},{p.max():.0f}]GeV) ==")
    h = (f"{'arm':<26}{'params':>8}{'med um':>9}{'p95':>10}{'specMed':>9}{'specP95':>10}"
         f"   byQ hi->lo p [um]")
    print(h); print("-" * len(h))
    for k, m in results.items():
        bq = "/".join(f"{v:.0f}" for v in m["median_dx_um_by_qop_quartile_hi2lo_p"])
        pr = m.get("params", "")
        print(f"{k:<26}{str(pr):>8}{m['median_dx_um']:>9.1f}{m['p95_dx_um']:>10.1f}"
              f"{m['spec_weighted_median_dx_um']:>9.1f}{m['spec_weighted_p95_dx_um']:>10.1f}   [{bq}]")
    (REF / "wave2_three_arm.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(REF / "wave2_three_arm_arrays.npz", **arrays)
    print("wrote wave2_three_arm.json + _arrays.npz")
    return results


def run_frozen_pool():
    f = np.load(LAB / "data" / "utt_pool_gen4_frozen.npz")
    X, Y = f["X"], f["Y"]
    qop = X[:, 4]; p = C_QP / np.abs(qop)
    results = {}
    results["straight_line"], _ = metrics(straight_frozen(X), Y, qop, p)
    with torch.no_grad():
        for name in discover_models():
            model, npar = load_model(name)
            pred = model(torch.from_numpy(X.astype(np.float32))).numpy()
            results[name], _ = metrics(pred, Y, qop, p)
            results[name]["params"] = int(npar)
    print(f"\n== FROZEN UT->T POOL ({X.shape[0]} tracks from gen-4) ==")
    h = f"{'arm':<26}{'med um':>9}{'p95 um':>10}{'specMed':>9}   byQ hi->lo p [um]"
    print(h); print("-" * len(h))
    for k, m in results.items():
        bq = "/".join(f"{v:.0f}" for v in m["median_dx_um_by_qop_quartile_hi2lo_p"])
        print(f"{k:<26}{m['median_dx_um']:>9.1f}{m['p95_dx_um']:>10.1f}"
              f"{m['spec_weighted_median_dx_um']:>9.1f}   [{bq}]")
    (REF / "wave2_frozen_pool.json").write_text(json.dumps(results, indent=2))
    print("wrote wave2_frozen_pool.json")
    return results


def straight_frozen(X):
    dz = X[:, 6]
    return np.stack([X[:, 0] + X[:, 2] * dz, X[:, 1] + X[:, 3] * dz, X[:, 2], X[:, 3]], axis=1)


if __name__ == "__main__":
    run_plane_ref()
    run_frozen_pool()
