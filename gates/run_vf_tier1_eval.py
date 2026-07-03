#!/usr/bin/env python3
"""VF tier-1 — per-leg accuracy of the vertex-fit-corpus retrains (P2c).

Evaluates each experiment under experiments/vertexfit/trained_models/ on ITS
OWN held-out test split (test_indices.npy into the merged corpus), broken out
by leg:

    A (0) = UT-band [2300,2700] -> vertex basin [0,2300]   (backward, field-heavy)
    B (1) = T-band  [7500,9410] -> vertex basin [0,2300]   (backward, full magnet)
    C (2) = VELO    [0,770]     -> [-200,800]              (near field-free)
    D (3) = intra-band +-U(10,500) mm                      (short local steps)

Reports median / p95 of |dx|, |dy| (um) and |dtx|, |dty| (urad) per leg, and
the same for the STRAIGHT-LINE transport (what ParticleVertexFitter's Newton
loop uses inside an iteration) as the context baseline the NN must beat where
the field matters.

Spec bars (PLAN 3915d544-b9d9-81b7 v1, section 6, tier 1):
    leg A:  median |dx| <= 100 um,  p95 <= 1 mm,  median slope err <= 50 urad
    STOP-RULE (P2): if leg-A median >> 1 mm, halt and report.

Outputs: experiments/vertexfit/results/VF_tier1_{date}.json + stdout table.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LAB = Path(os.environ.get("VF_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit"))
CORPUS = LAB / "data" / "vf_corpus_10M.npz"
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))

from train import _build_model  # noqa: E402

LEG_NAMES = {0: "A UT->vtx", 1: "B T->vtx", 2: "C VELO", 3: "D intra"}
BARS_A = {"med_dx_um": 100.0, "p95_dx_um": 1000.0, "med_slope_urad": 50.0}


def _q(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q))


def _leg_stats(err4: np.ndarray) -> dict:
    """err4[N,4] = (pred - truth) for [x, y, tx, ty]."""
    ax, ay = np.abs(err4[:, 0]), np.abs(err4[:, 1])
    atx, aty = np.abs(err4[:, 2]), np.abs(err4[:, 3])
    return {
        "n": int(err4.shape[0]),
        "med_dx_um": _q(ax, 0.5) * 1e3, "p95_dx_um": _q(ax, 0.95) * 1e3,
        "med_dy_um": _q(ay, 0.5) * 1e3, "p95_dy_um": _q(ay, 0.95) * 1e3,
        "med_dtx_urad": _q(atx, 0.5) * 1e6, "p95_dtx_urad": _q(atx, 0.95) * 1e6,
        "med_dty_urad": _q(aty, 0.5) * 1e6, "p95_dty_urad": _q(aty, 0.95) * 1e6,
    }


@torch.no_grad()
def _predict(model, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i : i + batch])
        outs.append(model(xb).numpy())
    return np.concatenate(outs)


def eval_experiment(exp_dir: Path, X: np.ndarray, Y: np.ndarray, LEG: np.ndarray) -> dict:
    ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location="cpu")
    config = ckpt["config"]
    model = _build_model(config)
    model.load_normalization(str(exp_dir / "normalization.json"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx_path = exp_dir / "test_indices.npy"
    if idx_path.exists():
        idx = np.load(idx_path)
    else:
        # Training was cut before the end-of-run dump (session teardown,
        # 2026-07-03): rebuild the split deterministically — same seed, same
        # code path as train.stratified_split with balance_sign=False and
        # max_samples >= N — then PROVE it by matching the stored train-split
        # normalisation before trusting it.
        N = X.shape[0]
        rng = np.random.default_rng(config["seed"])
        keep = rng.permutation(N)[: min(int(config.get("max_samples") or N), N)]
        n_train = int(len(keep) * config["train_fraction"])
        n_val = int(len(keep) * config["val_fraction"])
        # torch's fp32 mean is accumulation-order-sensitive, so recomputing it
        # over the reconstructed index order and matching the stored value
        # fingerprints the exact permutation (a statistical mean check cannot:
        # any 80% subset has the same mean to ~std/sqrt(N)).
        mu = torch.from_numpy(X[keep[:n_train]]).mean(dim=0).numpy()
        stored = model.input_mean.numpy()
        diff = np.max(np.abs(mu - stored))
        assert diff < 1e-6, f"split reconstruction failed: fp32 mean fingerprint off by {diff:.2e}"
        print(f"  reconstructed split verified: fp32 mean fingerprint matches (max diff {diff:.1e})")
        idx = keep[n_train + n_val :]
        np.save(idx_path, idx)
    Xt, Yt, Lt = X[idx], Y[idx], LEG[idx]

    Yp = _predict(model, Xt)
    err_nn = Yp[:, :4] - Yt[:, :4]

    # straight-line context baseline: x += tx*dz, y += ty*dz, slopes frozen
    dz = Xt[:, 6:7]
    Ysl = np.concatenate(
        [Xt[:, 0:1] + Xt[:, 2:3] * dz, Xt[:, 1:2] + Xt[:, 3:4] * dz, Xt[:, 2:4]], axis=1
    )
    err_sl = Ysl - Yt[:, :4]

    out = {"experiment": exp_dir.name, "epoch": int(ckpt["epoch"]),
           "params": int(sum(p.numel() for p in model.parameters())),
           "n_test": int(len(idx)), "legs": {}}
    for code, name in LEG_NAMES.items():
        m = Lt == code
        if m.sum() == 0:
            continue
        out["legs"][name] = {"nn": _leg_stats(err_nn[m]), "straight": _leg_stats(err_sl[m])}

    a = out["legs"]["A UT->vtx"]["nn"]
    out["bars_leg_A"] = {
        "med_dx_um": {"value": a["med_dx_um"], "bar": BARS_A["med_dx_um"],
                      "pass": a["med_dx_um"] <= BARS_A["med_dx_um"]},
        "p95_dx_um": {"value": a["p95_dx_um"], "bar": BARS_A["p95_dx_um"],
                      "pass": a["p95_dx_um"] <= BARS_A["p95_dx_um"]},
        "med_slope_urad": {"value": max(a["med_dtx_urad"], a["med_dty_urad"]),
                           "bar": BARS_A["med_slope_urad"],
                           "pass": max(a["med_dtx_urad"], a["med_dty_urad"]) <= BARS_A["med_slope_urad"]},
    }
    out["stop_rule_triggered"] = bool(a["med_dx_um"] > 1000.0)  # leg-A median >> 1 mm
    return out


def main() -> None:
    with np.load(CORPUS) as d:
        X = d["X"].astype(np.float32)
        Y = d["Y"].astype(np.float32)
        LEG = d["LEG"]

    results = []
    for exp_dir in sorted((LAB / "trained_models").iterdir()):
        if not (exp_dir / "best_model.pt").exists():
            continue
        print(f"\n=== {exp_dir.name} ===")
        r = eval_experiment(exp_dir, X, Y, LEG)
        results.append(r)
        hdr = f"{'leg':>10s} {'n':>7s} | {'med|dx|':>9s} {'p95|dx|':>9s} {'med|dy|':>9s} | {'med|dtx|':>9s} {'med|dty|':>9s} |  straight med|dx|"
        print(hdr)
        for name, s in r["legs"].items():
            nn, sl = s["nn"], s["straight"]
            print(f"{name:>10s} {nn['n']:>7d} | {nn['med_dx_um']:>7.1f}um {nn['p95_dx_um']:>7.1f}um "
                  f"{nn['med_dy_um']:>7.1f}um | {nn['med_dtx_urad']:>5.1f}urad {nn['med_dty_urad']:>5.1f}urad "
                  f"|  {sl['med_dx_um']:>10.1f}um")
        for k, v in r["bars_leg_A"].items():
            print(f"  bar leg-A {k}: {v['value']:.1f} vs {v['bar']:.0f}  ->  {'PASS' if v['pass'] else 'FAIL'}")
        if r["stop_rule_triggered"]:
            print("  *** STOP-RULE TRIGGERED: leg-A median |dx| > 1 mm ***")

    (LAB / "results").mkdir(exist_ok=True)
    out_path = LAB / "results" / f"VF_tier1_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
