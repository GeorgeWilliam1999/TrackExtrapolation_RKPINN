#!/usr/bin/env python3
"""Wave-2 restratification: build the deployment-weighted training corpus + gates.

Outputs (in TE_LAB/data/):
  utt_focused_gen4.npz    merge of the UT->T-focused shards (the standalone
                          high-statistics UT->T corpus the plan asks to ADD)
  train_wave2_deploy.npz  deployment-weighted MIX:
                            general gen-4 (acceptance-capped) + UT->T-focused,
                            shuffled, with UT->T >= 10% (vs 0.145% in gen-4)
  train_wave2_deploy.meta.json / .schema.json   provenance + per-column schema

Restratification rules (Wave-2 plan step 1):
  * OVERSAMPLE the deployment regime: the focused corpus is 100% UT->T,
    long-forward (magnet-traversing), low-p weighted.  Mixed in so UT->T >= 10%.
  * CAP spatial extremes to the LHCb acceptance: keep only general-corpus tracks
    with |x|<3000 & |y|<2500 mm at BOTH z0 and z0+dz (drops the +-3900 field-map
    edge population that dilutes the realistic central region).
  * KEEP gen-4 as the "general" set (broad dz/z0/p coverage) -- it is not removed.

Validation gates (merge_validate_v2 style, all must pass):
  G-INT  re-propagate 1000 random tracks with the generator RK; worst |dY| < 1e-3 mm
  G-PHY  median |dtx| on long steps (|dz|>4000) in [0.02, 0.5] rad
  G-POP  >= 4M rows; both dz signs present; |qop| range sane
  G-STRAT  UT->T fraction >= 10%; spatial cap respected
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))
from generate_data_v2 import rk4  # noqa: E402  (same RK that produced the truth)

import os
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
DATA = LAB / "data"

C_QP = 0.299792458
X_CAP, Y_CAP = 3000.0, 2500.0       # LHCb acceptance cap (mm)
N_GENERAL = 4_000_000               # capped general tracks to mix in
SEED = 20260614


def utt_mask(X: np.ndarray) -> np.ndarray:
    z0 = X[:, 5]; zf = z0 + X[:, 6]
    return (z0 >= 2300) & (z0 <= 3000) & (zf >= 7600) & (zf <= 9500) & (X[:, 6] > 0)


def accept_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return ((np.abs(X[:, 0]) < X_CAP) & (np.abs(X[:, 1]) < Y_CAP)
            & (np.abs(Y[:, 0]) < X_CAP) & (np.abs(Y[:, 1]) < Y_CAP))


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ---- 1. merge the UT->T-focused shards -> standalone high-stat corpus ----
    shards = sorted(glob.glob(str(DATA / "utt_focused_shards" / "utt_shard_*.npz")))
    assert shards, "no UT->T-focused shards found -- run datagen_utt_focused.sub first"
    Xs, Ys = [], []
    for s in shards:
        d = np.load(s); Xs.append(d["X"]); Ys.append(d["Y"])
    Xf = np.concatenate(Xs); Yf = np.concatenate(Ys)
    Pf = (C_QP / np.abs(Xf[:, 4])).astype(np.float32)
    np.savez_compressed(DATA / "utt_focused_gen4.npz", X=Xf, Y=Yf, P=Pf)
    print(f"focused: merged {len(shards)} shards -> {Xf.shape[0]:,} tracks "
          f"(UT->T={100*utt_mask(Xf).mean():.1f}%)")

    # ---- 2. general gen-4, acceptance-capped, subsampled ----
    g = np.load(DATA / "train_10M_gen4.npz")
    Xg, Yg, Pg = g["X"], g["Y"], g["P"]
    keep = accept_mask(Xg, Yg)
    print(f"general: {Xg.shape[0]:,} -> {keep.sum():,} after acceptance cap "
          f"(|x|<{X_CAP:.0f},|y|<{Y_CAP:.0f} at z0 & zf; dropped {100*(1-keep.mean()):.1f}%)")
    gi = np.flatnonzero(keep)
    rng.shuffle(gi)
    gi = gi[:N_GENERAL]
    Xg, Yg, Pg = Xg[gi], Yg[gi], Pg[gi]

    # ---- 3. mix + shuffle ----
    X = np.concatenate([Xg, Xf]).astype(np.float32)
    Y = np.concatenate([Yg, Yf]).astype(np.float32)
    P = np.concatenate([Pg, Pf]).astype(np.float32)
    perm = rng.permutation(X.shape[0])
    X, Y, P = X[perm], Y[perm], P[perm]
    n = X.shape[0]
    utt_frac = float(utt_mask(X).mean())
    print(f"mixed: {n:,} tracks  UT->T={100*utt_frac:.1f}%  "
          f"fwd={100*(X[:,6]>0).mean():.1f}%")

    # ---- gates ----
    fails = []
    if n < 4_000_000: fails.append(f"G-POP n={n}")
    sgn_pos = float((X[:, 6] > 0).mean())
    if not (0.2 < sgn_pos < 0.95): fails.append(f"G-POP sign {sgn_pos:.2f}")
    qmax = float(np.abs(X[:, 4]).max())
    if not (0.25 < qmax < 0.35): fails.append(f"G-POP |qop|max {qmax:.3f}")
    if utt_frac < 0.10: fails.append(f"G-STRAT UT->T frac {utt_frac:.4f} < 0.10")
    if not accept_mask(X, Y).all(): fails.append("G-STRAT acceptance cap violated")

    lm = np.abs(X[:, 6]) > 4000
    bend = float(np.median(np.abs(Y[lm, 2] - X[lm, 2])))
    if not (0.02 < bend < 0.5): fails.append(f"G-PHY bend {bend:.4f}")
    print(f"G-PHY: median long-step |dtx| = {bend:.4f} rad")

    idx = rng.choice(n, 1000, replace=False)
    sub, Ysub = X[idx], Y[idx]
    worst = 0.0
    for key in np.unique(sub[:, 5:7], axis=0):
        gmask = (sub[:, 5] == key[0]) & (sub[:, 6] == key[1])
        S = sub[gmask, :5].astype(np.float64)
        T = rk4(S, float(key[0]), float(key[0]) + float(key[1]))
        worst = max(worst, float(np.nanmax(np.abs(T[:, :4] - Ysub[gmask, :4].astype(np.float64)))))
    if worst > 1e-3: fails.append(f"G-INT worst {worst:.2e} mm")
    print(f"G-INT: worst reprop diff = {worst:.2e} mm (1000 tracks)")
    print(f"G-POP: n={n:,} fwd={sgn_pos:.3f} |qop|max={qmax:.3f}")
    print(f"G-STRAT: UT->T={100*utt_frac:.1f}%  acceptance-capped OK")

    if fails:
        print("GATES FAILED:", "; ".join(fails)); sys.exit(1)

    out = DATA / "train_wave2_deploy.npz"
    np.savez_compressed(out, X=X, Y=Y, P=P)
    meta = dict(n=int(n), n_general_capped=int(Xg.shape[0]), n_utt_focused=int(Xf.shape[0]),
                utt_fraction=utt_frac, fwd_frac=sgn_pos, field="v8r1.down", kappa=1e-3,
                x_cap_mm=X_CAP, y_cap_mm=Y_CAP, median_long_dtx=bend, reprop_worst_mm=worst,
                focused_corpus="utt_focused_gen4.npz", general_corpus="train_10M_gen4.npz",
                built=time.strftime("%Y-%m-%d %H:%M"), wall_s=round(time.time() - t0, 1))
    (DATA / "train_wave2_deploy.meta.json").write_text(json.dumps(meta, indent=1))

    schema = {
        "file": "train_wave2_deploy.npz",
        "keys": {"X": "[N,7] float32", "Y": "[N,5] float32", "P": "[N] float32"},
        "X_columns": ["x@z0 mm", "y@z0 mm", "tx", "ty", "qop=c*q/p[1/GeV]",
                      "z0 mm [0..14000]", "dz signed mm"],
        "Y_columns": ["x@z0+dz mm", "y mm", "tx", "ty", "qop (=X[:,4])"],
        "P": "p[GeV] = 0.299792458/|qop|",
        "physics": {"kappa": "1e-3*qop", "field": "v8r1 down (raw MagDown By<0)",
                    "truth": "vectorised RK4 5mm", "polarity": "extrapUTT m_polarity=-1"},
        "restratification": {
            "composition": f"{int(Xg.shape[0])} general (acceptance-capped) + "
                           f"{int(Xf.shape[0])} UT->T-focused",
            "utt_fraction": round(utt_frac, 4),
            "acceptance_cap_mm": {"x": X_CAP, "y": Y_CAP},
            "vs_gen4": "UT->T 0.145% -> {:.1f}%; spatial extremes capped".format(100*utt_frac)},
        "gates": {"G-INT_mm": worst, "G-PHY_long_dtx_rad": bend, "G-POP_n": int(n),
                  "G-STRAT_utt_frac": round(utt_frac, 4)},
    }
    (DATA / "train_wave2_deploy.schema.json").write_text(json.dumps(schema, indent=2))
    print(f"\nALL GATES PASS -> {out}\n  wrote meta + schema json")


if __name__ == "__main__":
    main()
