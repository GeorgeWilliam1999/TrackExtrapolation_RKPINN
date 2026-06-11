#!/usr/bin/env python3
"""Gen-4 corpus: merge shards + validation gate.

Merge:   data/gen4_shards/shard_*.npz -> data/train_10M_gen4.npz
Gates (all must pass):
  G-INT  integrity   : re-propagate 1000 random tracks with the generator's own
                       RK (v8r1, kappa=1e-3); max |Y_stored - Y_reprop| < 1e-3 mm (float32 storage ulp ~3.6e-4 mm at |x|~3m)
  G-PHY  physics     : median |dtx| on long steps (|dz|>4000) in [0.02, 0.5] rad
  G-POP  population  : >= 8M tracks total; both dz signs present; qop range sane

Exit 0 + 'ALL GATES PASS' on success.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent
sys.path.insert(0, str(GEN3 / "utils"))
sys.path.insert(0, str(HERE))

from generate_data_v2 import rk4  # noqa: E402  (same code that generated the truth)

SHARDS = sorted(glob.glob(str(GEN3 / "data" / "gen4_shards" / "shard_*.npz")))
OUT = GEN3 / "data" / "train_10M_gen4.npz"


def main() -> None:
    print(f"shards found: {len(SHARDS)}")
    Xs, Ys = [], []
    for s in SHARDS:
        d = np.load(s)
        Xs.append(d["X"]); Ys.append(d["Y"])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    print(f"merged: X{X.shape} Y{Y.shape}")

    fails = []
    # G-POP
    n = X.shape[0]
    sgn_pos = float((X[:, 6] > 0).mean())
    qmax = float(np.abs(X[:, 4]).max())
    if n < 8_000_000: fails.append(f"G-POP n={n}")
    if not (0.3 < sgn_pos < 0.7): fails.append(f"G-POP sign balance {sgn_pos:.2f}")
    if not (0.25 < qmax < 0.35): fails.append(f"G-POP qop max {qmax:.3f}")
    print(f"G-POP: n={n:,}  fwd-frac={sgn_pos:.3f}  |qop|max={qmax:.3f}")

    # G-PHY
    lm = np.abs(X[:, 6]) > 4000
    bend = float(np.median(np.abs(Y[lm, 2] - X[lm, 2])))
    if not (0.02 < bend < 0.5): fails.append(f"G-PHY bend {bend:.4f}")
    print(f"G-PHY: median long-step |dtx| = {bend:.4f} rad")

    # G-INT: re-propagate 1000 random tracks
    rng = np.random.default_rng(7)
    idx = rng.choice(n, 1000, replace=False)
    worst = 0.0
    # group by (z0,dz) blocks for vectorised reprop
    sub = X[idx]
    Ysub = Y[idx]
    for key in np.unique(sub[:, 5:7], axis=0):   # exact float32 block keys
        g = (sub[:, 5] == key[0]) & (sub[:, 6] == key[1])
        if not g.any():
            continue
        S = sub[g, :5].astype(np.float64)
        T = rk4(S, float(key[0]), float(key[0]) + float(key[1]))
        worst = max(worst, float(np.nanmax(np.abs(T[:, :4] - Ysub[g, :4].astype(np.float64)))))
    if worst > 1e-3: fails.append(f"G-INT worst diff {worst:.2e} mm")
    print(f"G-INT: worst reprop diff = {worst:.2e} mm (1000 tracks)")

    if fails:
        print("GATES FAILED:", "; ".join(fails)); sys.exit(1)

    np.savez_compressed(OUT, X=X.astype(np.float32), Y=Y.astype(np.float32))
    meta = dict(n=int(n), shards=len(SHARDS), field="v8r1.down", kappa=1e-3,
                fwd_frac=sgn_pos, median_long_dtx=bend, reprop_worst_mm=worst)
    (GEN3 / "data" / "train_10M_gen4.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"ALL GATES PASS -> {OUT}")


if __name__ == "__main__":
    main()
