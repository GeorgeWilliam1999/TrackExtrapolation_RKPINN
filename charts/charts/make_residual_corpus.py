#!/usr/bin/env python3
"""F2 prep — build the residual corpus: Y_res = Y - chart_baseline.

The analytic chart already captures the bend; the network only has to learn the
small smooth residual. We take a stratified 2M subset (matching train.py's
seed/stratification), compute the chart baseline in chunks, and save
residual_2M.npz with Y_res = [res_x, res_y, res_tx, res_ty, qop_passthrough].
At eval time the chart is added back.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
REPO = FLAT.parent
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
LAB_FLAT = LAB.parent / "flattening"
sys.path.insert(0, str(HERE))
from chart import chart_predict, load_chart  # noqa: E402

SEED, MAXN, CHUNK = 42, 2_000_000, 200_000


def main():
    d = np.load(LAB / "data" / "train_10M_gen3.npz")
    X, Y, P = d["X"], d["Y"], d["P"]
    N = X.shape[0]
    # stratified-on-sign(dz) subsample, mirroring train.py
    rng = np.random.default_rng(SEED)
    sdz = np.sign(X[:, 6]).astype(np.int8)
    ip = np.flatnonzero(sdz > 0); ineg = np.flatnonzero(sdz < 0)
    rng.shuffle(ip); rng.shuffle(ineg)
    ne = min(MAXN // 2, len(ip), len(ineg))
    keep = np.concatenate([ip[:ne], ineg[:ne]]); rng.shuffle(keep)
    Xs, Ys, Ps = X[keep].astype(np.float32), Y[keep].astype(np.float32), P[keep].astype(np.float32)
    print(f"subset {len(keep):,} (stratified). computing chart baseline in chunks...")

    chart = load_chart()
    base = np.empty((len(Xs), 4), np.float64)
    for i in range(0, len(Xs), CHUNK):
        base[i:i+CHUNK] = chart_predict(Xs[i:i+CHUNK], chart)
        if (i // CHUNK) % 5 == 0:
            print(f"  {i+CHUNK:,}/{len(Xs):,}")

    Yres = Ys.copy()
    Yres[:, :4] = Ys[:, :4] - base.astype(np.float32)      # residual target
    # diagnostics on the residual magnitude
    rx = np.abs(Yres[:, 0]) * 1e3
    print(f"residual |x|: median {np.median(rx):.1f} um  p95 {np.quantile(rx,.95):.0f} um  "
          f"(raw |dx_straight| median was ~mm-scale)")

    out = LAB_FLAT / "data"; out.mkdir(exist_ok=True)
    np.savez(out / "residual_2M.npz", X=Xs, Y=Yres, P=Ps,
             chart_baseline=base.astype(np.float32), keep_idx=keep)
    print(f"saved -> {out/'residual_2M.npz'}  (X, Y=residual, chart_baseline, keep_idx)")


if __name__ == "__main__":
    main()
