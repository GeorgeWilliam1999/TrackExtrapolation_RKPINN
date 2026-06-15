#!/usr/bin/env python3
"""Wave-2 UT->T-focused corpus generator (deployment-weighted, high statistics).

Same physics contract as datagen/generate_data_v2.py (v8r1 field, kappa=1e-3,
vectorised RK4 5mm step, raw MagDown sign) but RESTRATIFIED to the deployment
regime the network is actually judged on:

  geometry  : z0 in [2300,3000] (UT region), zf in [7600,9500] (SciFi T), dz>0
              -> every track traverses the magnet (the hard long-forward step)
  population: 100% PV-pointing (the production population): sample at PV z=0
              (x,y in +-0.5mm, tx in +-0.27, ty in +-0.22), RK to z0 through field
  momentum  : LOW-P WEIGHTED mixture 50% logU[1,10] / 30% logU[10,50] / 20% logU[50,200]
              (general gen-4 is log-uniform[1,200]; this oversamples the hard low-p tail)
  spatial   : capped to the LHCb acceptance (|x|<3000, |y|<2500 at both z0 and zf)
  layout    : blocks of 500 share (z0,zf) so propagation vectorises; X[N,7], Y[N,5]

This is the gen-4 "UT->T-focused high-statistics corpus" the Wave-2 plan asks to
ADD alongside (not replace) the general train_10M_gen4.npz.

Usage: generate_utt_focused.py <shard_id> <n_tracks> <out_dir>
Seed : 17000 + shard_id*7919  (distinct namespace from the general gen, 9000+).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
from field_v8r1 import FieldV8R1  # noqa: E402
from generate_data_v2 import rk4  # noqa: E402  (identical truth dynamics: v8r1, kappa=1e-3)

C_QP = 0.299792458
BLOCK = 500

# deployment windows (match gates/run_r7_utt_eval.py + utt_pool_gen4_frozen)
Z0_LO, Z0_HI = 2300.0, 3000.0
ZF_LO, ZF_HI = 7600.0, 9500.0
# LHCb acceptance cap (mm) -- tighter than the +-3900 field-map edge
X_CAP, Y_CAP = 3000.0, 2500.0


def _sample_p(rng, n):
    """Low-p-weighted momentum [GeV]: 50% [1,10] / 30% [10,50] / 20% [50,200]."""
    u = rng.random(n)
    p = np.empty(n)
    a = u < 0.5
    b = (u >= 0.5) & (u < 0.8)
    c = u >= 0.8
    p[a] = np.exp(rng.uniform(np.log(1.0),  np.log(10.0),  a.sum()))
    p[b] = np.exp(rng.uniform(np.log(10.0), np.log(50.0),  b.sum()))
    p[c] = np.exp(rng.uniform(np.log(50.0), np.log(200.0), c.sum()))
    return p


def main() -> None:
    shard, n_tracks, out_dir = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    rng = np.random.default_rng(17000 + shard * 7919)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    Xs, Ys = [], []
    n_blocks = max(1, n_tracks // BLOCK)
    for b in range(n_blocks):
        z0 = float(rng.uniform(Z0_LO, Z0_HI))
        zf = float(rng.uniform(ZF_LO, ZF_HI))
        dz = zf - z0
        if dz < 25.0:
            continue

        p = _sample_p(rng, BLOCK)
        qop = rng.choice([-1.0, 1.0], BLOCK) * C_QP / p
        tx = rng.uniform(-0.27, 0.27, BLOCK)
        ty = rng.uniform(-0.22, 0.22, BLOCK)

        # PV-pointing: start at z=0 with small transverse offset, RK to z0 through field.
        zpv = rng.uniform(-50.0, 50.0)
        S0 = np.stack([rng.uniform(-0.5, 0.5, BLOCK), rng.uniform(-0.5, 0.5, BLOCK),
                       tx, ty, qop], axis=1)
        S = rk4(S0, zpv, z0)

        ok = (np.all(np.isfinite(S), axis=1) & (np.abs(S[:, 0]) < X_CAP)
              & (np.abs(S[:, 1]) < Y_CAP) & (np.abs(S[:, 2]) < 1.0) & (np.abs(S[:, 3]) < 1.0))
        S = S[ok]
        if S.shape[0] == 0:
            continue
        T = rk4(S, z0, zf)
        ok2 = (np.all(np.isfinite(T), axis=1) & (np.abs(T[:, 0]) < X_CAP)
               & (np.abs(T[:, 1]) < Y_CAP) & (np.abs(T[:, 2]) < 1.0) & (np.abs(T[:, 3]) < 1.0))
        S, T = S[ok2], T[ok2]
        n = S.shape[0]
        if n == 0:
            continue
        Xs.append(np.concatenate([S[:, :5], np.full((n, 1), z0), np.full((n, 1), dz)], axis=1))
        Ys.append(np.concatenate([T[:, :4], S[:, 4:5]], axis=1))

    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    long_m = np.abs(X[:, 6]) > 4000
    bend = float(np.median(np.abs(Y[long_m, 2] - X[long_m, 2]))) if long_m.any() else -1.0
    P = (C_QP / np.abs(X[:, 4])).astype(np.float32)
    meta = dict(shard=shard, n=int(X.shape[0]), kappa=1e-3, field="v8r1.down",
                regime="UT->T focused (z0 2300-3000, zf 7600-9500, pointing, low-p weighted)",
                polarity_convention="raw MagDown By<0; extrapUTT m_polarity=-1",
                seed=17000 + shard * 7919, median_long_step_dtx_rad=bend,
                frac_1to3GeV=float(((P >= 1) & (P <= 3)).mean()),
                wall_s=round(time.time() - t0, 1))
    np.savez_compressed(out_dir / f"utt_shard_{shard:04d}.npz", X=X, Y=Y, P=P)
    (out_dir / f"utt_shard_{shard:04d}.json").write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
