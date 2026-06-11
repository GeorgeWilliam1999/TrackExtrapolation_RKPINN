#!/usr/bin/env python3
"""Gen-4 corpus generator — physical kappa, canonical v8r1 field, LHCb-convention population.

Contract (P0.0 regeneration, 2026-06-11):
  field    : LHCb FieldMap v8r1 down (CVMFS magfield.bin Allen consumes), raw sign (MagDown By<0)
  kappa    : 1e-3 * qop  (qop = 0.299792458 * q/p[1/GeV] = Allen c*q/p)   [P0.0 fix]
  popn     : 70% PV-pointing (|z_pv|<50mm, +-0.5mm transverse — the production population)
             30% broad non-pointing (x in +-1000, y in +-800 — Kalman intermediate-state cover)
  steps    : signed dz, |dz| log-uniform [25,10000] mm, sign 50/50; z0 uniform [0,14000];
             zf clipped into [-400,13900]
  momentum : p log-uniform [1,200] GeV, both charges
  truth    : vectorised RK4, 5mm fixed step (same EOM code validated vs extrapUTT at 15um)
  layout   : blocks of 500 tracks share (z0,dz) so propagation vectorises; X[N,7], Y[N,5]

Usage: generate_data_v2.py <shard_id> <n_tracks> <out_dir>
Seed: 9000 + shard_id*7919.
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

KAPPA = 1.0e-3
C_QP = 0.299792458
STEP = 5.0
BLOCK = 500
FIELD = FieldV8R1()


def deriv(S, z):
    x, y, tx, ty, qop = S.T
    Bx, By, Bz = FIELD(x, y, np.full_like(x, z))
    k = KAPPA * qop
    N = np.sqrt(1 + tx * tx + ty * ty)
    return np.stack([tx, ty,
                     k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
                     k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
                     np.zeros_like(x)], axis=1)


def rk4(S, z0, z1, step=STEP):
    S = S.astype(np.float64).copy()
    z = float(z0)
    if z1 == z0:
        return S
    h = step if z1 > z0 else -step
    while (z1 - z) * np.sign(h) > abs(h):
        k1 = deriv(S, z); k2 = deriv(S + 0.5 * h * k1, z + 0.5 * h)
        k3 = deriv(S + 0.5 * h * k2, z + 0.5 * h); k4 = deriv(S + h * k3, z + h)
        S = S + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4); z += h
    r = z1 - z
    if abs(r) > 1e-12:
        k1 = deriv(S, z); k2 = deriv(S + 0.5 * r * k1, z + 0.5 * r)
        k3 = deriv(S + 0.5 * r * k2, z + 0.5 * r); k4 = deriv(S + r * k3, z + r)
        S = S + (r / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def main() -> None:
    shard, n_tracks, out_dir = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    rng = np.random.default_rng(9000 + shard * 7919)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    Xs, Ys = [], []
    n_blocks = n_tracks // BLOCK
    for b in range(n_blocks):
        z0 = rng.uniform(0.0, 14000.0)
        absdz = np.exp(rng.uniform(np.log(25.0), np.log(10000.0)))
        sgn = rng.choice([-1.0, 1.0])
        zf = float(np.clip(z0 + sgn * absdz, -400.0, 13900.0))
        dz = zf - z0
        if abs(dz) < 25.0:
            continue

        p = np.exp(rng.uniform(np.log(1.0), np.log(200.0), BLOCK))
        qop = rng.choice([-1.0, 1.0], BLOCK) * C_QP / p
        tx = rng.uniform(-0.30, 0.30, BLOCK)
        ty = rng.uniform(-0.25, 0.25, BLOCK)

        pointing = b % 10 < 7  # 70/30 mix
        if pointing:
            zpv = rng.uniform(-50.0, 50.0, BLOCK).mean()  # block-shared PV plane
            S = np.stack([rng.uniform(-0.5, 0.5, BLOCK), rng.uniform(-0.5, 0.5, BLOCK),
                          tx, ty, qop], axis=1)
            S = rk4(S, zpv, z0)  # PV -> z0 through the field
        else:
            S = np.stack([rng.uniform(-1000.0, 1000.0, BLOCK), rng.uniform(-800.0, 800.0, BLOCK),
                          tx, ty, qop], axis=1)

        ok = np.all(np.isfinite(S), axis=1) & (np.abs(S[:, 0]) < 3900) & (np.abs(S[:, 1]) < 3900) \
             & (np.abs(S[:, 2]) < 1.0) & (np.abs(S[:, 3]) < 1.0)
        S = S[ok]
        if S.shape[0] == 0:
            continue
        T = rk4(S, z0, zf)
        ok2 = np.all(np.isfinite(T), axis=1) & (np.abs(T[:, 0]) < 3900) & (np.abs(T[:, 1]) < 3900) \
              & (np.abs(T[:, 2]) < 1.0) & (np.abs(T[:, 3]) < 1.0)
        S, T = S[ok2], T[ok2]
        n = S.shape[0]
        if n == 0:
            continue
        Xs.append(np.concatenate([S[:, :5], np.full((n, 1), z0), np.full((n, 1), dz)], axis=1))
        Ys.append(np.concatenate([T[:, :4], S[:, 4:5]], axis=1))

    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    # sanity: physical bend scale on long steps
    long_m = np.abs(X[:, 6]) > 4000
    bend = float(np.median(np.abs(Y[long_m, 2] - X[long_m, 2]))) if long_m.any() else -1
    meta = dict(shard=shard, n=int(X.shape[0]), kappa=KAPPA, field="v8r1.down",
                polarity_convention="raw MagDown By<0; extrapUTT m_polarity=-1",
                seed=9000 + shard * 7919, median_long_step_dtx_rad=bend,
                wall_s=round(time.time() - t0, 1))
    np.savez_compressed(out_dir / f"shard_{shard:04d}.npz", X=X, Y=Y)
    (out_dir / f"shard_{shard:04d}.json").write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
