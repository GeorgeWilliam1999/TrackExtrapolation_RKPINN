#!/usr/bin/env python3
"""F3a — Decompose the rung-1.5 win: geometry vs transverse field.

rung 0    straight line
rung 1.0  on-axis kick, NO geometric factor:  dtx = -k*[F1-F0]
rung 1.25 on-axis kick WITH geo = N*(1+tx^2): DEPLOYMENT-LEGAL (1-D F,G tables only)
rung 1.5  path-integrated kick WITH geo:      needs the 3-D map (reference)

If rung 1.25 ~ rung 1.5, the win is geometry (free, no multipoles needed).
If rung 1.25 ~ rung 1.0, the win is the transverse field (F3 multipoles essential).

The slope EOM leading term is  dtx/dz = -k * N * (1+tx^2) * By(path),
so 'geo' = N*(1+tx0^2) frozen at the initial slope.
"""
from __future__ import annotations
import sys, json
from datetime import datetime
from pathlib import Path
import os

import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
REPO = FLAT.parent
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
sys.path.insert(0, str(REPO / "core"))
from magnetic_field import get_field_numpy  # noqa: E402

UT_Z, T_Z = (2300.0, 3000.0), (7600.0, 9500.0)


def rep(dx_mm, dtx, qop, label, store):
    a = np.abs(dx_mm) * 1e3
    q = np.abs(qop)
    e = np.quantile(q, [.25, .5, .75]); b = np.digitize(q, e)
    byq = [float(np.median(a[b == i])) for i in range(4)]
    store[label] = {"median_dx_um": float(np.median(a)), "p95_dx_um": float(np.quantile(a, .95)),
                    "median_dtx_urad": float(np.median(np.abs(dtx)) * 1e6),
                    "byq_um": byq}
    print(f"{label:<30} med={np.median(a):9.1f}um p95={np.quantile(a,.95):9.0f} "
          f"|dtx|med={np.median(np.abs(dtx))*1e6:8.1f}urad byQ={[round(v,1) for v in byq]}")


def main():
    t = np.load(FLAT / "charts" / "field_integrals.npz")
    zg, F, G, k0 = t["z_grid"], t["F"], t["G"], float(t["kappa0"])
    d = np.load(LAB / "data" / "train_10M_gen3.npz")
    X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0 + dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    x0, y0, tx0, ty0, qop = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    z0s, dzs = Xs[:, 5], Xs[:, 6]; z1s = z0s + dzs
    n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    print(f"UT->T pool {n:,}   kappa0={k0:.4e}   <geo>={geo.mean():.4f}\n")

    F0, F1 = np.interp(z0s, zg, F), np.interp(z1s, zg, F)
    G0, G1 = np.interp(z0s, zg, G), np.interp(z1s, zg, G)
    store = {}

    rep(x0 + tx0*dzs - Ys[:, 0], tx0 - Ys[:, 2], qop, "rung0 straight", store)
    rep(x0 + tx0*dzs - k*(G1-G0-F0*dzs) - Ys[:, 0],
        tx0 - k*(F1-F0) - Ys[:, 2], qop, "rung1.0 on-axis (no geo)", store)
    rep(x0 + tx0*dzs - k*geo*(G1-G0-F0*dzs) - Ys[:, 0],
        tx0 - k*geo*(F1-F0) - Ys[:, 2], qop, "rung1.25 on-axis x geo [LEGAL]", store)

    # rung 1.5 — path-integrated By (3-D map), reference
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    NS = 120
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :]*dzs[:, None]
    xp = x0[:, None] + tx0[:, None]*(zp - z0s[:, None])
    yp = y0[:, None] + ty0[:, None]*(zp - z0s[:, None])
    _, By, _ = field(xp.ravel(), yp.ravel(), zp.ravel())
    By = np.asarray(By, np.float64).reshape(n, NS)
    dzc = (dzs/(NS-1))[:, None]
    I1 = np.sum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)
    Ftrk = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5*(Ftrk[:, 1:]+Ftrk[:, :-1])*dzc, axis=1)
    rep(x0 + tx0*dzs - k*geo*I2 - Ys[:, 0], tx0 - k*geo*I1 - Ys[:, 2], qop,
        "rung1.5 path-integrated x geo", store)

    out = FLAT / "results" / f"F3a_decompose_{datetime.now().strftime('%Y-%m-%d')}.json"
    json.dump(store, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
