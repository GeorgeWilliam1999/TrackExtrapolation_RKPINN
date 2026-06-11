#!/usr/bin/env python3
"""F3.3 — rung-2 (Magnus-2 / one Picard) path iteration.

Rung-1.5 integrates B_y along the STRAIGHT chord and leaves the O(kappa^2)
straight-chord floor (5.7 um median with the true field). Rung-2 runs one
iteration: use the rung-1.5 bent trajectory x(z) and re-integrate B_y along it.

We test both field sources:
  * true 3-D field  -> establishes the new floor below 5.7 um,
  * even-multipole chart tables (deployment-legal) -> the realisable number.
Reference: chart rung-1.5 = 11.9/371, true rung-1.5 = 5.7/159.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
GEN3 = FLAT.parent / "gen_3"
sys.path.insert(0, str(FLAT / "charts"))
sys.path.insert(0, str(GEN3 / "utils"))
from chart import load_chart  # noqa: E402
from magnetic_field import get_field_numpy  # noqa: E402

UT_Z, T_Z = (2300.0, 3000.0), (7600.0, 9500.0)
NS = 80


def cumint(f, dzc):
    return np.concatenate([np.zeros((f.shape[0], 1)),
                           np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dzc, axis=1)], axis=1)


def by_multipole(xp, yp, zp, chart):
    zpl, terms, C, k0, xnx, xny, clamp, _ = chart
    n, ns = xp.shape
    un, vn = xp / xnx, yp / xny
    if clamp:
        un = np.clip(un, -1., 1.); vn = np.clip(vn, -1., 1.)
    By = np.zeros((n, ns))
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, ns); By += cz*(un**a)*(vn**b)
    return By


def metrics(Xs, Ys, By, k0, x0, tx0, ty0, dzs, dzc):
    n = len(Xs); k = k0 * Xs[:, 4]
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    I2 = np.sum(0.5 * (cumint(By, dzc)[:, 1:] + cumint(By, dzc)[:, :-1]) * dzc, axis=1)
    a = np.abs(x0 + tx0*dzs - k*geo*I2 - Ys[:, 0]) * 1e3
    return np.median(a), np.quantile(a, .95), np.quantile(a, .99)


def run(Xs, Ys, chart, use_true, field, k0):
    x0, y0, tx0, ty0 = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3]
    z0s, dzs = Xs[:, 5], Xs[:, 6]; n = len(Xs); k = k0 * Xs[:, 4]
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    dzc = (dzs/(NS-1))[:, None]
    # rung-1.5: straight chord
    xp = x0[:, None] + tx0[:, None]*(zp-z0s[:, None]); yp = y0[:, None] + ty0[:, None]*(zp-z0s[:, None])

    def evalf(xc, yc):
        if use_true:
            _, By, _ = field(xc.ravel(), yc.ravel(), zp.ravel()); return np.asarray(By, np.float64).reshape(n, NS)
        return by_multipole(xc, yc, zp, chart)

    By0 = evalf(xp, yp)
    r15 = metrics(Xs, Ys, By0, k0, x0, tx0, ty0, dzs, dzc)
    # bent trajectory from rung-1.5 running kick
    Ft = cumint(By0, dzc)                                   # running int By
    I2c = cumint(Ft, dzc)                                   # running double int -> x correction
    x_bent = x0[:, None] + tx0[:, None]*(zp-z0s[:, None]) - (k*geo)[:, None]*I2c
    By1 = evalf(x_bent, yp)
    r2 = metrics(Xs, Ys, By1, k0, x0, tx0, ty0, dzs, dzc)
    return r15, r2


def main():
    chart = load_chart(); k0 = chart[3]
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    d = np.load(GEN3 / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0+dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    print(f"UT->T pool {len(Xs):,}\n")
    for use_true, lbl in ((True, "TRUE 3-D field"), (False, "even-multipole chart (deployment)")):
        r15, r2 = run(Xs, Ys, chart, use_true, field, k0)
        print(f"  {lbl}")
        print(f"    rung-1.5 (straight): median {r15[0]:6.1f}  p95 {r15[1]:7.0f}  p99 {r15[2]:7.0f} um")
        print(f"    rung-2   (bent):     median {r2[0]:6.1f}  p95 {r2[1]:7.0f}  p99 {r2[2]:7.0f} um\n")


if __name__ == "__main__":
    main()
