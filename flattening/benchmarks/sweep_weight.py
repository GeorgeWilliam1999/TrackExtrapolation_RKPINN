#!/usr/bin/env python3
"""F3.2 — does the chart median come from the center-weighted fit starving large-|x|?

The even-multipole tables are fit with a Gaussian weight w=exp(-(x^2+y^2)/2 sigma_w^2).
At sigma_w=1000, a path point at x=4500 has weight exp(-10)~4e-5 -> the fit ignores
exactly the bending-plane region the UT->T paths traverse. Sweep sigma_w (and a flat
weight) at O12, measure UT->T median |dx|. Reference: chart 11.9 um, true field 5.7 um.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
GEN3 = FLAT.parent / "gen_3"
sys.path.insert(0, str(GEN3 / "utils"))
from magnetic_field import get_field_numpy  # noqa: E402

UT_Z, T_Z = (2300.0, 3000.0), (7600.0, 9500.0)
ZMIN, ZMAX, ZSTEP = 2000.0, 9800.0, 25.0
NS = 80


def even_terms(order):
    return [(a, b) for a in range(0, order + 1, 2) for b in range(0, order + 1 - a, 2)]


def build(order, xn, yn, sigma_w, ng=61, flat=False):
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    axx = np.linspace(-xn, xn, ng); axy = np.linspace(-yn, yn, ng)
    XX, YY = np.meshgrid(axx, axy, indexing="ij"); xf, yf = XX.ravel(), YY.ravel()
    w = np.ones_like(xf) if flat else np.exp(-(xf**2 + yf**2) / (2 * sigma_w**2))
    sw = np.sqrt(w)
    terms = even_terms(order)
    A = np.stack([(xf/xn)**a * (yf/yn)**b for (a, b) in terms], axis=1)
    Apinv = np.linalg.pinv(sw[:, None] * A)
    zpl = np.arange(ZMIN, ZMAX + ZSTEP, ZSTEP)
    C = np.empty((len(zpl), len(terms)))
    for i, z in enumerate(zpl):
        _, By, _ = field(xf, yf, np.full_like(xf, z))
        C[i] = Apinv @ (sw * np.asarray(By, np.float64))
    return zpl, terms, C


def evalchart(Xs, Ys, zpl, terms, C, k0, xn, yn, clamp=True):
    x0, y0, tx0, ty0, qop = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    z0s, dzs = Xs[:, 5], Xs[:, 6]; n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xp = x0[:, None] + tx0[:, None]*(zp-z0s[:, None]); yp = y0[:, None] + ty0[:, None]*(zp-z0s[:, None])
    un, vn = xp/xn, yp/yn
    if clamp:
        un = np.clip(un, -1., 1.); vn = np.clip(vn, -1., 1.)
    By = np.zeros((n, NS))
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, NS); By += cz*(un**a)*(vn**b)
    dzc = (dzs/(NS-1))[:, None]
    Ft = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5*(Ft[:, 1:]+Ft[:, :-1])*dzc, axis=1)
    a = np.abs(x0+tx0*dzs-k*geo*I2 - Ys[:, 0])*1e3
    return np.median(a), np.quantile(a, .95), np.quantile(a, .99)


def main():
    k0 = float(np.load(FLAT / "charts" / "field_integrals.npz")["kappa0"])
    d = np.load(GEN3 / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0+dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    print(f"UT->T pool {len(Xs):,}   (chart=11.9/371, true=5.7/159)\n")
    XN, YN = 4500., 2800.
    for sw, flat, lbl in ((1000, False, "sigma_w=1000 (current)"),
                          (2000, False, "sigma_w=2000"),
                          (3000, False, "sigma_w=3000"),
                          (4500, False, "sigma_w=4500"),
                          (0, True, "flat (uniform weight)")):
        zpl, terms, C = build(12, XN, YN, sw, ng=61, flat=flat)
        med, p95, p99 = evalchart(Xs, Ys, zpl, terms, C, k0, XN, YN, clamp=True)
        print(f"  O12 {lbl:<26} median {med:6.1f}  p95 {p95:7.0f}  p99 {p99:7.0f} um")


if __name__ == "__main__":
    main()
