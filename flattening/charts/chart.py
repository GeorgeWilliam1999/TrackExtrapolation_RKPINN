#!/usr/bin/env python3
"""Canonical analytic flattening chart (F3.1 winner: O12 anisotropic + clamp).

build_chart() : fit even-multipole transverse profiles c_ab(z) of B_y over the
                FULL z-range, save chart_tables.npz.
chart_predict(): deployment-legal baseline prediction (x,y,tx,ty) for a batch,
                using ONLY the 1-D c_ab(z) tables + a fixed path quadrature.
                x,tx carry the dipole kick; y,ty are straight-line (the small
                B_x/B_z bend is left to the residual network).

Reaches ~12 um median / 371 um p95 on the UT->T pool with ZERO trained params
(reference true-field path integral: 5.7 um / 159 um).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent.parent / "gen_3"
sys.path.insert(0, str(GEN3 / "utils"))
from magnetic_field import get_field_numpy, C_LIGHT  # noqa: E402

# F3.1 winner config
ORDER = 12
XN_X, XN_Y = 4500.0, 2800.0      # bending plane (x) wider than y
SIGMA_W = 3000.0                 # F3.2 sweep: flatter weight halves the p99 tail
#                                  (371/1489 -> 267/884 p95/p99) at no median cost;
#                                  sigma_w=1000 starved the large-|x| paths.
NG = 61
ZSTEP = 25.0
ZMIN, ZMAX = -500.0, 14000.0     # full field-map z-range
NS = 80
CLAMP = True


def even_terms(order):
    return [(a, b) for a in range(0, order + 1, 2) for b in range(0, order + 1 - a, 2)]


def build_chart(out=None, kappa0=None):
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    axx = np.linspace(-XN_X, XN_X, NG); axy = np.linspace(-XN_Y, XN_Y, NG)
    XX, YY = np.meshgrid(axx, axy, indexing="ij"); xf, yf = XX.ravel(), YY.ravel()
    w = np.exp(-(xf**2 + yf**2) / (2 * SIGMA_W**2)); sw = np.sqrt(w)
    terms = even_terms(ORDER)
    A = np.stack([(xf/XN_X)**a * (yf/XN_Y)**b for (a, b) in terms], axis=1)
    Apinv = np.linalg.pinv(sw[:, None] * A)
    zpl = np.arange(ZMIN, ZMAX + ZSTEP, ZSTEP)
    C = np.empty((len(zpl), len(terms)))
    for i, z in enumerate(zpl):
        _, By, _ = field(xf, yf, np.full_like(xf, z))
        C[i] = Apinv @ (sw * np.asarray(By, np.float64))
    if kappa0 is None:
        kappa0 = float(np.load(HERE / "field_integrals.npz")["kappa0"])
    out = out or (HERE / "chart_tables.npz")
    np.savez(out, z=zpl, C=C, terms=np.array(terms), order=ORDER,
             xn_x=XN_X, xn_y=XN_Y, clamp=CLAMP, ns=NS, kappa0=kappa0,
             c_light=C_LIGHT, source="twodip.rtf")
    print(f"built chart: {len(terms)} terms, z[{zpl[0]:.0f},{zpl[-1]:.0f}]@{ZSTEP}mm, "
          f"{C.size*4/1024:.1f} kB -> {out}")
    return zpl, terms, C, kappa0


def load_chart(path=None):
    t = np.load(path or (HERE / "chart_tables.npz"), allow_pickle=True)
    return (t["z"], [tuple(x) for x in t["terms"]], t["C"], float(t["kappa0"]),
            float(t["xn_x"]), float(t["xn_y"]), bool(t["clamp"]), int(t["ns"]))


def chart_predict(X, chart=None):
    """X[N,7]=(x,y,tx,ty,qop,z0,dz) -> baseline (x,y,tx,ty)[N,4]. Deployment-legal."""
    if chart is None:
        chart = load_chart()
    zpl, terms, C, k0, xnx, xny, clamp, ns = chart
    X = np.asarray(X, np.float64)
    x0, y0, tx0, ty0, qop = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    z0s, dzs = X[:, 5], X[:, 6]; n = len(X); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0., 1., ns)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xp = x0[:, None] + tx0[:, None] * (zp - z0s[:, None])
    yp = y0[:, None] + ty0[:, None] * (zp - z0s[:, None])
    un, vn = xp / xnx, yp / xny
    if clamp:
        un = np.clip(un, -1., 1.); vn = np.clip(vn, -1., 1.)
    By = np.zeros((n, ns))
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, ns)
        By += cz * (un**a) * (vn**b)
    dzc = (dzs / (ns - 1))[:, None]
    I1 = np.sum(0.5 * (By[:, 1:] + By[:, :-1]) * dzc, axis=1)
    Ft = np.concatenate([np.zeros((n, 1)),
                         np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5 * (Ft[:, 1:] + Ft[:, :-1]) * dzc, axis=1)
    xb = x0 + tx0 * dzs - k * geo * I2
    txb = tx0 - k * geo * I1
    yb = y0 + ty0 * dzs       # straight-line; residual net learns the small B_x/B_z bend
    tyb = ty0
    return np.stack([xb, yb, txb, tyb], axis=1)


if __name__ == "__main__":
    build_chart()
    # quick self-check on the UT->T pool
    d = np.load(GEN3 / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0 + dz
    m = (z0 >= 2300) & (z0 <= 3000) & (zf >= 7600) & (zf <= 9500) & (dz > 0)
    pred = chart_predict(X[m])
    a = np.abs(pred[:, 0] - Y[m, 0]) * 1e3
    print(f"self-check UT->T: median |dx| = {np.median(a):.1f} um, p95 = {np.quantile(a,.95):.0f} um")
