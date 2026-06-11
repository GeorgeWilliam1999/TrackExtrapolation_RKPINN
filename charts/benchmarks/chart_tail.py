#!/usr/bin/env python3
"""F3.1 — Diagnose and fix the multipole-chart tail (p95 ~ 2000 um).

Hypothesis: a population of large-bend (low-p) tracks fan beyond the +-XN fit
window, where the polynomial extrapolates and blows up. Diagnose the envelope,
then sweep the fix: wider window, anisotropic window (bending plane = x), and
coordinate clamping.
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
ZMIN, ZMAX, ZSTEP = 2000.0, 9800.0, 25.0
NS = 80


def even_terms(order):
    return [(a, b) for a in range(0, order + 1, 2) for b in range(0, order + 1 - a, 2)]


def build_mp(order, xmax, ymax, xn, yn, sigma_w, ng=53):
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    axx = np.linspace(-xmax, xmax, ng)
    axy = np.linspace(-ymax, ymax, ng)
    XX, YY = np.meshgrid(axx, axy, indexing="ij")
    xf, yf = XX.ravel(), YY.ravel()
    w = np.exp(-(xf**2 + yf**2) / (2 * sigma_w**2)); sw = np.sqrt(w)
    terms = even_terms(order)
    A = np.stack([(xf/xn)**a * (yf/yn)**b for (a, b) in terms], axis=1)
    Apinv = np.linalg.pinv(sw[:, None] * A)
    zpl = np.arange(ZMIN, ZMAX + ZSTEP, ZSTEP)
    C = np.empty((len(zpl), len(terms)))
    for i, z in enumerate(zpl):
        _, By, _ = field(xf, yf, np.full_like(xf, z))
        C[i] = Apinv @ (sw * np.asarray(By, np.float64))
    return zpl, terms, C


def rung2(Xs, Ys, zpl, terms, C, k0, xn, yn, clamp=False):
    x0, y0, tx0, ty0, qop = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    z0s, dzs = Xs[:, 5], Xs[:, 6]; n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xp = x0[:, None] + tx0[:, None] * (zp - z0s[:, None])
    yp = y0[:, None] + ty0[:, None] * (zp - z0s[:, None])
    un, vn = xp / xn, yp / yn
    if clamp:
        un = np.clip(un, -1.0, 1.0); vn = np.clip(vn, -1.0, 1.0)
    By = np.zeros((n, NS))
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, NS)
        By += cz * (un**a) * (vn**b)
    dzc = (dzs / (NS - 1))[:, None]
    I1 = np.sum(0.5 * (By[:, 1:] + By[:, :-1]) * dzc, axis=1)
    Ft = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5 * (Ft[:, 1:] + Ft[:, :-1]) * dzc, axis=1)
    return x0 + tx0*dzs - k*geo*I2 - Ys[:, 0], tx0 - k*geo*I1 - Ys[:, 2], qop


def rep(dx, dtx, qop, label, store):
    a = np.abs(dx) * 1e3
    q = np.abs(qop); e = np.quantile(q, [.25, .5, .75]); b = np.digitize(q, e)
    byq = [round(float(np.median(a[b == i])), 1) for i in range(4)]
    store[label] = {"median_um": float(np.median(a)), "p95_um": float(np.quantile(a, .95)),
                    "p99_um": float(np.quantile(a, .99)), "byq": byq}
    print(f"{label:<42} med={np.median(a):7.1f} p95={np.quantile(a,.95):8.0f} "
          f"p99={np.quantile(a,.99):8.0f} byQ={byq}")


def main():
    t = np.load(FLAT / "charts" / "field_integrals.npz"); k0 = float(t["kappa0"])
    d = np.load(LAB / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0 + dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    n = len(Xs)

    # --- envelope diagnostic ---
    x0, y0, tx0, ty0 = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3]
    z0s, dzs = Xs[:, 5], Xs[:, 6]
    s = np.linspace(0, 1, NS)
    xp = x0[:, None] + tx0[:, None] * (s[None, :] * dzs[:, None])
    yp = y0[:, None] + ty0[:, None] * (s[None, :] * dzs[:, None])
    maxx = np.abs(xp).max(1); maxy = np.abs(yp).max(1)
    print(f"UT->T pool {n:,}")
    print(f"path |x| max: median {np.median(maxx):.0f}  p95 {np.quantile(maxx,.95):.0f}  "
          f"p99 {np.quantile(maxx,.99):.0f}  abs-max {maxx.max():.0f} mm")
    print(f"path |y| max: median {np.median(maxy):.0f}  p95 {np.quantile(maxy,.95):.0f}  "
          f"p99 {np.quantile(maxy,.99):.0f}  abs-max {maxy.max():.0f} mm")
    print(f"frac tracks with |x|>2600: {(maxx>2600).mean()*100:.1f}%   |y|>2600: {(maxy>2600).mean()*100:.1f}%\n")

    store = {}
    # baseline (current O8)
    zpl, terms, C = build_mp(8, 2600, 2600, 2600, 2600, 750)
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 2600, 2600), "O8 baseline (XN=2600)", store)
    # clamp
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 2600, 2600, clamp=True), "O8 + clamp", store)
    # wider isotropic
    zpl, terms, C = build_mp(8, 3600, 3600, 3600, 3600, 900)
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 3600, 3600), "O8 wide (XN=3600)", store)
    # anisotropic: wide in x (bending plane), tighter in y, higher order
    zpl, terms, C = build_mp(10, 3800, 2400, 3800, 2400, 900)
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 3800, 2400), "O10 aniso (x=3800,y=2400)", store)
    # anisotropic + clamp
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 3800, 2400, clamp=True), "O10 aniso + clamp", store)
    # push order + window in the bending plane
    zpl, terms, C = build_mp(12, 4500, 2800, 4500, 2800, 1000, ng=61)
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 4500, 2800, clamp=True), "O12 aniso+clamp (x=4500,y=2800)", store)
    zpl, terms, C = build_mp(14, 5000, 3200, 5000, 3200, 1100, ng=71)
    rep(*rung2(Xs, Ys, zpl, terms, C, k0, 5000, 3200, clamp=True), "O14 aniso+clamp (x=5000,y=3200)", store)
    # reference: rung1.5 true-field path integral
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    x0, y0, tx0, ty0, qop = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    z0s, dzs = Xs[:, 5], Xs[:, 6]; n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0, 1, NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xpp = x0[:, None] + tx0[:, None]*(zp - z0s[:, None]); ypp = y0[:, None] + ty0[:, None]*(zp - z0s[:, None])
    _, By, _ = field(xpp.ravel(), ypp.ravel(), zp.ravel()); By = np.asarray(By, np.float64).reshape(n, NS)
    dzc = (dzs/(NS-1))[:, None]; I1 = np.sum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)
    Ft = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5*(Ft[:, 1:]+Ft[:, :-1])*dzc, axis=1)
    rep(x0+tx0*dzs-k*geo*I2-Ys[:, 0], tx0-k*geo*I1-Ys[:, 2], qop, "REF rung1.5 (true 3-D field)", store)

    out = FLAT / "results" / f"F3p1_tail_{datetime.now().strftime('%Y-%m-%d')}.json"
    json.dump(store, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
