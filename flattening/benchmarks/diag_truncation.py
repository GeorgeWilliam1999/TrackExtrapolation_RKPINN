#!/usr/bin/env python3
"""F3.2 diagnostic — WHERE does the chart's multipole-truncation error live?

The chart median on UT->T is 11.9 um; the true-field straight-chord reference is
5.7 um. The ~10.4 um quadrature difference is the even-multipole reconstruction
error of B_y integrated along the path. Before choosing a fix (Maxwell y-extension
vs better x-basis vs wider window), decompose that error:

  * true B_y                              -> the 5.7 um floor (chart machinery, exact field)
  * even-multipole B_y, CLAMPED  (=chart) -> 11.9 um
  * even-multipole B_y, UNCLAMPED         -> isolates the clamp (large-|x| tail) contribution

and report the error's dependence on |x|_max along the path. This tells us whether
the lever is the y-extension (Maxwell) or the x-representation/window (bending plane).
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


def by_multipole(xp, yp, zp, chart, clamp):
    zpl, terms, C, k0, xnx, xny, _, _ = chart
    n, ns = xp.shape
    un, vn = xp / xnx, yp / xny
    if clamp:
        un = np.clip(un, -1., 1.); vn = np.clip(vn, -1., 1.)
    By = np.zeros((n, ns))
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, ns)
        By += cz * (un**a) * (vn**b)
    return By


def kick_dx(By, Xs, k0):
    x0, tx0, ty0, qop = Xs[:, 0], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    dzs = Xs[:, 6]; n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    dzc = (dzs / (NS - 1))[:, None]
    I1 = np.sum(0.5 * (By[:, 1:] + By[:, :-1]) * dzc, axis=1)
    Ft = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5 * (Ft[:, 1:] + Ft[:, :-1]) * dzc, axis=1)
    return x0 + tx0*dzs - k*geo*I2, tx0 - k*geo*I1


def med(a):
    return float(np.median(a)), float(np.quantile(a, .95)), float(np.quantile(a, .99))


def main():
    chart = load_chart(); k0 = chart[3]
    d = np.load(GEN3 / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0 + dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64); n = len(Xs)
    print(f"UT->T pool {n:,}")

    x0, y0, tx0, ty0 = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3]
    z0s, dzs = Xs[:, 5], Xs[:, 6]
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xp = x0[:, None] + tx0[:, None] * (zp - z0s[:, None])
    yp = y0[:, None] + ty0[:, None] * (zp - z0s[:, None])
    maxx = np.abs(xp).max(1)

    field = get_field_numpy(use_interpolated=True, polarity=-1)
    _, Byt, _ = field(xp.ravel(), yp.ravel(), zp.ravel())
    Byt = np.asarray(Byt, np.float64).reshape(n, NS)

    By_cl = by_multipole(xp, yp, zp, chart, clamp=True)
    By_nc = by_multipole(xp, yp, zp, chart, clamp=False)

    for name, By in (("true field (floor)", Byt),
                     ("multipole CLAMPED (=chart)", By_cl),
                     ("multipole UNCLAMPED", By_nc)):
        xb, txb = kick_dx(By, Xs, k0)
        a = np.abs(xb - Ys[:, 0]) * 1e3
        print(f"  {name:<30} median |dx| = {med(a)[0]:6.1f}  p95 {med(a)[1]:7.0f}  p99 {med(a)[2]:7.0f} um")

    # where does the field-reconstruction error concentrate?
    print("\nfield-reconstruction error |By_chart - By_true| along paths (clamped):")
    err = np.abs(By_cl - Byt)
    print(f"  per-point median {np.median(err)*1e3:.2f} mT, p95 {np.quantile(err,.95)*1e3:.1f} mT")
    inwin = (np.abs(xp) <= chart[4]) & (np.abs(yp) <= chart[5])
    print(f"  path points inside x/y window: {inwin.mean()*100:.1f}%")
    print(f"  median err INSIDE window:  {np.median(err[inwin])*1e3:.2f} mT")
    if (~inwin).any():
        print(f"  median err OUTSIDE window: {np.median(err[~inwin])*1e3:.2f} mT  (clamped region)")

    # split chart-error by whether the track ever leaves the window
    leaves = maxx > chart[4]
    print(f"\ntracks that fan beyond x-window (|x|>{chart[4]:.0f}): {leaves.mean()*100:.1f}%")
    xb, _ = kick_dx(By_cl, Xs, k0); a = np.abs(xb - Ys[:, 0]) * 1e3
    print(f"  chart median |dx|  inside-window tracks: {np.median(a[~leaves]):6.1f} um")
    print(f"  chart median |dx|  beyond-window tracks: {np.median(a[leaves]):6.1f} um")
    xbt, _ = kick_dx(Byt, Xs, k0); at = np.abs(xbt - Ys[:, 0]) * 1e3
    print(f"  true  median |dx|  inside-window tracks: {np.median(at[~leaves]):6.1f} um")
    print(f"  true  median |dx|  beyond-window tracks: {np.median(at[leaves]):6.1f} um")


if __name__ == "__main__":
    main()
