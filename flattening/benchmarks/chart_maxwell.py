#!/usr/bin/env python3
"""F3.2 — Maxwell-consistent transverse expansion vs the even-multipole fit.

Build g(x,z)=B_y(x,0,z) on a uniform (x,z) grid from the true field, then extend
to y analytically via the harmonic (generalized-gradient) series
    B_y(x,y,z) = sum_n (-1)^n y^{2n}/(2n)! (d_x^2 + d_z^2)^n g(x,z),
computing (d_x^2+d_z^2)^n g by finite differences. Measure UT->T median |dx| for
n=0,1,2 and the per-point field error, against the even-multipole chart (11.9/371)
and the true-field floor (5.7/159). Decides whether the median floor is the
y-extension (Maxwell helps) or the x-representation (it does not).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
GEN3 = FLAT.parent / "gen_3"
sys.path.insert(0, str(GEN3 / "utils"))
from magnetic_field import get_field_numpy  # noqa: E402

UT_Z, T_Z = (2300.0, 3000.0), (7600.0, 9500.0)
NS = 80
XN = 5200.0
DXG, DZG = 40.0, 25.0
ZMIN, ZMAX = 1900.0, 9900.0


def laplacian_xz(g, dx, dz):
    L = np.zeros_like(g)
    L[1:-1, :] += (g[2:, :] - 2*g[1:-1, :] + g[:-2, :]) / dx**2
    L[:, 1:-1] += (g[:, 2:] - 2*g[:, 1:-1] + g[:, :-2]) / dz**2
    L[0, :] = L[1, :]; L[-1, :] = L[-2, :]; L[:, 0] = L[:, 1]; L[:, -1] = L[:, -2]
    return L


def kick_med(Xs, Ys, By, k0):
    x0, tx0, ty0, qop = Xs[:, 0], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    dzs = Xs[:, 6]; n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    dzc = (dzs/(NS-1))[:, None]
    Ft = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(By[:, 1:]+By[:, :-1])*dzc, axis=1)], axis=1)
    I2 = np.sum(0.5*(Ft[:, 1:]+Ft[:, :-1])*dzc, axis=1)
    a = np.abs(x0 + tx0*dzs - k*geo*I2 - Ys[:, 0]) * 1e3
    return np.median(a), np.quantile(a, .95), np.quantile(a, .99)


def main():
    k0 = float(np.load(FLAT / "charts" / "field_integrals.npz")["kappa0"])
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    d = np.load(GEN3 / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0+dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64); n = len(Xs)
    print(f"UT->T pool {n:,}   (even-multipole chart=11.9/371, true=5.7/159)\n")

    # build midplane trace g(x,z) and its laplacian powers
    xg = np.arange(-XN, XN + DXG, DXG); zg = np.arange(ZMIN, ZMAX + DZG, DZG)
    XXg, ZZg = np.meshgrid(xg, zg, indexing="ij")
    _, Byg, _ = field(XXg.ravel(), np.zeros(XXg.size), ZZg.ravel())
    g = np.asarray(Byg, np.float64).reshape(XXg.shape)
    Ln = [g]
    for _ in range(2):
        Ln.append(laplacian_xz(Ln[-1], DXG, DZG))
    interps = [RegularGridInterpolator((xg, zg), Lk, bounds_error=False, fill_value=None) for Lk in Ln]

    # path points
    x0, y0, tx0, ty0 = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3]
    z0s, dzs = Xs[:, 5], Xs[:, 6]
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]
    xp = x0[:, None] + tx0[:, None]*(zp-z0s[:, None]); yp = y0[:, None] + ty0[:, None]*(zp-z0s[:, None])
    pts = np.stack([np.clip(xp.ravel(), -XN, XN), np.clip(zp.ravel(), ZMIN, ZMAX)], axis=1)
    L_at = [ip(pts).reshape(n, NS) for ip in interps]
    yy = yp

    _, Byt, _ = field(xp.ravel(), yp.ravel(), zp.ravel()); Byt = np.asarray(Byt, np.float64).reshape(n, NS)

    By = np.zeros((n, NS))
    for nmax in (0, 1, 2):
        By = By + ((-1)**nmax) * (yy**(2*nmax)) / math.factorial(2*nmax) * L_at[nmax]
        med, p95, p99 = kick_med(Xs, Ys, By, k0)
        err = np.abs(By - Byt)
        print(f"  Maxwell n<= {nmax}   median {med:6.1f}  p95 {p95:7.0f}  p99 {p99:7.0f} um"
              f"   | field err/pt median {np.median(err)*1e3:.2f} mT p95 {np.quantile(err,.95)*1e3:.0f} mT")
    # reference: true field through the same machinery
    med, p95, p99 = kick_med(Xs, Ys, Byt, k0)
    print(f"  true field (floor)  median {med:6.1f}  p95 {p95:7.0f}  p99 {p99:7.0f} um")


if __name__ == "__main__":
    main()
