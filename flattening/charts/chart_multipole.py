#!/usr/bin/env python3
"""F3b/c — Transverse multipole chart: reproduce the path-integrated kick
with NO 3-D field lookup at inference.

At each z-plane fit  B_y(x,y,z) ~ sum_{a+b<=ORDER} c_ab(z) x^a y^b  over the
track envelope. The c_ab(z) are 1-D tables (the transverse multipole profiles).
The chart kick is then a FIXED quadrature along the straight path with B_y
reconstructed from the c_ab tables only (deployment-legal):

    By_mp(x,y,z) = sum_ab c_ab(z) x^a y^b
    dtx = -k * geo * INT By_mp(x0+tx s, y0+ty s, z) ds      (geo=N(1+tx^2))
    dx  = -k * geo * INT (INT By_mp) ds  + x0 + tx0 dz

Compares rung-2 (multipole, orders 1..4) to rung-1.5 (true 3-D path integral)
and to RK truth on the UT->T pool.
"""
from __future__ import annotations
import sys, json, itertools
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
GEN3 = FLAT.parent / "gen_3"
sys.path.insert(0, str(GEN3 / "utils"))
from magnetic_field import get_field_numpy  # noqa: E402

UT_Z, T_Z = (2300.0, 3000.0), (7600.0, 9500.0)
XYMAX = 2600.0          # fit window (mm) — covers the UT->T track envelope
XN = 2600.0             # coordinate normalisation: basis uses (x/XN, y/XN) in [-1,1] -> well-conditioned
NG = 53                 # xy fit grid per axis
ZSTEP = 25.0            # z resolution of the multipole tables (mm)
ZMIN, ZMAX = 2000.0, 9800.0   # only need the UT->T z span (+margin)
NS = 80                 # path-quadrature samples
SIGMA_W = 750.0         # Gaussian fit weight (mm) — prioritise the populated near-axis region


def terms_upto(order):
    # even-even only: B_y is even in x and in y for a midplane-symmetric dipole
    return [(a, b) for a in range(0, order + 1, 2) for b in range(0, order + 1 - a, 2)]


def build_multipoles(order):
    field = get_field_numpy(use_interpolated=True, polarity=-1)
    ax = np.linspace(-XYMAX, XYMAX, NG)
    XX, YY = np.meshgrid(ax, ax, indexing="ij")
    xf, yf = XX.ravel(), YY.ravel()
    w = np.exp(-(xf**2 + yf**2) / (2 * SIGMA_W**2))
    sw = np.sqrt(w)
    terms = terms_upto(order)
    un, vn = xf / XN, yf / XN                                   # normalised coords in [-1,1]
    A = np.stack([un**a * vn**b for (a, b) in terms], axis=1)   # [n_xy, n_terms]
    Apinv_w = np.linalg.pinv(sw[:, None] * A)                   # weighted pseudo-inverse
    zpl = np.arange(ZMIN, ZMAX + ZSTEP, ZSTEP)
    C = np.empty((len(zpl), len(terms)))
    fit_res = []
    for i, z in enumerate(zpl):
        _, By, _ = field(xf, yf, np.full_like(xf, z))
        By = np.asarray(By, np.float64)
        c = Apinv_w @ (sw * By)
        C[i] = c
        # weighted RMS (the metric that matters for the integral)
        fit_res.append(np.sqrt(np.sum(w * (A @ c - By) ** 2) / np.sum(w)))
    return zpl, terms, C, float(np.median(fit_res)), float(np.max(fit_res))


def rung2(Xs, Ys, zpl, terms, C, k0, NS=NS):
    x0, y0, tx0, ty0, qop = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    z0s, dzs = Xs[:, 5], Xs[:, 6]
    n = len(Xs); k = k0 * qop
    geo = np.sqrt(1 + tx0**2 + ty0**2) * (1 + tx0**2)
    s = np.linspace(0., 1., NS)
    zp = z0s[:, None] + s[None, :] * dzs[:, None]          # [n,NS]
    xp = x0[:, None] + tx0[:, None] * (zp - z0s[:, None])
    yp = y0[:, None] + ty0[:, None] * (zp - z0s[:, None])
    # reconstruct By from multipole tables (interp c_ab at each sample z)
    By = np.zeros((n, NS))
    un, vn = xp / XN, yp / XN
    for ti, (a, b) in enumerate(terms):
        cz = np.interp(zp.ravel(), zpl, C[:, ti]).reshape(n, NS)
        By += cz * (un**a) * (vn**b)
    dzc = (dzs / (NS - 1))[:, None]
    I1 = np.sum(0.5 * (By[:, 1:] + By[:, :-1]) * dzc, axis=1)
    Ftrk = np.concatenate([np.zeros((n, 1)),
                           np.cumsum(0.5 * (By[:, 1:] + By[:, :-1]) * dzc, axis=1)], axis=1)
    I2 = np.sum(0.5 * (Ftrk[:, 1:] + Ftrk[:, :-1]) * dzc, axis=1)
    dtx = tx0 - k * geo * I1 - Ys[:, 2]
    dx = x0 + tx0 * dzs - k * geo * I2 - Ys[:, 0]
    return dx, dtx, qop


def rep(dx_mm, dtx, qop, label, store):
    a = np.abs(dx_mm) * 1e3
    q = np.abs(qop); e = np.quantile(q, [.25, .5, .75]); b = np.digitize(q, e)
    byq = [float(np.median(a[b == i])) for i in range(4)]
    store[label] = {"median_dx_um": float(np.median(a)), "p95_dx_um": float(np.quantile(a, .95)),
                    "median_dtx_urad": float(np.median(np.abs(dtx)) * 1e6), "byq_um": byq}
    print(f"{label:<34} med={np.median(a):8.1f}um p95={np.quantile(a,.95):8.0f} "
          f"|dtx|={np.median(np.abs(dtx))*1e6:7.1f}urad byQ={[round(v,1) for v in byq]}")


def main():
    t = np.load(FLAT / "charts" / "field_integrals.npz"); k0 = float(t["kappa0"])
    d = np.load(GEN3 / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0 + dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    print(f"UT->T pool {len(Xs):,}\n")
    store = {}
    for order in (2, 4, 6, 8, 10):
        zpl, terms, C, med_res, max_res = build_multipoles(order)
        dx, dtx, qop = rung2(Xs, Ys, zpl, terms, C, k0)
        lab = f"rung2 multipole O{order} ({len(terms)} terms)"
        rep(dx, dtx, qop, lab, store)
        store[lab]["fit_rms_med_T"] = med_res
        store[lab]["fit_rms_max_T"] = max_res
        store[lab]["n_terms"] = len(terms)
        print(f"    (per-plane fit RMS over +-{XYMAX:.0f}mm: median {med_res*1e3:.2f} mT, max {max_res*1e3:.1f} mT)")
        # persist the best-order tables for deployment use
        if order == 8:
            np.savez(HERE / "multipole_tables.npz", z=zpl, C=C,
                     terms=np.array(terms), order=order, kappa0=k0,
                     xymax=XYMAX, source="twodip.rtf")
    out = FLAT / "results" / f"F3_multipole_{datetime.now().strftime('%Y-%m-%d')}.json"
    json.dump(store, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}  (+ charts/multipole_tables.npz, order 3)")


if __name__ == "__main__":
    main()
