#!/usr/bin/env python3
"""P0.0 — external validation of the kappa x1000 fix.

Generates PV-pointing tracks (the population extrapUTT was fitted for),
propagates them with the FIXED RK (kappa = 1e-3 * qop_allen, imported from
utils/rk4_propagator so this script always reflects the live constant):

    PV (z ~ 0)  --RK-->  plane z=2665  --RK-->  truth z=7826

then writes the plane states for the extrapUTT driver. After running the
driver, compare_kappa.py-style metrics are printed by this same script when
the polynomial predictions exist.

PASS criterion: median |dx_poly - dx_RK| at the polynomial's fit-residual
scale (<< the physical bend, which is O(100 mm) at low p). That validates
kappa, the field map, and the qop conventions in one shot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent
sys.path.insert(0, str(GEN3 / "utils"))

from magnetic_field import get_field_numpy  # noqa: E402
from rk4_propagator import _ALLEN_KAPPA_PREFACTOR as KAPPA  # noqa: E402  (live constant)

ZINI, ZFIN = 2665.0, 7826.0
N = 4000
SEED = 20260611
STEP = 5.0

print(f"live _ALLEN_KAPPA_PREFACTOR = {KAPPA:g}  (expect 1e-3 post-fix)")
field = get_field_numpy(use_interpolated=True, polarity=-1)


def deriv(S, z):
    x, y, tx, ty, qop = S.T
    Bx, By, Bz = field(x, y, np.full_like(x, z))
    kap = KAPPA * qop
    Nf = np.sqrt(1 + tx * tx + ty * ty)
    dtx = kap * Nf * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz)
    dty = kap * Nf * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz)
    return np.stack([tx, ty, dtx, dty, np.zeros_like(x)], axis=1)


def rk4(S, z0, z1, step=STEP):
    S = S.astype(np.float64).copy()
    z = float(z0)
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
    rng = np.random.default_rng(SEED)
    z_pv = rng.uniform(-50, 50, N)
    x_pv = rng.uniform(-0.5, 0.5, N)
    y_pv = rng.uniform(-0.5, 0.5, N)
    tx = rng.uniform(-0.27, 0.27, N)
    ty = rng.uniform(-0.22, 0.22, N)
    p_gev = np.exp(rng.uniform(np.log(2.0), np.log(100.0), N))
    qop = rng.choice([-1.0, 1.0], N) * 0.299792458 / p_gev   # allen qop = c*q/p

    S0 = np.stack([x_pv, y_pv, tx, ty, qop], axis=1)
    # PV -> plane, per-z_pv (group into 20 bins to vectorise the short hop)
    plane = np.empty_like(S0)
    bins = np.digitize(z_pv, np.linspace(-50, 50, 21))
    for b in np.unique(bins):
        g = bins == b
        zb = z_pv[g].mean()
        plane[g] = rk4(S0[g], zb, ZINI)
    ok = np.all(np.isfinite(plane), axis=1)
    truth = rk4(plane[ok], ZINI, ZFIN)
    ok2 = np.all(np.isfinite(truth), axis=1)
    plane, truth, qop_s = plane[ok][ok2], truth[ok2], qop[ok][ok2]
    print(f"pointing set: {plane.shape[0]} tracks; "
          f"physical bend scale: median |tx_f - tx_i| = "
          f"{np.median(np.abs(truth[:,2]-plane[:,2])):.4f} rad "
          f"(low-p quartile {np.median(np.abs(truth[:,2]-plane[:,2])[np.abs(qop_s)>np.quantile(np.abs(qop_s),0.75)]):.4f})")

    np.savez_compressed(HERE / "kappa_val_ref.npz", X_plane=plane, Y_true=truth)
    np.savetxt(HERE / "kappa_val_states.csv", plane, delimiter=",",
               header="x,y,tx,ty,qop_corpus", comments="")
    print("wrote kappa_val_states.csv — now run extraputt_baseline, then rerun "
          "this script's compare step:")

    # compare step (if polynomial predictions exist)
    for tag in ("m1", "p1"):
        f = HERE / f"kappa_val_poly_pol{tag}.csv"
        if not f.exists():
            continue
        P = np.loadtxt(f, delimiter=",", skiprows=1)
        dx = np.abs(P[:, 0] - truth[:, 0])
        dtx = np.abs(P[:, 2] - truth[:, 2])
        q = np.abs(qop_s)
        hi = q > np.quantile(q, 0.75)  # lowest momentum
        print(f"\npolarity {tag}:  median|dx| = {np.median(dx):8.3f} mm   "
              f"p95 = {np.quantile(dx,0.95):8.3f} mm   low-p median = {np.median(dx[hi]):8.3f} mm")
        print(f"             median|dtx| = {np.median(dtx)*1e3:7.4f} mrad")


if __name__ == "__main__":
    main()
