#!/usr/bin/env python3
"""F0 — Build the on-axis field-integral tables F(z), G(z) and calibrate kappa.

Theory (see the Notion write-up "Analytic Flattening — Theory" §3):
  For the idealised on-axis field B = B_y(z) ŷ, the canonical momentum
  P_x = p_x + q·F(z) is exactly conserved, with F(z) = ∫ B_y(0,0,z') dz'.
  In slope variables the order-1 (kick) chart is
      tx(z1) = tx0 − κ·[F(z1) − F(z0)]
      x(z1)  = x0 + tx0·Δz − κ·[G(z1) − G(z0) − F(z0)·Δz],   G = ∫F
  with κ = κ0·qop. The unit prefactor κ0 is NOT derived from conventions
  (unit traps) but CALIBRATED empirically: regress Δtx_true against
  −qop·ΔF on high-momentum, small-|dz| tracks where the order-1 chart is
  near-exact.

Outputs: charts/field_integrals.npz  (z_grid, F, G, kappa0, metadata)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent.parent / "gen_3"
sys.path.insert(0, str(GEN3 / "utils"))

from magnetic_field import get_field_numpy, C_LIGHT  # noqa: E402

Z_MIN, Z_MAX, DZ = -500.0, 14000.0, 5.0   # mm — full map range, fine grid
POLARITY = -1                              # MagDown: matches the gen-3 corpus


def build_tables():
    field = get_field_numpy(use_interpolated=True, polarity=POLARITY)
    z = np.arange(Z_MIN, Z_MAX + DZ, DZ)
    zeros = np.zeros_like(z)
    Bx, By, Bz = field(zeros, zeros, z)
    By = np.asarray(By, dtype=np.float64)

    # F(z) = ∫ B_y(0,0,z') dz'  (cumulative trapezoid, F(z_min)=0)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (By[1:] + By[:-1]) * np.diff(z))])
    # G(z) = ∫ F(z') dz'
    G = np.concatenate([[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(z))])

    print(f"grid: z ∈ [{z[0]:.0f}, {z[-1]:.0f}] mm, {len(z)} points, Δz={DZ} mm")
    print(f"peak |B_y| = {np.abs(By).max():.4f} T at z = {z[np.argmax(np.abs(By))]:.0f} mm")
    print(f"total integrated field I1(full) = {F[-1]:.2f} T·mm = {F[-1]/1000:.3f} T·m")
    return z, F, G, By


def calibrate_kappa(z_grid, F):
    """Empirical kappa0: Δtx_true ≈ −kappa0·qop·ΔF on high-p, moderate-dz tracks."""
    data = np.load(GEN3 / "data" / "train_10M_gen3.npz")
    X, Y = data["X"], data["Y"]
    qop, z0, dz = X[:, 4], X[:, 5], X[:, 6]
    # selection: forward, well inside the map, high momentum (small |qop| -> chart near-exact),
    # near-axis start (|x0|,|y0| small) to suppress transverse-field contamination
    m = (
        (dz > 1000) & (dz < 8000)
        & (z0 > 0) & (z0 + dz < 13900)
        & (np.abs(qop) < 0.05)            # p > 20 GeV: trajectory feedback ~ O(kappa^2) negligible
        & (np.abs(X[:, 0]) < 300) & (np.abs(X[:, 1]) < 300)
        & (np.abs(X[:, 2]) < 0.15) & (np.abs(X[:, 3]) < 0.15)
    )
    idx = np.flatnonzero(m)[:200_000]
    print(f"calibration sample: {len(idx):,} tracks")

    Fi = np.interp(X[idx, 5], z_grid, F)
    Ff = np.interp(X[idx, 5] + X[idx, 6], z_grid, F)
    dtx_true = (Y[idx, 2] - X[idx, 2]).astype(np.float64)
    u = -(X[idx, 4].astype(np.float64)) * (Ff - Fi)      # predictor: −qop·ΔF

    kappa0 = float(np.dot(u, dtx_true) / np.dot(u, u))   # least squares through origin
    resid = dtx_true - kappa0 * u
    r2 = 1.0 - np.var(resid) / np.var(dtx_true)
    print(f"kappa0 = {kappa0:.8e}   (C_LIGHT = {C_LIGHT:.8e}, ratio = {kappa0/C_LIGHT:.4f})")
    print(f"R^2 of order-1 slope chart on calibration sample: {r2:.6f}")
    print(f"residual slope error: median {np.median(np.abs(resid))*1e6:.2f} µrad, "
          f"p95 {np.quantile(np.abs(resid),0.95)*1e6:.2f} µrad")
    return kappa0, r2


def main():
    z, F, G, By = build_tables()
    kappa0, r2 = calibrate_kappa(z, F)
    out = HERE / "field_integrals.npz"
    np.savez(
        out,
        z_grid=z, F=F, G=G, By_on_axis=By,
        kappa0=kappa0, calib_r2=r2,
        polarity=POLARITY, dz_grid=DZ,
        source="experiments/field_maps/twodip.rtf",
        c_light=C_LIGHT,
    )
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
