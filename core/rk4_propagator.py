#!/usr/bin/env python3
"""
Gen-3 RK4 Track Propagator (Allen-convention aware).

Differences from Gen-2 (experiments/gen_2/utils/rk4_propagator.py):

1. New ``qop_convention`` argument on ``RK4Integrator`` (default ``"allen"``):
     - "allen":  qop = c_light * q/p_MeV  (c_light = 299.792458 mm/ns/eplus)
                 => kappa = 1.0e-6 * qop
     - "legacy": qop = q/p_MeV
                 => kappa = 2.99792458e-4 * qop  (Gen-2 behaviour)
   The Lorentz RHS is unchanged; only the numerical value of ``c_light`` used
   to form ``kappa`` differs. The two conventions produce IDENTICAL kappa
   (and therefore identical trajectories) for the same physical track.

2. Signed dz is supported natively (same as Gen-2): the step sign is flipped
   when dz < 0, and the integrator halts cleanly at z_end on either side.

3. Acceptance cuts during integration are loose (|x|, |y| > 5000 mm or
   |tx|, |ty| > 1.0 => NaN return). Tighter Gen-3 cuts (|tx_f|, |ty_f| > 0.5)
   are applied post-hoc by the data generator, not in-loop.

Author: G. Scriven
Date: 2026-04-23
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Optional

# Prefer the Gen-3 magnetic_field copy (bundles the same InterpolatedFieldNumpy)
sys.path.insert(0, str(Path(__file__).parent))
from magnetic_field import get_field_numpy, C_LIGHT as _LEGACY_C_LIGHT  # noqa: E402

# Kappa prefactor when qop is already in Allen units (qop = 299.792458 * q/p_MeV).
#
# P0.0 FIX (2026-06-11): was 1.0e-6 — x1000 too weak. Derivation:
#   dtheta/ds = 0.299792458 [GeV/(T m)] * B[T] / p[GeV]  per metre
#             = 2.99792458e-4 * B * (q/p)[1/GeV]         per mm
#             = 0.299792458  * B * (q/p)[1/MeV]          per mm
#   with qop_allen = 299.792458 * (q/p)[1/MeV]:
#   dtheta/ds = 1.0e-3 * qop_allen * B  per mm.
# The old value matched the "legacy" pairing (2.998e-4 with q/p in 1/MeV),
# which is itself wrong by x1000 (2.998e-4 belongs with q/p in 1/GeV) — the
# bug therefore predates gen-3. Verified externally against the production
# extrapUTT polynomial (see paper_p0/, P0.1 bake-off 2026-06-11).
_ALLEN_KAPPA_PREFACTOR = 1.0e-3


class RK4Integrator:
    """Fourth-order RK4 integrator with configurable qop convention."""

    def __init__(
        self,
        field=None,
        step_size: float = 5.0,
        use_interpolated_field: bool = True,
        polarity: int = -1,
        qop_convention: str = "allen",
        verbose: bool = False,
    ):
        if field is not None:
            self.field = field
        else:
            self.field = get_field_numpy(
                use_interpolated=use_interpolated_field, polarity=polarity
            )

        self.step_size = float(step_size)

        if qop_convention == "allen":
            self.c_light = _ALLEN_KAPPA_PREFACTOR
        elif qop_convention == "legacy":
            self.c_light = _LEGACY_C_LIGHT
        else:
            raise ValueError(
                f"qop_convention must be 'allen' or 'legacy', got {qop_convention!r}"
            )
        self.qop_convention = qop_convention

        if verbose:
            print(
                f"[RK4] field={type(self.field).__name__} "
                f"step={self.step_size}mm qop_convention={qop_convention} "
                f"c_light_eff={self.c_light:.6e}"
            )

    # ------------------------------------------------------------------
    # Core equations of motion
    # ------------------------------------------------------------------
    def derivatives(self, state: np.ndarray, z: float) -> np.ndarray:
        x, y, tx, ty, qop = state
        Bx, By, Bz = self.field(x, y, z)
        kappa = self.c_light * qop
        N = np.sqrt(1.0 + tx * tx + ty * ty)

        dtx_dz = kappa * N * (tx * ty * Bx - (1.0 + tx * tx) * By + ty * Bz)
        dty_dz = kappa * N * ((1.0 + ty * ty) * Bx - tx * ty * By - tx * Bz)
        return np.array([tx, ty, dtx_dz, dty_dz, 0.0])

    def rk4_step(self, state: np.ndarray, z: float, h: float) -> np.ndarray:
        k1 = self.derivatives(state, z)
        k2 = self.derivatives(state + 0.5 * h * k1, z + 0.5 * h)
        k3 = self.derivatives(state + 0.5 * h * k2, z + 0.5 * h)
        k4 = self.derivatives(state + h * k3, z + h)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def propagate(
        self,
        state: np.ndarray,
        z_start: float,
        z_end: float,
        save_trajectory: bool = False,
    ) -> np.ndarray:
        current_state = state.astype(np.float64, copy=True)
        current_z = float(z_start)
        dz_total = float(z_end) - float(z_start)

        if dz_total == 0.0:
            return current_state.copy() if not save_trajectory else np.array(
                [[current_z, *current_state]]
            )

        step = self.step_size if dz_total > 0 else -self.step_size
        trajectory = [[current_z, *current_state]] if save_trajectory else None

        with np.errstate(over="ignore", invalid="ignore"):
            while (z_end - current_z) * np.sign(step) > abs(step):
                current_state = self.rk4_step(current_state, current_z, step)
                current_z += step

                if not np.all(np.isfinite(current_state)):
                    return np.full(5, np.nan) if not save_trajectory else np.array(trajectory)

                # Loose in-loop divergence cuts (tighter cuts done in caller)
                if abs(current_state[0]) > 5000.0 or abs(current_state[1]) > 5000.0:
                    return np.full(5, np.nan) if not save_trajectory else np.array(trajectory)
                if abs(current_state[2]) > 1.0 or abs(current_state[3]) > 1.0:
                    return np.full(5, np.nan) if not save_trajectory else np.array(trajectory)

                if save_trajectory:
                    trajectory.append([current_z, *current_state])

            remaining = z_end - current_z
            if abs(remaining) > 1e-6:
                current_state = self.rk4_step(current_state, current_z, remaining)
                if save_trajectory:
                    trajectory.append([z_end, *current_state])

        if save_trajectory:
            return np.array(trajectory)
        return current_state


__all__ = ["RK4Integrator"]
