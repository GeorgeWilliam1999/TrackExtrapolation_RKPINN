#!/usr/bin/env python3
"""P0.1 — build the fixed-plane UT->T reference set for the extrapUTT bake-off.

The production polynomial (params_UTT_v0.tab, 25v0) is fitted for exactly
ZINI=2665 -> ZFIN=7826 mm. To compare NN vs polynomial vs RK fairly, we need
states *at* z=2665 and fine-RK truth *at* z=7826:

  1. take corpus tracks with z0 within +-300 mm of ZINI,
  2. RK-propagate each from its z0 to ZINI (short hop)  -> plane state,
  3. RK-propagate ZINI -> ZFIN (5 mm steps)             -> ground truth.

Vectorised RK4 (numpy, batch) mirroring utils/rk4_propagator.RK4Integrator
semantics exactly (full 5 mm steps + one remainder step, allen qop convention,
polarity -1); validated against the scalar reference on a subsample.

Outputs (in this directory):
  utt_plane_ref.npz   X_plane[N,5] (x,y,tx,ty,qop_corpus) @ZINI, Y_true[N,5] @ZFIN
  plane_states.csv    x,y,tx,ty,qop_corpus   (input for the C++ extrapUTT driver)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
sys.path.insert(0, str(REPO / "core"))

from magnetic_field import get_field_numpy  # noqa: E402
from rk4_propagator import RK4Integrator, _ALLEN_KAPPA_PREFACTOR  # noqa: E402

ZINI, ZFIN = 2665.0, 7826.0   # from params_UTT_v0.tab line 1
N_TRACKS = 10_000
STEP = 5.0
SEED = 20260611

field = get_field_numpy(use_interpolated=True, polarity=-1)
KAPPA = _ALLEN_KAPPA_PREFACTOR  # corpus 'allen' qop convention


def deriv_batch(S: np.ndarray, z: float) -> np.ndarray:
    """Vectorised EOM. S[N,5] = (x,y,tx,ty,qop)."""
    x, y, tx, ty, qop = S.T
    Bx, By, Bz = field(x, y, np.full_like(x, z))
    kap = KAPPA * qop
    N = np.sqrt(1.0 + tx * tx + ty * ty)
    dtx = kap * N * (tx * ty * Bx - (1.0 + tx * tx) * By + ty * Bz)
    dty = kap * N * ((1.0 + ty * ty) * Bx - tx * ty * By - tx * Bz)
    return np.stack([tx, ty, dtx, dty, np.zeros_like(x)], axis=1)


def rk4_batch(S: np.ndarray, z0: float, z1: float, step: float = STEP) -> np.ndarray:
    """Batch RK4 mirroring RK4Integrator.propagate (full steps + remainder)."""
    S = S.astype(np.float64).copy()
    z = float(z0)
    h = step if z1 > z0 else -step
    while (z1 - z) * np.sign(h) > abs(h):
        k1 = deriv_batch(S, z)
        k2 = deriv_batch(S + 0.5 * h * k1, z + 0.5 * h)
        k3 = deriv_batch(S + 0.5 * h * k2, z + 0.5 * h)
        k4 = deriv_batch(S + h * k3, z + h)
        S = S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z += h
    r = z1 - z
    if abs(r) > 1e-12:
        k1 = deriv_batch(S, z)
        k2 = deriv_batch(S + 0.5 * r * k1, z + 0.5 * r)
        k3 = deriv_batch(S + 0.5 * r * k2, z + 0.5 * r)
        k4 = deriv_batch(S + r * k3, z + r)
        S = S + (r / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def main() -> None:
    print(f"Loading corpus ...", flush=True)
    d = np.load(LAB / "data" / "train_10M_gen3.npz")
    X = d["X"]  # (x,y,tx,ty,qop,z0,dz)
    m = np.abs(X[:, 5] - ZINI) <= 300.0
    idx = np.flatnonzero(m)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(idx, size=min(N_TRACKS, idx.size), replace=False)
    sel = X[pick]
    print(f"  {idx.size} tracks with |z0-{ZINI}|<=300; using {pick.size}", flush=True)

    # hop each track from its own z0 to the ZINI plane (per-z0 groups)
    plane = np.empty((pick.size, 5))
    for z0 in np.unique(np.round(sel[:, 5], 6)):
        g = np.flatnonzero(np.isclose(sel[:, 5], z0))
        plane[g] = rk4_batch(sel[g, :5], float(z0), ZINI)
    ok = np.all(np.isfinite(plane), axis=1)
    plane = plane[ok]
    print(f"  plane states finite: {plane.shape[0]}", flush=True)

    # validate the vectorised integrator against the scalar reference
    ref = RK4Integrator(step_size=STEP, polarity=-1, qop_convention="allen")
    sub = plane[:25]
    scal = np.stack([ref.propagate(s, ZINI, ZFIN) for s in sub])
    vect = rk4_batch(sub, ZINI, ZFIN)
    dmax = np.nanmax(np.abs(scal - vect))
    print(f"  vectorised-vs-scalar max|diff| over 25 tracks = {dmax:.3e}", flush=True)
    assert dmax < 1e-6, "vectorised RK4 disagrees with the scalar reference"

    print(f"  propagating {plane.shape[0]} tracks {ZINI} -> {ZFIN} ...", flush=True)
    truth = rk4_batch(plane, ZINI, ZFIN)
    ok = np.all(np.isfinite(truth), axis=1)
    plane, truth = plane[ok], truth[ok]
    print(f"  final reference set: {plane.shape[0]} tracks", flush=True)

    np.savez_compressed(HERE / "utt_plane_ref.npz",
                        X_plane=plane.astype(np.float64),
                        Y_true=truth.astype(np.float64),
                        zini=ZINI, zfin=ZFIN, step=STEP, seed=SEED)
    np.savetxt(HERE / "plane_states.csv", plane,
               delimiter=",", header="x,y,tx,ty,qop_corpus", comments="")
    print("DONE: utt_plane_ref.npz + plane_states.csv", flush=True)


if __name__ == "__main__":
    main()
