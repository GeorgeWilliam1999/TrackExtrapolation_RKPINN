#!/usr/bin/env python3
"""Build the canonical UT->T plane reference, v8r1 throughout (P0.1 / gen-4 rematch).

Physical PV-pointing tracks generated with the SAME dynamics as the gen-4 corpus
(FieldV8R1, kappa=1e-3, rk4 imported from the corpus generator) so the plane
states are in-distribution for the gen-4 NNs AND valid inputs for the production
extrapUTT polynomial (defined exactly at z 2665 -> 7826):

    PV (z=0) --v8r1 RK--> plane z=2665  --v8r1 RK--> truth z=7826

OUTPUT
  plane_ref_v8r1.npz       X_plane[N,5]=(x,y,tx,ty,qop) @2665, Y_true[N,5] @7826
  plane_states_v8r1.csv    x,y,tx,ty,qop_corpus  (input for extraputt_baseline)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent
sys.path.insert(0, str(GEN3 / "data_generation"))
sys.path.insert(0, str(GEN3 / "utils"))
from generate_data_v2 import rk4  # noqa: E402  (v8r1 field, kappa=1e-3 — gen-4 truth dynamics)

ZPV, ZINI, ZFIN = 0.0, 2665.0, 7826.0
N = 8000
SEED = 20260614


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-0.5, 0.5, N); y = rng.uniform(-0.5, 0.5, N)
    tx = rng.uniform(-0.27, 0.27, N); ty = rng.uniform(-0.22, 0.22, N)
    p = np.exp(rng.uniform(np.log(2.0), np.log(100.0), N))
    qop = rng.choice([-1.0, 1.0], N) * 0.299792458 / p
    S0 = np.stack([x, y, tx, ty, qop], axis=1)

    plane = rk4(S0, ZPV, ZINI)                 # PV -> plane (physical)
    ok = np.all(np.isfinite(plane), axis=1) & (np.abs(plane[:, 0]) < 3900)
    plane = plane[ok]
    truth = rk4(plane, ZINI, ZFIN)             # plane -> T-stations (truth)
    ok2 = np.all(np.isfinite(truth), axis=1) & (np.abs(truth[:, 0]) < 3900)
    plane, truth = plane[ok2], truth[ok2]

    np.savez_compressed(HERE / "plane_ref_v8r1.npz", X_plane=plane, Y_true=truth)
    np.savetxt(HERE / "plane_states_v8r1.csv", plane, delimiter=",",
               header="x,y,tx,ty,qop_corpus", comments="")
    bend = np.median(np.abs(truth[:, 2] - plane[:, 2]))
    print(f"plane_ref_v8r1: {plane.shape[0]} tracks  median |dtx| over leg = {bend:.4f} rad")
    print(f"  x@plane range [{plane[:,0].min():.0f},{plane[:,0].max():.0f}] mm  "
          f"(physical pointing tracks, v8r1)")


if __name__ == "__main__":
    main()
