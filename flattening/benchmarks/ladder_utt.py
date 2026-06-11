#!/usr/bin/env python3
"""F1 — The ladder benchmark: analytic flattening rungs vs RK truth on UT->T.

Rungs evaluated here (no training involved):
  rung 0  straight line:  x1 = x0 + tx0*dz,                 tx1 = tx0
  rung 1  kick chart:     tx1 = tx0 - k*[F(z1)-F(z0)]
                          x1  = x0 + tx0*dz - k*[G(z1)-G(z0)-F(z0)*dz]
          with k = kappa0*qop and F,G from charts/field_integrals.npz.
  (y-sector: straight in both rungs at this order.)

Scored on the FULL-corpus UT->T pool (analytic rungs have no training
leakage, so the full pool is valid), same window as run_r7:
  z0 in [2300,3000], z0+dz in [7600,9500], dz>0.
NN numbers from results/R7_utt_eval_2026-06-10.json are quoted alongside
(theirs are on their own test splits — same window, comparable population).

Falsifiable prediction (theory write-up §6): the rung-1 residual's
median-|dx| by-|q/p|-quartile profile is ~flat (kappa^2 envelope), unlike
rung 0 (prop. to kappa) and unlike the kick-scaled-head NNs (Q4 blow-up).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
GEN3 = FLAT.parent / "gen_3"

UT_Z = (2300.0, 3000.0)
T_Z = (7600.0, 9500.0)


def metrics(dx_mm, dtx, qop, label):
    a = np.abs(dx_mm) * 1e3  # um
    q = np.abs(qop)
    edges = np.quantile(q, [0.25, 0.5, 0.75])
    bins = np.digitize(q, edges)
    byq = [float(np.median(a[bins == k])) for k in range(4)]
    out = {
        "label": label,
        "n": int(a.size),
        "median_dx_um": float(np.median(a)),
        "p68_dx_um": float(np.quantile(a, 0.68)),
        "p95_dx_um": float(np.quantile(a, 0.95)),
        "p99_dx_um": float(np.quantile(a, 0.99)),
        "median_dtx_urad": float(np.median(np.abs(dtx)) * 1e6),
        "median_dx_um_by_qop_quartile": byq,
    }
    print(f"{label:<26} med={out['median_dx_um']:9.1f}um  p95={out['p95_dx_um']:9.0f}  "
          f"|dtx|med={out['median_dtx_urad']:8.1f}urad  byQ={[round(v,1) for v in byq]}")
    return out


def main():
    t = np.load(FLAT / "charts" / "field_integrals.npz")
    zg, F, G, kappa0 = t["z_grid"], t["F"], t["G"], float(t["kappa0"])
    print(f"kappa0 = {kappa0:.6e}   I1(full) = {F[-1]/1000:.3f} T·m")

    d = np.load(GEN3 / "data" / "train_10M_gen3.npz")
    X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]
    zf = z0 + dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    print(f"UT->T pool (full corpus): {Xs.shape[0]:,} tracks\n")

    x0, y0, tx0, ty0, qop = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4]
    z0s, dzs = Xs[:, 5], Xs[:, 6]
    z1s = z0s + dzs
    k = kappa0 * qop

    results = {}

    # rung 0 — straight line
    x_r0 = x0 + tx0 * dzs
    results["rung0_straight"] = metrics(x_r0 - Ys[:, 0], tx0 - Ys[:, 2], qop, "rung0 straight")

    # rung 1 — kick chart
    F0, F1 = np.interp(z0s, zg, F), np.interp(z1s, zg, F)
    G0, G1 = np.interp(z0s, zg, G), np.interp(z1s, zg, G)
    tx_r1 = tx0 - k * (F1 - F0)
    x_r1 = x0 + tx0 * dzs - k * (G1 - G0 - F0 * dzs)
    results["rung1_kick_chart"] = metrics(x_r1 - Ys[:, 0], tx_r1 - Ys[:, 2], qop, "rung1 kick chart")

    # NN reference numbers (R7, own test splits, same window) for context
    r7 = GEN3 / "results" / "R7_utt_eval_2026-06-10.json"
    if r7.exists():
        rj = json.load(open(r7))
        print("\n-- NN references (R7, test splits) --")
        for name in ["pinn_v2_small_v1", "pinn_v2_kick_10M"]:
            mm = rj.get(name, {})
            if "median_dx_um" in mm:
                print(f"{name:<26} med={mm['median_dx_um']:9.1f}um  p95={mm['p95_dx_um']:9.0f}  "
                      f"byQ={[round(v,1) for v in mm['median_dx_um_by_|q/p|_quartile']]}")
                results[f"nn_{name}"] = {k2: mm[k2] for k2 in
                                         ("median_dx_um", "p95_dx_um", "median_dx_um_by_|q/p|_quartile")}

    out = FLAT / "results" / f"F1_ladder_{datetime.now().strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
