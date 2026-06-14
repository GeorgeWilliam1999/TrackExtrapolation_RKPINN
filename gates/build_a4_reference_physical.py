#!/usr/bin/env python3
"""Rebuild the A4 Jacobian reference at PHYSICAL kappa (gen-4 contract).

Replaces the stale 2026-05-12 weak-field (kappa=1e-6) reference, which made the
A4 Jacobian gate meaningless for every gen-4 model.  The new reference uses the
*identical* truth dynamics as the gen-4 corpus generator
(data_generation/generate_data_v2.py): LHCb FieldMap v8r1 down (raw sign,
MagDown By<0), kappa = 1e-3 * qop, vectorised RK4 @ 5 mm fixed step.

What it produces
----------------
  X_a4.npy            (N, 7) fp64 input states [x,y,tx,ty,qop,z0,dz]
  J_rk4_reference.npy (N, 5, 5) fp64 d(state_out)/d(state_in) Jacobians

Draw (documented, seeded)
-------------------------
N = 300 representative *in-distribution* states, seed=20260614:
  * 200 general states drawn uniformly at random (no replacement) from the
    actual gen-4 training corpus train_10M_gen4.npz X[9.19M,7] — i.e. exactly
    the marginal the gen-4 models trained on (70% PV-pointing / 30% broad,
    log-uniform p in [1,200] GeV both charges, z0~U[0,14000], |dz| log-uniform
    [25,10000]).
  * 100 UT->T states drawn uniformly at random from the frozen UT->T evaluation
    pool utt_pool_gen4_frozen.npz X[13349,7] (z0 ~ UT plane ~2.3-3.0 m, dz ~
    UT->T gap ~4.7-7.1 m).  UT->T is the wave-2 priority arm and is
    under-represented (0.145%) in the corpus marginal, so it is explicitly
    boosted here.

Jacobian (fp64 central finite differences, convergence-checked)
---------------------------------------------------------------
J[:, j] = (prop(s + e_j) - prop(s - e_j)) / (2 e_j), all fp64.  Per-coordinate
steps match For_Allen/src/for_allen/eval/jacobian._jac_rk4_fd so the saved
reference is consistent with the gate's own FD definition:
    x,y : 1e-3 mm     tx,ty : 1e-6     qop : max(|qop|*1e-4, 1e-8)
Convergence is verified by halving every step and requiring the per-track
relative Frobenius change ||J_h - J_{h/2}||_F / ||J_{h/2}||_F < 1e-3.

NOT fp32 finite differences: fp32 round-off (~1e-7 floor) historically caused a
false A4 failure; everything here is fp64 end-to-end.

Usage:  build_a4_reference_physical.py [out_dir]
        (default out_dir = For_Allen/artifacts/phase1a)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Big data + the corpus generator live in the lab, not in the repo mirror.
LAB = Path(os.environ.get(
    "TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))

# Import the EXACT corpus truth dynamics (FIELD, KAPPA=1e-3, rk4, deriv).
# Generator dir is "data_generation/" in the lab and "datagen/" in the repo.
for _cand in (LAB / "data_generation", LAB / "datagen",
              HERE.parent.parent / "data_generation", HERE.parent / "datagen"):
    if (_cand / "generate_data_v2.py").exists():
        sys.path.insert(0, str(_cand))
        break
import generate_data_v2 as g  # noqa: E402

SEED = 20260614
N_CORPUS = 200
N_UTT = 100
CORPUS = LAB / "data" / "train_10M_gen4.npz"
UTT_POOL = LAB / "data" / "utt_pool_gen4_frozen.npz"

# Per-coordinate central-difference steps (match jacobian._jac_rk4_fd).
_EPS_FD = np.array([1e-3, 1e-3, 1e-6, 1e-6, np.nan], dtype=np.float64)


def _qop_step(qop: float) -> float:
    return max(abs(qop) * 1e-4, 1e-8)


def propagate_batch(states: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """fp64 RK4 of a [M,5] state batch from z0 to z1 (identical to corpus EOM)."""
    return g.rk4(np.asarray(states, dtype=np.float64), float(z0), float(z1))


def jac_fd(x7: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """5x5 fp64 central-difference Jacobian d(out)/d(in) for one state.

    `scale` multiplies every FD step (use 0.5 for the convergence check).
    All 10 perturbed states share (z0,dz) so they propagate in one vectorised
    RK4 call.
    """
    z0 = float(x7[5])
    z1 = float(x7[5] + x7[6])
    s0 = x7[:5].astype(np.float64)
    steps = np.array([(_EPS_FD[j] if j < 4 else _qop_step(s0[4])) * scale
                      for j in range(5)], dtype=np.float64)
    perts = np.empty((10, 5), dtype=np.float64)
    for j in range(5):
        sp = s0.copy(); sp[j] += steps[j]
        sm = s0.copy(); sm[j] -= steps[j]
        perts[2 * j] = sp
        perts[2 * j + 1] = sm
    out = propagate_batch(perts, z0, z1)            # [10,5]
    J = np.empty((5, 5), dtype=np.float64)
    for j in range(5):
        J[:, j] = (out[2 * j] - out[2 * j + 1]) / (2.0 * steps[j])
    return J


def draw_states() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(SEED)
    corpus = np.load(CORPUS)
    Xc = corpus["X"]
    idx_c = rng.choice(Xc.shape[0], N_CORPUS, replace=False)
    Xg = Xc[idx_c].astype(np.float64)

    pool = np.load(UTT_POOL)
    Xu_all = pool["X"]
    idx_u = rng.choice(Xu_all.shape[0], N_UTT, replace=False)
    Xu = Xu_all[idx_u].astype(np.float64)

    X = np.concatenate([Xg, Xu], axis=0)            # [300,7], corpus first
    meta = dict(
        seed=SEED,
        n_total=int(X.shape[0]),
        n_corpus=N_CORPUS,
        n_utt=N_UTT,
        corpus_file=str(CORPUS),
        utt_pool_file=str(UTT_POOL),
        corpus_rows=int(Xc.shape[0]),
        utt_pool_rows=int(Xu_all.shape[0]),
        fwd_frac=float(np.mean(X[:, 6] > 0)),
        qop_abs_min=float(np.min(np.abs(X[:, 4]))),
        qop_abs_max=float(np.max(np.abs(X[:, 4]))),
        absdz_min=float(np.min(np.abs(X[:, 6]))),
        absdz_max=float(np.max(np.abs(X[:, 6]))),
    )
    return X, meta


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else LAB / "For_Allen" / "artifacts" / "phase1a"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[a4] field   : {g.FIELD.info()}")
    print(f"[a4] kappa   : {g.KAPPA}  (physical)   rk4 step: {g.STEP} mm")
    X, meta = draw_states()
    n = X.shape[0]
    print(f"[a4] drew X_a4: {n} states  ({meta['n_corpus']} corpus + {meta['n_utt']} UT->T)"
          f"  fwd_frac={meta['fwd_frac']:.3f}")

    print(f"[a4] computing fp64 central-difference Jacobians on {n} states…")
    t0 = time.time()
    J = np.empty((n, 5, 5), dtype=np.float64)
    Jh = np.empty((n, 5, 5), dtype=np.float64)
    for i in range(n):
        J[i] = jac_fd(X[i], scale=1.0)
        Jh[i] = jac_fd(X[i], scale=0.5)
        if (i + 1) % 50 == 0:
            print(f"  [a4] {i+1}/{n}  ({time.time()-t0:.0f}s)")
    elapsed = time.time() - t0

    # convergence: relative Frobenius change on halving the FD step
    num = np.linalg.norm((J - Jh).reshape(n, -1), axis=1)
    den = np.maximum(np.linalg.norm(Jh.reshape(n, -1), axis=1), 1e-30)
    conv = num / den
    conv_stats = dict(
        max=float(conv.max()),
        p95=float(np.quantile(conv, 0.95)),
        median=float(np.median(conv)),
        n_fail_above_1e_3=int(np.sum(conv >= 1e-3)),
    )
    passed = conv_stats["max"] < 1e-3
    print(f"[a4] FD convergence (halve step): max={conv_stats['max']:.2e} "
          f"p95={conv_stats['p95']:.2e} median={conv_stats['median']:.2e}  "
          f"-> {'PASS' if passed else 'FAIL'} (<1e-3)")

    # qop-column physicality check (this is what weak-field destroyed)
    qcol = np.linalg.norm(J[:, :, 4], axis=1)       # ||d(out)/d(qop)|| per track
    print(f"[a4] ||d(out)/d(qop)|| : median={np.median(qcol):.3g} "
          f"max={qcol.max():.3g}  (weak-field-era would be ~1000x smaller)")

    np.save(out_dir / "X_a4.npy", X)
    np.save(out_dir / "J_rk4_reference.npy", J)
    meta.update(
        date=str(date.today()),
        kappa=g.KAPPA,
        field=g.FIELD.info(),
        rk4_step_mm=g.STEP,
        fd_steps=dict(x=1e-3, y=1e-3, tx=1e-6, ty=1e-6, qop="max(|qop|*1e-4,1e-8)"),
        fd_convergence=conv_stats,
        fd_convergence_pass=bool(passed),
        qop_col_norm_median=float(np.median(qcol)),
        qop_col_norm_max=float(qcol.max()),
        build_wall_s=round(elapsed, 1),
        dynamics_source="data_generation/generate_data_v2.py (imported)",
    )
    (out_dir / "a4_reference_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[a4] saved -> {out_dir}/X_a4.npy  J_rk4_reference.npy  a4_reference_meta.json")
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
