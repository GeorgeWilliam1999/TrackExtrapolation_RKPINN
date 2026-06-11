#!/usr/bin/env python3
# Copyright 2026 CERN for the benefit of the LHCb Collaboration.
# Licensed under the Apache License, Version 2.0.
"""
Phase 1a — Architectural ablation (NO training).

Per ``For_Allen/PLAN.md`` Phase 1a and ADRs 0002 (Action N: corrector
removed) and 0003 (multi-step RK4 mandatory):

  Sweep grid:   n_rk_steps ∈ {1, 2, 4, 8, 16}  ×  corrector ∈ {on, off}

For each cell:

  * A4 Frobenius rel-err  ‖J_model − J_RK45‖_F / ‖J_RK45‖_F
    on N_A4 random states from the frozen test set, against an
    independent fine-grained RK4 reference (5 mm step, analytic dipole).
    Gate (Phase 1a, relaxed): < 0.10.

  * Stage-1 endpoint metrics on N_S1 random states from the frozen
    test set, split by z_f window:
        VELO_exit  : 500  < z_f < 1500   gate ⟨|Δx|⟩ < 24 µm
        UT_entry   : 1500 < z_f < 4000   gate ⟨|Δx|⟩ < 100 µm

The winner is the cell with the smallest n_rk_steps that satisfies all
three gates simultaneously with corrector OFF (ADR 0002 is frozen).

Outputs:
  * ``artifacts/phase1a/ablation.csv``
  * ``artifacts/phase1a/summary.txt``
  * ``pins/n_rk_steps_prod.txt``           (winner step count)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
FOR_ALLEN = HERE.parent
GEN3_ROOT = FOR_ALLEN.parent
sys.path.insert(0, str(GEN3_ROOT))
sys.path.insert(0, str(GEN3_ROOT / "core"))

from models.architectures import NeuralRK4  # noqa: E402
from rk4_propagator import RK4Integrator  # type: ignore  # noqa: E402


# --------------------------------------------------------------------------- #
# Sweep grid (PLAN §"Sweep grid")
# --------------------------------------------------------------------------- #
N_RK_STEPS_GRID: Tuple[int, ...] = (1, 2, 4, 8, 16)
# ADR 0002 (Action N) freezes the corrector OFF as the production decision.
# The corrector-ON column in the original PLAN sweep grid is informational
# only; we omit it here because (a) it cannot change the production choice,
# and (b) the M1 checkpoint's corrector_net was trained with a 6-feature
# input that pre-dates the current 8-feature definition, so loading it
# would require reviving the legacy code path outside this script's scope.
CORRECTOR_GRID: Tuple[bool, ...] = (False,)

# Phase 1a relaxed gates (PLAN §"Acceptance", ACCEPTANCE.md row 1a):
GATE_A4_FROB = 0.10        # ‖J − J_RK4‖_F / ‖J_RK4‖_F
GATE_VELO_UM = 24.0        # ⟨|Δx|⟩ at VELO exit (µm)
GATE_UT_UM = 100.0         # ⟨|Δx|⟩ at UT entry  (µm)

# §A4 finite-difference step sizes (per gen-3 protocol):
EPS_FD = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-4], dtype=np.float64)

# Stage-1 z-windows (deep-dive §6):
Z_WINDOWS: Dict[str, Tuple[float, float, float]] = {
    "VELO_exit":  (500.0,  1500.0, GATE_VELO_UM),
    "UT_entry":   (1500.0, 4000.0, GATE_UT_UM),
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_model(run_dir: Path) -> NeuralRK4:
    """Load the frozen NRK4 checkpoint with its original training config."""
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    model = NeuralRK4(
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        n_rk_steps=int(cfg["n_rk_steps"]),
        # corrector ablation is set per sweep cell, NOT here
    )
    model.load_normalization(str(run_dir / "normalization.json"))
    state = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # The M1 checkpoint corrector_net has a 6-feature input layer; the current
    # NeuralRK4 corrector_net expects 8 features.  With Action N (ADR 0002)
    # the corrector is disabled in this sweep, so we drop the legacy
    # correction_net.* / log_corr_scale tensors from the state dict and
    # exercise the pure Lorentz-RK4 path unchanged.
    state = {
        k: v for k, v in state.items()
        if not (k.startswith("correction_net.") or k == "log_corr_scale")
    }
    incompat = model.load_state_dict(state, strict=False)
    bad = [
        k for k in (list(incompat.missing_keys) + list(incompat.unexpected_keys))
        if not (k.startswith("correction_net.") or k == "log_corr_scale")
    ]
    if bad:
        raise RuntimeError(f"unexpected state_dict mismatches: {bad}")
    model.disable_correction = True
    model.eval()
    return model


def load_test_data(
    data_path: Path, test_idx_path: Path, max_tracks: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_test, Y_test) — (B, 7) inputs and (B, 5) targets."""
    npz = np.load(data_path, allow_pickle=False)
    X_all = npz["X"].astype(np.float32)
    Y_all = npz["Y"].astype(np.float32)
    idx = np.load(test_idx_path)
    if max_tracks is not None and idx.size > max_tracks:
        idx = idx[:max_tracks]
    return X_all[idx], Y_all[idx]


# --------------------------------------------------------------------------- #
# A4 — Jacobian agreement
# --------------------------------------------------------------------------- #
def jac_model_autograd(model: NeuralRK4, x7: np.ndarray) -> np.ndarray:
    """5×5 model Jacobian wrt (x, y, tx, ty, qop) via autograd, fp64.

    The reference Jacobian is built from a fine-grained fp64 RK4 propagator;
    matching it with FD on a fp32 model gives spurious results once the
    model is accurate to fp32 round-off (~ few × 1e-7 relative). Casting
    the model to fp64 and using autograd both removes the FD noise floor
    and gives an exact (to machine epsilon) Jacobian of the model.
    """
    xt = torch.from_numpy(x7.astype(np.float64)).unsqueeze(0).requires_grad_(True)
    y = model(xt)
    jac = torch.zeros(5, 5, dtype=torch.float64)
    for i in range(5):
        grad = torch.autograd.grad(
            y[0, i], xt, retain_graph=(i < 4), create_graph=False,
        )[0][0, :5]
        jac[i, :] = grad
    return jac.detach().numpy()


def jac_rk4_fd(rk4: RK4Integrator, x7: np.ndarray) -> np.ndarray:
    """5×5 reference Jacobian via the fine-grained RK4 propagator (fp64)."""
    z0, dz = float(x7[5]), float(x7[6])
    J = np.zeros((5, 5), dtype=np.float64)
    for j in range(5):
        eps_j = EPS_FD[j] if j != 4 else max(abs(x7[4]) * 1e-4, 1e-8)
        sp = x7[:5].astype(np.float64).copy(); sp[j] += eps_j
        sm = x7[:5].astype(np.float64).copy(); sm[j] -= eps_j
        yp = rk4.propagate(sp, z0, z0 + dz)
        ym = rk4.propagate(sm, z0, z0 + dz)
        J[:, j] = (yp - ym) / (2.0 * eps_j)
    return J


def compute_a4_frobenius(
    model_fp64: NeuralRK4, X_a4: np.ndarray, J_ref_stack: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """Per-track Frobenius + per-element off-diagonal rel-err.

    Returns the summary metrics dict and the stacked model Jacobians for
    forensic re-analysis. The off-diagonal metric mirrors the deep-dive
    §22 per-element check on a clean (fp64, autograd) footing.
    """
    n = X_a4.shape[0]
    rel_errs = np.empty(n, dtype=np.float64)
    J_m_stack = np.empty_like(J_ref_stack)
    # Build a (5,5) mask that excludes the (4,*) and (*,4) qop pass-through row+col
    # (always identity by construction) and the diagonal.
    off_mask = np.ones((5, 5), dtype=bool)
    off_mask[np.eye(5, dtype=bool)] = False
    off_mask[4, :] = False
    off_mask[:, 4] = False
    off_max_rel_per_track = np.empty(n, dtype=np.float64)
    for i in range(n):
        J_m = jac_model_autograd(model_fp64, X_a4[i].astype(np.float64))
        J_m_stack[i] = J_m
        J_r = J_ref_stack[i]
        rel_errs[i] = (
            np.linalg.norm(J_m - J_r) / max(np.linalg.norm(J_r), 1e-30)
        )
        # Per-element rel-err on the kinematic off-diagonals only.
        ref_off = np.abs(J_r[off_mask])
        diff_off = np.abs((J_m - J_r)[off_mask])
        # Guard against ref entries that are structurally zero (use abs-gate).
        denom = np.maximum(ref_off, 1e-12)
        off_rel = diff_off / denom
        off_max_rel_per_track[i] = float(off_rel.max())
    return (
        {
            "frob_rel_mean":   float(rel_errs.mean()),
            "frob_rel_median": float(np.median(rel_errs)),
            "frob_rel_p95":    float(np.quantile(rel_errs, 0.95)),
            "off_max_rel_median": float(np.median(off_max_rel_per_track)),
            "off_max_rel_p95":    float(np.quantile(off_max_rel_per_track, 0.95)),
        },
        J_m_stack,
    )


# --------------------------------------------------------------------------- #
# Stage-1 endpoint
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_batched(model: NeuralRK4, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    out = np.empty((X.shape[0], 5), dtype=np.float32)
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i + batch])
        out[i:i + batch] = model(xb).numpy()
    return out


def compute_stage1(
    model: NeuralRK4, X: np.ndarray, Y: np.ndarray
) -> Dict[str, float]:
    """Stage-1 ⟨|Δx|⟩ per z-window in µm."""
    Yp = predict_batched(model, X)
    z_f = X[:, 5] + X[:, 6]
    dx_um = (Yp[:, 0] - Y[:, 0]) * 1000.0
    out: Dict[str, float] = {}
    for name, (z_lo, z_hi, _gate) in Z_WINDOWS.items():
        m = (z_f >= z_lo) & (z_f < z_hi)
        if m.sum() < 5:
            out[f"{name}_dx_um"] = float("nan")
            out[f"{name}_n"] = int(m.sum())
        else:
            out[f"{name}_dx_um"] = float(np.abs(dx_um[m]).mean())
            out[f"{name}_n"] = int(m.sum())
    return out


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=GEN3_ROOT / "trained_models" / "nrk4_tiny_1step_v1",
        help="trained model directory (must contain best_model.pt, "
             "config.json, normalization.json, test_indices.npy)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=GEN3_ROOT / "data" / "train_10M_gen3.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FOR_ALLEN / "artifacts" / "phase1a",
    )
    parser.add_argument("--n-a4", type=int, default=200,
                        help="number of tracks for A4 Jacobian metric")
    parser.add_argument("--n-stage1", type=int, default=5000,
                        help="number of tracks for stage-1 endpoint metric")
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ----- Load frozen model -----
    print(f"[phase1a] Loading frozen model from {args.run_dir}")
    model = load_model(args.run_dir)
    print(f"[phase1a] Model trainable params: {model.count_parameters():,}")

    # An fp64 copy is used exclusively for the autograd Jacobian on the
    # A4 metric: matching the fp64 RK4 reference with fp32 model + FD
    # introduces a ~1e-6 relative floor that masks the true integrator
    # behaviour above n_rk_steps ~ 4.
    import copy as _copy
    model_fp64 = _copy.deepcopy(model).double().eval()

    # ----- Load test set -----
    test_idx_path = args.run_dir / "test_indices.npy"
    if not test_idx_path.exists():
        print(f"[phase1a] FATAL: {test_idx_path} not found", file=sys.stderr)
        return 2
    X_test, Y_test = load_test_data(args.data_path, test_idx_path)
    print(f"[phase1a] Test set: X={X_test.shape}  Y={Y_test.shape}")

    # ----- Pick A4 + stage-1 subsets (balanced over fwd / bwd) -----
    n_test = X_test.shape[0]
    fwd_idx = np.where(X_test[:, 6] > 0)[0]
    bwd_idx = np.where(X_test[:, 6] < 0)[0]
    n_a4 = min(args.n_a4, len(fwd_idx) + len(bwd_idx))
    n_a4_half = n_a4 // 2
    a4_idx = np.concatenate([
        rng.choice(fwd_idx, min(n_a4_half, len(fwd_idx)), replace=False),
        rng.choice(bwd_idx, n_a4 - min(n_a4_half, len(fwd_idx)),
                   replace=False),
    ])
    rng.shuffle(a4_idx)
    n_s1 = min(args.n_stage1, n_test)
    s1_idx = rng.choice(n_test, n_s1, replace=False)
    X_a4 = X_test[a4_idx].astype(np.float64)
    X_s1 = X_test[s1_idx]
    Y_s1 = Y_test[s1_idx]

    # ----- Precompute reference Jacobians once (independent of cell) -----
    print(f"[phase1a] Building RK4 reference Jacobians on {n_a4} tracks "
          f"(this is the slow step — do it once)…")
    rk4 = RK4Integrator(
        use_interpolated_field=True, polarity=-1,
        qop_convention="allen", step_size=5.0, verbose=False,
    )
    t0 = time.time()
    J_ref_stack = np.empty((n_a4, 5, 5), dtype=np.float64)
    for i in range(n_a4):
        J_ref_stack[i] = jac_rk4_fd(rk4, X_a4[i])
        if (i + 1) % 50 == 0:
            print(f"  reference jacobian {i+1}/{n_a4}  "
                  f"({(time.time()-t0):.0f}s elapsed)")
    print(f"[phase1a] Reference Jacobians done in {time.time()-t0:.0f} s")

    # ----- Sweep cells -----
    rows: List[Dict] = []
    for corr_on in CORRECTOR_GRID:
        for n_steps in N_RK_STEPS_GRID:
            label = (
                f"n_rk_steps={n_steps:>2d} corrector={'ON ' if corr_on else 'OFF'}"
            )
            print(f"\n[phase1a] Cell  {label}")
            model.n_rk_steps = int(n_steps)
            model.disable_correction = not corr_on
            model_fp64.n_rk_steps = int(n_steps)
            model_fp64.disable_correction = not corr_on

            t_cell = time.time()
            a4, J_m_stack = compute_a4_frobenius(model_fp64, X_a4, J_ref_stack)
            t_a4 = time.time() - t_cell
            t_cell = time.time()
            s1 = compute_stage1(model, X_s1, Y_s1)
            t_s1 = time.time() - t_cell

            velo_ok = (
                np.isfinite(s1["VELO_exit_dx_um"]) and
                s1["VELO_exit_dx_um"] < GATE_VELO_UM
            )
            ut_ok = (
                np.isfinite(s1["UT_entry_dx_um"]) and
                s1["UT_entry_dx_um"] < GATE_UT_UM
            )
            a4_ok = a4["frob_rel_mean"] < GATE_A4_FROB
            all_ok = bool(velo_ok and ut_ok and a4_ok)

            print(f"    A4 frob rel (mean / median / p95)  "
                  f"{a4['frob_rel_mean']:.3e} / "
                  f"{a4['frob_rel_median']:.3e} / "
                  f"{a4['frob_rel_p95']:.3e}    "
                  f"gate<{GATE_A4_FROB:.2f}   "
                  f"{'PASS' if a4_ok else 'FAIL'}")
            print(f"    A4 off-diag rel (median / p95)     "
                  f"{a4['off_max_rel_median']:.3e} / "
                  f"{a4['off_max_rel_p95']:.3e}")
            print(f"    VELO ⟨|Δx|⟩ = {s1['VELO_exit_dx_um']:7.2f} µm  "
                  f"(n={s1['VELO_exit_n']})  "
                  f"gate<{GATE_VELO_UM} µm   {'PASS' if velo_ok else 'FAIL'}")
            print(f"    UT   ⟨|Δx|⟩ = {s1['UT_entry_dx_um']:7.2f} µm  "
                  f"(n={s1['UT_entry_n']})  "
                  f"gate<{GATE_UT_UM} µm  {'PASS' if ut_ok else 'FAIL'}")
            print(f"    cell wall time: A4 {t_a4:.0f}s, stage-1 {t_s1:.0f}s   "
                  f"OVERALL {'PASS' if all_ok else 'FAIL'}")

            # Persist per-track model Jacobians for forensic
            np.save(
                args.out_dir / f"J_model_n{n_steps:02d}_corr_"
                f"{'on' if corr_on else 'off'}.npy",
                J_m_stack,
            )

            rows.append({
                "n_rk_steps": n_steps,
                "corrector":  "on" if corr_on else "off",
                **a4,
                **s1,
                "a4_pass":    bool(a4_ok),
                "velo_pass":  bool(velo_ok),
                "ut_pass":    bool(ut_ok),
                "all_pass":   bool(all_ok),
                "wall_a4_s":  t_a4,
                "wall_s1_s":  t_s1,
            })

    # Also save the inputs once for reproducibility
    np.save(args.out_dir / "X_a4.npy", X_a4)
    np.save(args.out_dir / "J_rk4_reference.npy", J_ref_stack)

    # ----- Write CSV -----
    csv_path = args.out_dir / "ablation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[phase1a] Wrote {csv_path}")

    # ----- Pick winner (corrector OFF; ADR 0002 frozen) -----
    off_rows = [r for r in rows if r["corrector"] == "off" and r["all_pass"]]
    off_rows.sort(key=lambda r: r["n_rk_steps"])  # smallest n that passes
    summary_lines: List[str] = []
    summary_lines.append("Phase 1a — Architectural ablation summary")
    summary_lines.append("=" * 60)
    summary_lines.append(
        f"Gates: A4 frob < {GATE_A4_FROB},  "
        f"VELO < {GATE_VELO_UM} µm,  UT < {GATE_UT_UM} µm"
    )
    summary_lines.append("")
    if off_rows:
        winner = off_rows[0]
        summary_lines.append(
            f"WINNER (corrector OFF, smallest n_rk_steps passing all gates):"
        )
        summary_lines.append(
            f"  n_rk_steps = {winner['n_rk_steps']}"
        )
        summary_lines.append(
            f"  A4 frob rel-err = {winner['frob_rel_mean']:.3f}  "
            f"(median {winner['frob_rel_median']:.3f}, "
            f"p95 {winner['frob_rel_p95']:.3f})"
        )
        summary_lines.append(
            f"  VELO ⟨|Δx|⟩    = {winner['VELO_exit_dx_um']:.2f} µm"
        )
        summary_lines.append(
            f"  UT   ⟨|Δx|⟩    = {winner['UT_entry_dx_um']:.2f} µm"
        )
        # Pin
        pins_dir = FOR_ALLEN / "pins"
        pins_dir.mkdir(exist_ok=True)
        pin_path = pins_dir / "n_rk_steps_prod.txt"
        pin_path.write_text(f"{winner['n_rk_steps']}\n")
        summary_lines.append(f"\nPinned {pin_path} = {winner['n_rk_steps']}")
    else:
        summary_lines.append(
            "NO CELL with corrector OFF passed all three gates."
        )
        summary_lines.append(
            "Escalate: kill criterion (see PLAN §Phase 1a / kill criteria)."
        )
        summary_lines.append(
            "Likely actions: widen RHS hidden dims (Phase 1c) "
            "or learned residual on RK45 base (different family)."
        )

    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print("\n" + "\n".join(summary_lines))
    print(f"\n[phase1a] Wrote {summary_path}")
    return 0 if off_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
