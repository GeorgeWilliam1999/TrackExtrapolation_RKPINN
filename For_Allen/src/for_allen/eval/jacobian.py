"""jacobian.py — Generic A4 Jacobian evaluation utility (Phase R2).

Lifted and generalised from ``For_Allen/scripts/phase1a_arch_ablation.py``
which was NeuralRK4-specific.  This module works with *any* gen-3 model that
follows the standard ``model(x7) -> y5`` interface, including MLP and PINN_v2.

A4 gate (from REPLACEMENT_PLAN §5 / Allen constraints):
    frob_rel_mean  < 0.05   (Frobenius rel-err, mean over test tracks)
    off_max_rel_p95 < 0.20  (max off-diagonal rel-err, 95th percentile)

Usage
-----
    from for_allen.eval.jacobian import evaluate_a4, load_reference_jacobians

    J_ref = load_reference_jacobians(artifacts_dir / "J_rk4_reference.npy")
    X_a4  = np.load(artifacts_dir / "X_a4.npy")
    report = evaluate_a4(model, X_a4, J_ref)
    print(report.summary())
    print("PASS" if report.passes_gate() else "FAIL")
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# A4 hard gates (REPLACEMENT_PLAN §5)
# off_max_frob = max|J_m[i,j] - J_r[i,j]| / ||J_r||_F  for off-diag (excl qop)
# This matrix-scale-normalised metric replaces the original element-wise
# relative error which blows up whenever J_r[i,j] ~ 0 (physically irrelevant).
GATE_FROB_REL_MEAN = 0.05
GATE_OFF_MAX_FROB_P95 = 0.05

# Finite-difference step sizes used when building the RK4 reference Jacobian.
# Carried over from phase1a to ensure the reference is reproducible.
_EPS_FD = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-4], dtype=np.float64)

# Off-diagonal mask: exclude diagonal + qop row/col (index 4 — always identity).
_OFF_MASK = np.ones((5, 5), dtype=bool)
_OFF_MASK[np.eye(5, dtype=bool)] = False
_OFF_MASK[4, :] = False
_OFF_MASK[:, 4] = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class A4Report:
    model_name: str
    n_tracks: int

    frob_rel_mean:    float
    frob_rel_median:  float
    frob_rel_p95:     float
    # off_max_frob = max|J_m[i,j]-J_r[i,j]| / ||J_r||_F  (per track, off-diag excl qop)
    off_max_frob_median: float
    off_max_frob_p95:    float

    # Per-element mean rel-err matrix (5×5), useful for diagnosis.
    per_element_rel_err: np.ndarray = field(repr=False)
    # Per-track Frobenius rel-err array for distribution plots.
    frob_rel_per_track: np.ndarray = field(repr=False)

    eval_time_s: float = 0.0

    def passes_gate(self) -> bool:
        return (
            self.frob_rel_mean < GATE_FROB_REL_MEAN
            and self.off_max_frob_p95 < GATE_OFF_MAX_FROB_P95
        )

    def verdict(self) -> str:
        if self.passes_gate():
            return "PASS"
        reasons = []
        if self.frob_rel_mean >= GATE_FROB_REL_MEAN:
            reasons.append(
                f"frob_rel_mean={self.frob_rel_mean:.4f} >= {GATE_FROB_REL_MEAN}"
            )
        if self.off_max_frob_p95 >= GATE_OFF_MAX_FROB_P95:
            reasons.append(
                f"off_max_frob_p95={self.off_max_frob_p95:.4f} >= {GATE_OFF_MAX_FROB_P95}"
            )
        return "FAIL: " + "; ".join(reasons)

    def summary(self) -> str:
        lines = [
            f"A4 report — {self.model_name}  ({self.n_tracks} tracks, {self.eval_time_s:.1f}s)",
            f"  Frobenius rel-err  : mean={self.frob_rel_mean:.4f}  median={self.frob_rel_median:.4f}  p95={self.frob_rel_p95:.4f}  (gate < {GATE_FROB_REL_MEAN})",
            f"  Off-diag / ||J||_F : median={self.off_max_frob_median:.6f}  p95={self.off_max_frob_p95:.6f}  (gate < {GATE_OFF_MAX_FROB_P95})",
            f"  Verdict            : {self.verdict()}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name":           self.model_name,
            "n_tracks":             self.n_tracks,
            "frob_rel_mean":        self.frob_rel_mean,
            "frob_rel_median":      self.frob_rel_median,
            "frob_rel_p95":         self.frob_rel_p95,
            "off_max_frob_median":  self.off_max_frob_median,
            "off_max_frob_p95":     self.off_max_frob_p95,
            "gate_frob":            GATE_FROB_REL_MEAN,
            "gate_off_frob":        GATE_OFF_MAX_FROB_P95,
            "passes": self.passes_gate(),
            "verdict": self.verdict(),
            "eval_time_s": self.eval_time_s,
        }


# ---------------------------------------------------------------------------
# Core Jacobian routines
# ---------------------------------------------------------------------------

def _jac_model_autograd(model: torch.nn.Module, x7: np.ndarray) -> np.ndarray:
    """5×5 model Jacobian wrt state inputs [x,y,tx,ty,qop] via fp64 autograd.

    Casting to fp64 eliminates the fp32 round-off noise floor (~1e-7) that
    made finite-difference comparison unreliable in the original phase1a run.
    The autograd result is exact to double-precision machine epsilon.
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


def _jac_rk4_fd(rk4_integrator: Any, x7: np.ndarray) -> np.ndarray:
    """5×5 RK4 reference Jacobian via central-difference finite differences (fp64).

    ``rk4_integrator`` must expose ``propagate(state5, z0, z1) -> state5_out``
    where all arrays are fp64.
    """
    z0, dz = float(x7[5]), float(x7[6])
    J = np.zeros((5, 5), dtype=np.float64)
    for j in range(5):
        eps_j = _EPS_FD[j] if j != 4 else max(abs(x7[4]) * 1e-4, 1e-8)
        sp = x7[:5].astype(np.float64).copy(); sp[j] += eps_j
        sm = x7[:5].astype(np.float64).copy(); sm[j] -= eps_j
        yp = rk4_integrator.propagate(sp, z0, z0 + dz)
        ym = rk4_integrator.propagate(sm, z0, z0 + dz)
        J[:, j] = (yp - ym) / (2.0 * eps_j)
    return J


# ---------------------------------------------------------------------------
# Reference Jacobian cache
# ---------------------------------------------------------------------------

def load_reference_jacobians(path: Path | str) -> np.ndarray:
    """Load a pre-computed (N, 5, 5) fp64 RK4 reference Jacobian stack."""
    return np.load(str(path)).astype(np.float64)


def build_reference_jacobians(
    rk4_integrator: Any,
    X_a4: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Compute RK4 reference Jacobians from scratch via FD.

    Typically takes ~1-5 seconds for 200 tracks.  Save the result and reuse
    rather than recomputing on every run.
    """
    n = X_a4.shape[0]
    J_ref = np.empty((n, 5, 5), dtype=np.float64)
    t0 = time.time()
    for i, x7 in enumerate(X_a4):
        J_ref[i] = _jac_rk4_fd(rk4_integrator, x7.astype(np.float64))
        if verbose and (i + 1) % 50 == 0:
            print(f"  reference Jacobian {i+1}/{n}  ({time.time()-t0:.1f}s)")
    return J_ref


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_a4(
    model: torch.nn.Module,
    X_a4: np.ndarray,
    J_ref_stack: np.ndarray,
    model_name: str = "model",
    verbose: bool = True,
) -> A4Report:
    """Evaluate the A4 Jacobian gate for *any* gen-3 replacement model.

    Parameters
    ----------
    model       : trained PyTorch model, fp32 or fp64; will be cast to fp64
                  internally for autograd.
    X_a4        : (N, 7) fp64 input states [x,y,tx,ty,qop,z_start,dz].
    J_ref_stack : (N, 5, 5) fp64 RK4 reference Jacobians.
    model_name  : string label for the report.
    verbose     : print progress.

    Returns
    -------
    A4Report with frob/off-diagonal statistics and pass/fail verdict.
    """
    assert X_a4.shape[0] == J_ref_stack.shape[0], "X_a4 and J_ref_stack must have same N"
    n = X_a4.shape[0]

    # Cast model to fp64 once
    model_fp64 = copy.deepcopy(model).double().eval()

    frob_rel  = np.empty(n, dtype=np.float64)
    off_max   = np.empty(n, dtype=np.float64)
    elem_diffs = np.zeros((5, 5), dtype=np.float64)

    t0 = time.time()
    for i in range(n):
        J_m = _jac_model_autograd(model_fp64, X_a4[i].astype(np.float64))
        J_r = J_ref_stack[i]

        # Frobenius relative error
        frob_ref = max(np.linalg.norm(J_r), 1e-30)
        frob_rel[i] = np.linalg.norm(J_m - J_r) / frob_ref

        # Off-diagonal metric: max|J_m[i,j] - J_r[i,j]| / ||J_r||_F  (off-diag, excl qop)
        # Matrix-scale normalisation avoids blowup when J_r[i,j] ~ 0.
        diff_off = np.abs((J_m - J_r)[_OFF_MASK])
        off_max[i] = float(diff_off.max() / frob_ref)

        # Accumulate per-element absolute differences for diagnosis
        elem_diffs += np.abs(J_m - J_r) / np.maximum(np.abs(J_r), 1e-12)

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{model_name}] A4 {i+1}/{n}  frob_rel={frob_rel[:i+1].mean():.4f}")

    elapsed = time.time() - t0
    per_elem = elem_diffs / n

    report = A4Report(
        model_name=model_name,
        n_tracks=n,
        frob_rel_mean=float(frob_rel.mean()),
        frob_rel_median=float(np.median(frob_rel)),
        frob_rel_p95=float(np.quantile(frob_rel, 0.95)),
        off_max_frob_median=float(np.median(off_max)),
        off_max_frob_p95=float(np.quantile(off_max, 0.95)),
        per_element_rel_err=per_elem,
        frob_rel_per_track=frob_rel,
        eval_time_s=elapsed,
    )
    if verbose:
        print(report.summary())
    return report
