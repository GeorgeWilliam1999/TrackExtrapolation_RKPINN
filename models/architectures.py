#!/usr/bin/env python3
"""
Gen-3 neural architectures for LHCb track extrapolation.

This is a faithful port of the gen-2 ``v2_fixes`` recipe (which produced the
best result, 0.125 mm with ``nrk4_small_1step``) adapted to the gen-3 data
conventions mandated by the Allen integration audit:

  * **Fix C1** — Allen ``qop`` convention ``kappa = 1e-6 * qop`` (gen-2 used
    ``qop_MeV * 2.99792458e-4``).  Numerically identical at physical inputs.
  * **Fix C2** — ``dz`` may be signed (backward tracks through the magnet
    are part of the Kalman filter's usage pattern).
  * **Fix H**  — the input layout includes ``z_start`` so the correction
    network knows where on the detector it is being asked to propagate from.

Three families are provided, matching gen-2:

  * :class:`MLP`        — supervised feedforward baseline.
                          **True replacement** candidate (see REPLACEMENT_PLAN.md).
  * :class:`PINN_v2`    — physics-informed network with forward-mode JVP
                          and stochastic MC collocation (Fixes A+B+C+F).
                          **True replacement** candidate.
  * :class:`NeuralRK4`  — classical RK4 integrator of the Lorentz ODE with
                          a small learned RHS residual.
                          **DEPRECATED for deployment as of 2026-05-19**: this is a
                          *hybrid* (evaluates the analytic Lorentz RHS and the
                          field map at inference) and therefore does not satisfy
                          the project goal of *replacing* the RK extrapolator.
                          Retained as a research baseline and as the F2/F3
                          audit reference.  M1 candidates are MLP and PINN_v2
                          — see REPLACEMENT_PLAN.md.

Input layout (7 dims):  ``[x, y, tx, ty, qop, z_start, dz]``
Output layout (5 dims): ``[x_f, y_f, tx_f, ty_f, qop_f]``

``qop_f`` is propagated identically to ``qop`` (energy is conserved in a
static magnetic field); the training loss ignores component 4, so this is
purely a book-keeping output required by the Allen 5-dim Kalman state.

Signed-dz improvements over gen-2
---------------------------------
Gen-2 trained on ``dz in [25, 10000] mm`` (always forward).  Gen-3 trains on
``dz in [-10000, 10000]``.  A 1-step RK4 on a backward 10-m track accumulates
~76 mm truncation error, versus ~1 mm forward (the magnet field profile is
asymmetric about the VELO-end).  Two gen-3-specific architectural changes
help:

  1. The ``NeuralRK4`` correction network receives ``log10(|dz|/100)`` and
     ``sign(dz)`` as explicit features (gen-2 was blind to step size since
     all samples had ``dz in [25, 10k]``).
  2. The ``z`` normalisation for the correction net uses the empirical
     ``input_mean[5], input_std[5]`` of ``z_start`` rather than the
     hardcoded ``(z - 5000)/5000`` used in gen-2.

Both are justified by the Gen-3 data distribution and do not affect Allen
compatibility.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo shared physics utils (<repo>/core)
_UTILS_DIR = Path(__file__).parent.parent / "core"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from magnetic_field import get_field_torch  # noqa: E402


# =============================================================================
# Physical constants
# =============================================================================

# Allen convention: qop = 299.792458 * (q/p)[1/MeV].
# P0.0 FIX (2026-06-11): was 1.0e-6 — x1000 too weak (the old comment matched
# gen-2's legacy pairing 2.998e-4 * q/p[1/MeV], but 2.998e-4 belongs with q/p
# in 1/GeV). Correct: dtheta/ds = 1.0e-3 * qop_allen * B[T] per mm. Must match
# core/rk4_propagator._ALLEN_KAPPA_PREFACTOR (the ground-truth generator).
# Externally validated against the production extrapUTT polynomial (P0.1).
_ALLEN_KAPPA_PREFACTOR: float = 1.0e-3


# =============================================================================
# Activation registry
# =============================================================================

_ACTIVATIONS: Dict[str, type] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def _get_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; available {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# =============================================================================
# Base class
# =============================================================================

class BaseTrackExtrapolator(nn.Module):
    """Common input/output normalisation and parameter bookkeeping.

    Input layout  (gen-3, 7): [x, y, tx, ty, qop, z_start, dz]
    Output layout (gen-3, 5): [x_f, y_f, tx_f, ty_f, qop_f]
    """

    def __init__(self, input_dim: int = 7, output_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("output_mean", torch.zeros(output_dim))
        self.register_buffer("output_std", torch.ones(output_dim))
        self._normalization_set = False

    def set_normalization(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> None:
        self.input_mean = X.mean(dim=0).clone()
        self.input_std = X.std(dim=0).clone() + eps
        self.output_mean = Y.mean(dim=0).clone()
        self.output_std = Y.std(dim=0).clone() + eps
        # Guard against degenerate columns (no-op on gen-3 data, but cheap).
        self.input_std = torch.where(
            self.input_std < eps * 10, torch.ones_like(self.input_std), self.input_std
        )
        self._normalization_set = True

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.output_std + self.output_mean

    def save_normalization(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(
                {
                    "input_mean": self.input_mean.cpu().tolist(),
                    "input_std": self.input_std.cpu().tolist(),
                    "output_mean": self.output_mean.cpu().tolist(),
                    "output_std": self.output_std.cpu().tolist(),
                },
                f,
                indent=2,
            )

    def load_normalization(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            d = json.load(f)
        self.input_mean = torch.tensor(d["input_mean"], dtype=torch.float32)
        self.input_std = torch.tensor(d["input_std"], dtype=torch.float32)
        self.output_mean = torch.tensor(d["output_mean"], dtype=torch.float32)
        self.output_std = torch.tensor(d["output_std"], dtype=torch.float32)
        self._normalization_set = True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# MLP — supervised feedforward baseline
# =============================================================================

class MLP(BaseTrackExtrapolator):
    """Standard feedforward network.  Trained with the inv-output-std MSE only."""

    def __init__(
        self,
        hidden_dims: List[int] = [128, 128],
        activation: str = "silu",
        dropout: float = 0.0,
        engineered_features: bool = False,
    ):
        super().__init__(input_dim=7, output_dim=5)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.dropout_rate = float(dropout)
        self.engineered_features = bool(engineered_features)

        # When engineered_features=True append log10(|dz|/100 + 1e-3) and sign(dz)
        # to the normalised 7-dim input, giving a 9-dim network input.
        net_input_dim = 9 if self.engineered_features else 7

        layers: List[nn.Module] = []
        prev = net_input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, 4))  # learn positions + slopes only
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.normalize_input(x)
        if self.engineered_features:
            dz = x[:, 6:7]  # raw (unnormalised) dz [mm]
            log_step = torch.log10(torch.abs(dz) / 100.0 + 1e-3)  # ~[-3, 2]
            sign_step = torch.sign(dz)
            x_norm = torch.cat([x_norm, log_step, sign_step], dim=1)
        y_norm_4 = self.network(x_norm)
        y4 = y_norm_4 * self.output_std[:4] + self.output_mean[:4]
        qop = x[:, 4:5]  # A5 pass-through
        return torch.cat([y4, qop], dim=1)

    def get_config(self) -> Dict:
        return {
            "model_type": "MLP",
            "hidden_dims": self.hidden_dims,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "engineered_features": self.engineered_features,
            "parameters": self.count_parameters(),
        }


# =============================================================================
# PINN_v2 — physics-informed network (gen-2 Fixes A + B + C + F)
# =============================================================================

class PINN_v2(BaseTrackExtrapolator):
    """Physics-informed network with an IC-preserving envelope + forward-mode
    JVP + stochastic MC collocation.  Gen-3 uses per-sample ``z_start`` for
    the physical field lookup (gen-2 used ``z_start = 0``).
    """

    def __init__(
        self,
        hidden_dims: List[int] = [96, 96],
        activation: str = "tanh",
        dropout: float = 0.0,
        lambda_pde: float = 1.0,
        lambda_ic: float = 0.1,
        n_collocation: int = 2,
        field_model: Optional[nn.Module] = None,
        kick_scaled_head: bool = False,
        pde_scale_mode: str = "legacy",
        pde_ref_length: float = 5213.0,
    ):
        super().__init__(input_dim=7, output_dim=5)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.dropout_rate = float(dropout)
        self.lambda_pde = float(lambda_pde)
        self.lambda_ic = float(lambda_ic)
        self.n_collocation = int(n_collocation)
        # --- 2026-06-08 model-improvement flags (default off == locked candidate) ---
        self.kick_scaled_head = bool(kick_scaled_head)   # couple correction to qop*dz magnet kick (q/p-bias fix)
        self.pde_scale_mode = str(pde_scale_mode)        # "legacy" | "fixed_L"
        self.pde_ref_length = float(pde_ref_length)      # reference length for fixed_L PDE scaling [mm]

        self.field = field_model if field_model is not None else get_field_torch(
            use_interpolated=True, polarity=-1
        )

        # Encoder input: 5 state features + 1 z_frac  =>  6-dim.
        enc_layers: List[nn.Module] = []
        prev = 6
        for dim in self.hidden_dims:
            enc_layers.append(nn.Linear(prev, dim))
            enc_layers.append(_get_activation(activation))
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = dim
        self.encoder = nn.Sequential(*enc_layers)
        self.correction_head = nn.Linear(prev, 4)
        self._init_weights()
        if self.kick_scaled_head:
            # Per-channel learnable gain (init exp(0)=1) so the optimiser calibrates
            # the kick magnitude. The correction is multiplied by the qop-scaled kick
            # (kappa = qop * c, NO field lookup), so the network learns only a
            # dimensionless O(1) shape and the q/p magnitude is exact by construction.
            self.kick_loggain = nn.Parameter(torch.zeros(4))

    def _init_weights(self) -> None:
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.uniform_(self.correction_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.correction_head.bias)

    def forward_at_z(
        self, x0: torch.Tensor, z_frac: torch.Tensor, dz: torch.Tensor,
    ) -> torch.Tensor:
        B = x0.shape[0]
        if z_frac.dim() == 0:
            z_frac = z_frac.view(1, 1).expand(B, 1)
        elif z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(1)
        if z_frac.shape[0] == 1 and B > 1:
            z_frac = z_frac.expand(B, 1)

        x0_norm = (x0 - self.input_mean[:5]) / self.input_std[:5]
        enc_in = torch.cat([x0_norm, z_frac], dim=1)
        features = self.encoder(enc_in)
        corr = self.correction_head(features)

        x0p, y0p, tx0, ty0 = x0[:, 0], x0[:, 1], x0[:, 2], x0[:, 3]
        zf = z_frac.squeeze(1)
        delta_z = zf * dz

        if self.kick_scaled_head:
            # Couple the correction to the leading-order magnet kick. kappa = qop * c
            # (Allen units) carries the full q/p dependence, so the network only has to
            # learn a dimensionless O(1) shape; the q/p magnitude is built in. No field
            # lookup -> still a true replacement. IC preserved: every term carries zf.
            qop = x0[:, 4]
            kappa_dz = (_ALLEN_KAPPA_PREFACTOR * qop) * dz       # slope-kick scale  [B]
            g = torch.exp(self.kick_loggain)                     # per-channel gain  [4]
            tx_out = tx0 + zf * g[0] * kappa_dz * corr[:, 0]
            ty_out = ty0 + zf * g[1] * kappa_dz * corr[:, 1]
            x_out  = x0p + tx0 * delta_z + g[2] * kappa_dz * delta_z * corr[:, 2]
            y_out  = y0p + ty0 * delta_z + g[3] * kappa_dz * delta_z * corr[:, 3]
        else:
            x_out  = x0p + tx0 * delta_z + corr[:, 2] * zf * dz
            y_out  = y0p + ty0 * delta_z + corr[:, 3] * zf * dz
            tx_out = tx0 + corr[:, 0] * zf
            ty_out = ty0 + corr[:, 1] * zf
        return torch.stack([x_out, y_out, tx_out, ty_out], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, :5]
        dz = x[:, 6]
        z_frac = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        y4 = self.forward_at_z(x0, z_frac, dz=dz)
        return torch.cat([y4, x[:, 4:5]], dim=1)

    def compute_physics_loss(
        self, x: torch.Tensor, y_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B = x.shape[0]

        x0 = x[:, :5]
        qop = x[:, 4]
        z_start = x[:, 5]
        dz = x[:, 6]
        initial_state = x[:, :4]
        inv_output_std = 1.0 / self.output_std[:4]

        # IC loss (identically zero by envelope; kept as a sanity term).
        z_zero = torch.zeros((B, 1), device=device, dtype=dtype)
        y_at_z0 = self.forward_at_z(x0, z_zero, dz=dz)
        ic_loss = ((y_at_z0 - initial_state) * inv_output_std).pow(2).mean()

        n_mc = max(1, self.n_collocation)
        if n_mc > 1:
            x0_rep = x0.unsqueeze(1).expand(-1, n_mc, -1).reshape(-1, 5)
            dz_rep = dz.unsqueeze(1).expand(-1, n_mc).reshape(-1)
            qop_rep = qop.unsqueeze(1).expand(-1, n_mc).reshape(-1)
            zst_rep = z_start.unsqueeze(1).expand(-1, n_mc).reshape(-1)
        else:
            x0_rep, dz_rep, qop_rep, zst_rep = x0, dz, qop, z_start
        n_total = x0_rep.shape[0]

        z_flat = torch.rand(n_total, 1, device=device, dtype=dtype)

        def _fwd(z: torch.Tensor) -> torch.Tensor:
            return self.forward_at_z(x0_rep, z, dz=dz_rep)

        tangent = torch.ones_like(z_flat)
        try:
            y_c, dy_dz_frac = torch.func.jvp(_fwd, (z_flat,), (tangent,))
        except Exception:
            z_in = z_flat.detach().clone().requires_grad_(True)
            y_c = self.forward_at_z(x0_rep, z_in, dz=dz_rep)
            dy_list = []
            for i in range(4):
                g = torch.autograd.grad(
                    y_c[:, i].sum(), z_in, create_graph=True, retain_graph=True,
                )[0]
                dy_list.append(g.squeeze(-1))
            dy_dz_frac = torch.stack(dy_list, dim=1)

        z_phys = zst_rep + z_flat.squeeze(1) * dz_rep
        Bx, By, Bz = self.field(y_c[:, 0], y_c[:, 1], z_phys)
        tx_p, ty_p = y_c[:, 2], y_c[:, 3]
        sqrt_term = torch.sqrt(1 + tx_p**2 + ty_p**2)
        kappa_rep = _ALLEN_KAPPA_PREFACTOR * qop_rep

        dx_exp  = tx_p
        dy_exp  = ty_p
        dtx_exp = kappa_rep * sqrt_term * (tx_p * ty_p * Bx - (1 + tx_p**2) * By + ty_p * Bz)
        dty_exp = kappa_rep * sqrt_term * ((1 + ty_p**2) * Bx - tx_p * ty_p * By - tx_p * Bz)

        dz_abs_safe = dz_rep.abs().clamp(min=25.0)
        # Signed divide preserves direction; abs for positive scale.
        dy_dz_phys = dy_dz_frac / torch.where(
            dz_rep.abs() < 25.0, torch.sign(dz_rep) * 25.0, dz_rep
        ).unsqueeze(1)
        if self.pde_scale_mode == "fixed_L":
            # Non-dimensionalise the derivative residual by a *fixed* reference length
            # rather than per-sample |dz|. The legacy sigma/|dz| scaling amplifies
            # large-|dz| tracks ~linearly, over-weighting exactly the high-bend / heavy-
            # tail population implicated in the q/p residual bias (gen3_m1 audit). A
            # fixed L makes every track's residual comparable.
            pde_scale = (self.output_std[:4] / float(self.pde_ref_length)).unsqueeze(0)
            pde_scale = pde_scale.expand(dz_abs_safe.shape[0], -1)
        else:
            pde_scale = self.output_std[:4].unsqueeze(0) / dz_abs_safe.unsqueeze(1)

        r_x  = ((dy_dz_phys[:, 0] - dx_exp ) / pde_scale[:, 0]).pow(2)
        r_y  = ((dy_dz_phys[:, 1] - dy_exp ) / pde_scale[:, 1]).pow(2)
        r_tx = ((dy_dz_phys[:, 2] - dtx_exp) / pde_scale[:, 2]).pow(2)
        r_ty = ((dy_dz_phys[:, 3] - dty_exp) / pde_scale[:, 3]).pow(2)
        pde_loss = (r_x + r_y + r_tx + r_ty).mean()

        if torch.isnan(pde_loss) or torch.isinf(pde_loss):
            pde_loss = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        if torch.isnan(ic_loss) or torch.isinf(ic_loss):
            ic_loss = torch.zeros((), device=device, dtype=dtype, requires_grad=True)

        return {
            "ic": self.lambda_ic * ic_loss,
            "pde": self.lambda_pde * pde_loss,
        }

    def get_config(self) -> Dict:
        return {
            "model_type": "PINN_v2",
            "hidden_dims": self.hidden_dims,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "lambda_pde": self.lambda_pde,
            "lambda_ic": self.lambda_ic,
            "n_collocation": self.n_collocation,
            "kick_scaled_head": self.kick_scaled_head,
            "pde_scale_mode": self.pde_scale_mode,
            "pde_ref_length": self.pde_ref_length,
            "fixes_applied": ["A_jvp", "B_stochastic_colloc", "C_z_dependent_trunk", "H_z_start"]
                + (["K_kick_scaled_head"] if self.kick_scaled_head else [])
                + (["L_fixed_pde_scale"] if self.pde_scale_mode == "fixed_L" else []),
            "parameters": self.count_parameters(),
        }


# =============================================================================
# NeuralRK4 — classical RK4 of Lorentz ODE + small learned residual
# =============================================================================

class NeuralRK4(BaseTrackExtrapolator):
    """RK4 integrator of the Lorentz ODE with a small learned RHS correction.

    .. deprecated:: 2026-05-19

        ``NeuralRK4`` is a **hybrid** (classical RK4 + learned RHS residual)
        and does **not** satisfy the project goal of *replacing* the RK
        extrapolator: it evaluates the analytic Lorentz RHS and the magnetic
        field map at inference time.  See ``REPLACEMENT_PLAN.md`` §2 and
        ``CLEANUP_LIST.md`` §3.1.

        The class is retained as a research baseline and as the F2/F3 audit
        reference; no new training runs or configs should target it.
        Production candidates are :class:`MLP` and :class:`PINN_v2`.

    Gen-3 correction net input (8 features):
        [x_n, y_n, tx_n, ty_n, z_n, qop_n, log10(|dz|/100), sign(dz)]

    The last two are gen-3 additions: since dz is signed and spans
    [-10k, +10k] the network must know the size & direction of the step.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 64],
        activation: str = "tanh",
        n_rk_steps: int = 1,
        correction_scale_init: float = 1e-3,
        field_model: Optional[nn.Module] = None,
        disable_correction: bool = False,
    ):
        super().__init__(input_dim=7, output_dim=5)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.n_rk_steps = int(n_rk_steps)
        # Action N (ADR 0002): when True, the learned RHS correction is
        # skipped — the forward pass is pure classical RK4 on the Lorentz ODE.
        # The corrector weights remain in the state dict (so checkpoints
        # round-trip cleanly) but never contribute to the output.
        self.disable_correction = bool(disable_correction)

        self.field = field_model if field_model is not None else get_field_torch(
            use_interpolated=True, polarity=-1
        )

        layers: List[nn.Module] = []
        prev = 8
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(_get_activation(activation))
            prev = dim
        layers.append(nn.Linear(prev, 4))
        self.correction_net = nn.Sequential(*layers)

        self.log_corr_scale = nn.Parameter(
            torch.tensor(float(np.log(correction_scale_init)))
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _lorentz_rhs(
        self, y: torch.Tensor, qop: torch.Tensor, z_phys: torch.Tensor,
    ) -> torch.Tensor:
        x_p, y_p, tx, ty = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        Bx, By, Bz = self.field(x_p, y_p, z_phys)
        kappa = _ALLEN_KAPPA_PREFACTOR * qop
        N = torch.sqrt(1 + tx * tx + ty * ty)
        dxdz  = tx
        dydz  = ty
        dtxdz = kappa * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz)
        dtydz = kappa * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz)
        return torch.stack([dxdz, dydz, dtxdz, dtydz], dim=1)

    def _neural_correction(
        self, y: torch.Tensor, qop: torch.Tensor,
        z_phys: torch.Tensor, dz_signed: torch.Tensor,
    ) -> torch.Tensor:
        y_norm = (y - self.output_mean[:4]) / self.output_std[:4]
        z_norm = (z_phys - self.input_mean[5]) / self.input_std[5]
        qop_norm = (qop - self.input_mean[4]) / self.input_std[4]

        dz_abs = dz_signed.abs().clamp(min=25.0)
        log_dz = torch.log10(dz_abs / 100.0)
        sign_dz = torch.sign(dz_signed)

        feat = torch.stack(
            [
                y_norm[:, 0], y_norm[:, 1], y_norm[:, 2], y_norm[:, 3],
                z_norm, qop_norm, log_dz, sign_dz,
            ],
            dim=1,
        )
        raw = self.correction_net(feat)
        scale = self.output_std[:4].unsqueeze(0) / dz_abs.unsqueeze(1)
        return torch.exp(self.log_corr_scale) * scale * raw

    def _rhs(
        self, y: torch.Tensor, qop: torch.Tensor,
        z_phys: torch.Tensor, dz_signed: torch.Tensor,
    ) -> torch.Tensor:
        lorentz = self._lorentz_rhs(y, qop, z_phys)
        if self.disable_correction:
            return lorentz
        return lorentz + self._neural_correction(y, qop, z_phys, dz_signed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, :4].clone()
        qop = x[:, 4]
        z_start = x[:, 5]
        dz_full = x[:, 6]

        n = max(1, self.n_rk_steps)
        h = dz_full / n
        h_col = h.unsqueeze(1)
        z = z_start.clone()

        for _ in range(n):
            k1 = self._rhs(y, qop, z, dz_full)
            k2 = self._rhs(y + 0.5 * h_col * k1, qop, z + 0.5 * h, dz_full)
            k3 = self._rhs(y + 0.5 * h_col * k2, qop, z + 0.5 * h, dz_full)
            k4 = self._rhs(y + h_col * k3, qop, z + h, dz_full)
            y = y + (h_col / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z = z + h

        return torch.cat([y, qop.unsqueeze(1)], dim=1)

    def compute_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        y = self(x)
        B = x.shape[0]
        jac = torch.zeros(B, 5, 5, device=x.device, dtype=x.dtype)
        for i in range(5):
            grad = torch.autograd.grad(
                y[:, i].sum(), x, create_graph=False, retain_graph=(i < 4),
            )[0]
            jac[:, i, :] = grad[:, :5]
        return jac

    def get_config(self) -> Dict:
        fixes = [
            "C1_allen_qop", "C2_signed_dz", "H_z_start", "A5_qop_output",
            "D_real_rk4", "K_dz_aware_correction",
        ]
        if self.disable_correction:
            fixes.append("N_corrector_disabled")
        return {
            "model_type": "NeuralRK4",
            "hidden_dims": self.hidden_dims,
            "activation": self.activation_name,
            "n_rk_steps": self.n_rk_steps,
            "correction_scale": float(torch.exp(self.log_corr_scale).item()),
            "disable_correction": self.disable_correction,
            "fixes_applied": fixes,
            "parameters": self.count_parameters(),
        }


# =============================================================================
# Registry
# =============================================================================

MODEL_REGISTRY = {
    "mlp": MLP,
    "pinn_v2": PINN_v2,
    "neural_rk4": NeuralRK4,
}


def create_model(model_type: str, **kwargs) -> BaseTrackExtrapolator:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}; available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)
