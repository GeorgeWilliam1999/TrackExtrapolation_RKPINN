#!/usr/bin/env python3
"""
================================================================================
V5 Unified Training Script
================================================================================

Trains all 5 V5 model architectures from JSON config files:

  1. MLP         — Standard MLP baseline (6→hidden→4)
  2. Quadratic   — IC + z·c₁ + z²·c₂ polynomial residual
  3. ZFrac       — z_frac as 7th input with IC residual connection
  4. PDE         — True physics-informed loss via autograd + Lorentz force
  5. Compositional — Chain N short-step predictions

The PDE-residual model supports two modes:
  - "pure":   Uses MLP data (X,Y only), physics loss from autograd
  - "hybrid": Uses PINN data (X,Y,z_frac,Y_col), supervised + physics loss

Usage:
    python train_v5.py --config configs/mlp_v5_2L_1024_512.json
    python train_v5.py --config configs/pde_pure_v5_2L_1024_512.json

Author: G. Scriven
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys
import warnings

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# =============================================================================
# Physical Constants
# =============================================================================

# Speed of light factor: converts (q/p in 1/MeV) × (B in Tesla) → curvature in 1/mm
C_LIGHT = 2.99792458e-4


# =============================================================================
# Magnetic Field Models (self-contained copy from V3/utils/magnetic_field.py)
# =============================================================================
# These are needed for the PDE-residual physics loss (Option 3).
# Copied inline to keep V5 self-contained with no V3 dependencies.

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_FIELD_MAP = _BASE_DIR / 'field_maps' / 'twodip.rtf'


class GaussianFieldTorch(nn.Module):
    """Gaussian approximation of LHCb magnetic field (differentiable)."""

    def __init__(self, polarity: int = 1):
        super().__init__()
        self.register_buffer('B0', torch.tensor(polarity * -1.0182, dtype=torch.float32))
        self.register_buffer('z_center', torch.tensor(5007.0, dtype=torch.float32))
        self.register_buffer('z_width', torch.tensor(1744.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_rel = (z - self.z_center) / self.z_width
        By = self.B0 * torch.exp(-0.5 * z_rel ** 2)
        Bx = torch.zeros_like(By)
        Bz = torch.zeros_like(By)
        return Bx, By, Bz


class InterpolatedFieldTorch(nn.Module):
    """
    Trilinear interpolation of real LHCb field map (differentiable).
    Uses torch.nn.functional.grid_sample for GPU-accelerated interpolation.
    """

    def __init__(self, field_map_path: Optional[str] = None,
                 polarity: int = 1, device: str = 'cpu'):
        super().__init__()
        self.polarity = polarity
        path = Path(field_map_path) if field_map_path else _DEFAULT_FIELD_MAP
        if not path.exists():
            raise FileNotFoundError(f"Field map not found: {path}")
        self._load_field_map(path, device)

    def _load_field_map(self, path: Path, device: str) -> None:
        data = np.loadtxt(path)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        Bx, By, Bz = data[:, 3], data[:, 4], data[:, 5]

        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))
        z_unique = np.sort(np.unique(z))
        nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

        self.register_buffer('x_min', torch.tensor(x_unique[0], dtype=torch.float32))
        self.register_buffer('x_max', torch.tensor(x_unique[-1], dtype=torch.float32))
        self.register_buffer('y_min', torch.tensor(y_unique[0], dtype=torch.float32))
        self.register_buffer('y_max', torch.tensor(y_unique[-1], dtype=torch.float32))
        self.register_buffer('z_min', torch.tensor(z_unique[0], dtype=torch.float32))
        self.register_buffer('z_max', torch.tensor(z_unique[-1], dtype=torch.float32))

        Bx_grid = Bx.reshape(nz, nx, ny).transpose(1, 2, 0)
        By_grid = By.reshape(nz, nx, ny).transpose(1, 2, 0)
        Bz_grid = Bz.reshape(nz, nx, ny).transpose(1, 2, 0)

        B_grid = np.stack([Bx_grid, By_grid, Bz_grid], axis=0)  # [3, nx, ny, nz]
        B_grid = np.transpose(B_grid, (0, 3, 2, 1))              # [3, nz, ny, nx]
        B_grid = B_grid[np.newaxis, ...]                          # [1, 3, nz, ny, nx]

        self.register_buffer('B_grid', torch.tensor(B_grid, dtype=torch.float32))
        self.grid_shape = (nx, ny, nz)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        z_norm = 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0

        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        y_norm = torch.clamp(y_norm, -1.0, 1.0)
        z_norm = torch.clamp(z_norm, -1.0, 1.0)

        original_shape = x.shape
        n_points = x.numel()

        grid = torch.stack([x_norm.flatten(), y_norm.flatten(), z_norm.flatten()], dim=-1)
        grid = grid.view(1, 1, 1, n_points, 3)

        B_interp = torch.nn.functional.grid_sample(
            self.B_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
        )

        B_interp = B_interp.view(3, n_points).T
        Bx = B_interp[:, 0].view(original_shape) * self.polarity
        By = B_interp[:, 1].view(original_shape) * self.polarity
        Bz = B_interp[:, 2].view(original_shape)
        return Bx, By, Bz


def get_field_torch(field_map_path: Optional[str] = None,
                    polarity: int = 1,
                    device: str = 'cpu') -> nn.Module:
    """Factory: returns InterpolatedFieldTorch if field map exists, else Gaussian."""
    path = Path(field_map_path) if field_map_path else _DEFAULT_FIELD_MAP
    if path.exists():
        print(f"  Loading interpolated field map from {path}")
        return InterpolatedFieldTorch(str(path), polarity=polarity, device=device).to(device)
    else:
        warnings.warn(f"Field map not found at {path}. Falling back to Gaussian approximation.")
        return GaussianFieldTorch(polarity=polarity).to(device)


# =============================================================================
# V5 Architecture Definitions
# =============================================================================

class MLPV5(nn.Module):
    """Standard MLP baseline for track extrapolation.  Input: [x, y, tx, ty, qop, dz] → [x, y, tx, ty]."""

    def __init__(self, hidden_dims: List[int], activation: str = 'silu', **kwargs):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())

        layers = []
        in_dim = 6
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.network = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state_dz, z_frac=None):
        """z_frac is ignored — MLP only predicts endpoint."""
        return self.network(state_dz)


class QuadraticResidual(nn.Module):
    """
    Output = IC + z_frac × c₁ + z_frac² × c₂

    Guarantees IC at z_frac=0. The z² term captures parabolic position
    trajectories arising from integration of linearly-varying slopes.
    """

    def __init__(self, hidden_dims: List[int], activation: str = 'silu', **kwargs):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())

        layers = []
        in_dim = 6
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.head_linear = nn.Linear(in_dim, 4)
        self.head_quadratic = nn.Linear(in_dim, 4)

        for head in [self.head_linear, self.head_quadratic]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def forward(self, state_dz, z_frac=None):
        initial = state_dz[:, :4]
        features = self.backbone(state_dz)
        c1 = self.head_linear(features)
        c2 = self.head_quadratic(features)

        if z_frac is None:
            return initial + c1 + c2

        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)

        if z_frac.size(-1) == 1:
            return initial + z_frac * c1 + z_frac ** 2 * c2
        else:
            zf = z_frac.unsqueeze(-1)           # [B, N_col, 1]
            return (initial.unsqueeze(1) +
                    zf * c1.unsqueeze(1) +
                    zf ** 2 * c2.unsqueeze(1))


class PINNZFracInput(nn.Module):
    """
    z_frac as 7th input with residual IC guarantee.
    Output = IC + z_frac × network(state, dz, z_frac)
    """

    def __init__(self, hidden_dims: List[int], activation: str = 'silu', **kwargs):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())

        layers = []
        in_dim = 7  # 6 + z_frac
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.core = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.core[-1].weight, gain=0.1)
        nn.init.zeros_(self.core[-1].bias)

    def forward(self, state_dz, z_frac=None):
        B = state_dz.size(0)
        initial = state_dz[:, :4]

        if z_frac is None:
            z_frac = torch.ones(B, 1, device=state_dz.device)
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)

        if z_frac.size(-1) == 1:
            inp = torch.cat([state_dz, z_frac], dim=-1)
            correction = self.core(inp)
            return initial + z_frac * correction
        else:
            N_col = z_frac.size(-1)
            state_exp = state_dz.unsqueeze(1).expand(-1, N_col, -1)
            zf_exp = z_frac.unsqueeze(-1)   # [B, N_col, 1]
            inp = torch.cat([state_exp, zf_exp], dim=-1)
            correction = self.core(inp)
            return initial.unsqueeze(1) + zf_exp * correction


class PDEResidualPINN(nn.Module):
    """
    Architecture identical to PINNZFracInput (7 inputs, IC residual connection).
    The novelty is in the LOSS function: uses autograd to compute ds/dz and
    compares against the Lorentz force ODE, with the real magnetic field.

    Supports two loss modes (controlled by config, not by this class):
      - "pure":   Endpoint MSE + IC + PDE residual (no supervised collocation)
      - "hybrid": All of the above + supervised collocation MSE
    """

    def __init__(self, hidden_dims: List[int], activation: str = 'silu', **kwargs):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())

        layers = []
        in_dim = 7  # [x0, y0, tx0, ty0, qop, dz, z_frac]
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.core = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.core[-1].weight, gain=0.1)
        nn.init.zeros_(self.core[-1].bias)

    def forward(self, state_dz, z_frac=None):
        """Same forward as PINNZFracInput — IC + z_frac × correction."""
        B = state_dz.size(0)
        initial = state_dz[:, :4]

        if z_frac is None:
            z_frac = torch.ones(B, 1, device=state_dz.device)
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)

        if z_frac.size(-1) == 1:
            inp = torch.cat([state_dz, z_frac], dim=-1)
            correction = self.core(inp)
            return initial + z_frac * correction
        else:
            N_col = z_frac.size(-1)
            state_exp = state_dz.unsqueeze(1).expand(-1, N_col, -1)
            zf_exp = z_frac.unsqueeze(-1)
            inp = torch.cat([state_exp, zf_exp], dim=-1)
            correction = self.core(inp)
            return initial.unsqueeze(1) + zf_exp * correction


class CompositionalPINN(nn.Module):
    """
    Compositional / recurrent architecture: chains N short-step predictions.

    Each step: given current state [x, y, tx, ty, qop, dz_sub],
    predict next state [x', y', tx', ty'].

    The total extrapolation dz is divided into N sub-steps of dz/N each.
    The model learns to extrapolate over short distances and composes
    these to cover the full variable-dz range.

    At inference with z_frac, the chain is evaluated up to the
    appropriate number of full steps + one partial step.
    """

    def __init__(self, hidden_dims: List[int], activation: str = 'silu',
                 n_steps: int = 8, **kwargs):
        super().__init__()
        self.n_steps = n_steps

        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())

        # Single step model: [x, y, tx, ty, qop, dz_sub] → Δ[x, y, tx, ty]
        layers = []
        in_dim = 6
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.step_model = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.step_model[-1].weight, gain=0.1)
        nn.init.zeros_(self.step_model[-1].bias)

    def _single_step(self, state: torch.Tensor, qop: torch.Tensor,
                     dz_sub: torch.Tensor) -> torch.Tensor:
        """
        One step of the chain: state [B, 4] + qop [B, 1] + dz_sub [B, 1] → new state [B, 4].
        Uses residual connection: state_new = state + model([state, qop, dz_sub]).
        """
        inp = torch.cat([state, qop, dz_sub], dim=-1)  # [B, 6]
        delta = self.step_model(inp)                     # [B, 4]
        return state + delta

    def forward(self, state_dz: torch.Tensor, z_frac=None) -> torch.Tensor:
        """
        Forward pass. Chains N steps.

        Args:
            state_dz: [B, 6] = [x, y, tx, ty, qop, dz]
            z_frac:   None (endpoint), scalar [B] or [B, N_col]

        Returns:
            Predicted state(s) at z_frac position(s).
        """
        B = state_dz.size(0)
        state = state_dz[:, :4]                        # [B, 4]
        qop = state_dz[:, 4:5]                         # [B, 1]
        dz_total = state_dz[:, 5:6]                    # [B, 1]
        dz_sub = dz_total / self.n_steps               # [B, 1]

        if z_frac is None:
            # Endpoint: chain all N steps
            for _ in range(self.n_steps):
                state = self._single_step(state, qop, dz_sub)
            return state

        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)              # [B, 1]

        if z_frac.size(-1) == 1:
            return self._forward_at_zfrac(state, qop, dz_sub, z_frac.squeeze(-1))
        else:
            # Multiple collocation points: [B, N_col]
            results = []
            for col_idx in range(z_frac.size(-1)):
                zf = z_frac[:, col_idx]                # [B]
                result = self._forward_at_zfrac(state.clone(), qop, dz_sub, zf)
                results.append(result)
            return torch.stack(results, dim=1)         # [B, N_col, 4]

    def _forward_at_zfrac(self, initial_state: torch.Tensor, qop: torch.Tensor,
                          dz_sub: torch.Tensor, z_frac: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the chain at a specific z_frac ∈ [0, 1].

        Strategy: z_frac * N gives the number of continuous steps.
        We take floor(z_frac * N) full steps, then one partial step
        with dz_partial = (z_frac * N - floor(z_frac * N)) * dz_sub.
        """
        N = self.n_steps
        continuous_steps = z_frac * N                  # [B]
        n_full = torch.floor(continuous_steps).long()  # [B]
        frac_partial = continuous_steps - n_full.float()  # [B]

        # We need to handle variable n_full per sample.
        # For efficiency, iterate up to max steps and mask.
        max_steps = min(int(n_full.max().item()), N)
        state = initial_state.clone()

        for step in range(max_steps):
            mask = (n_full > step).float().unsqueeze(-1)  # [B, 1]
            new_state = self._single_step(state, qop, dz_sub)
            state = mask * new_state + (1 - mask) * state

        # Partial step
        dz_partial = frac_partial.unsqueeze(-1) * dz_sub  # [B, 1]
        has_partial = (frac_partial > 1e-6).float().unsqueeze(-1)
        partial_state = self._single_step(state, qop, dz_partial)
        state = has_partial * partial_state + (1 - has_partial) * state

        return state


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    'mlp': MLPV5,
    'quadratic': QuadraticResidual,
    'zfrac': PINNZFracInput,
    'pde': PDEResidualPINN,
    'compositional': CompositionalPINN,
}


# =============================================================================
# Dataset
# =============================================================================

class V5Dataset(Dataset):
    """
    Unified dataset for all V5 model types.

    For MLP / Compositional (no collocation): X, Y
    For PINN (Quadratic, ZFrac, PDE-hybrid): X, Y, z_frac, Y_col
    For PDE-pure: X, Y (z_frac sampled on-the-fly during loss computation)

    Always stores X_raw (unnormalized) alongside normalized X for
    PDE-residual physics loss (needs physical coordinates for B-field evaluation).
    """

    def __init__(self, X, Y, z_frac=None, Y_col=None, normalize=True):
        self.X_raw = torch.from_numpy(X).float()       # Always keep raw copy
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.has_collocation = z_frac is not None and Y_col is not None

        if self.has_collocation:
            self.z_frac = torch.from_numpy(z_frac).float()
            self.Y_col = torch.from_numpy(Y_col).float()

        # Normalization
        if normalize:
            self.input_mean = self.X.mean(dim=0)
            self.input_std = self.X.std(dim=0)
            self.input_std[self.input_std < 1e-6] = 1.0

            self.output_mean = self.Y.mean(dim=0)
            self.output_std = self.Y.std(dim=0)
            self.output_std[self.output_std < 1e-6] = 1.0

            self.X = (self.X - self.input_mean) / self.input_std
            self.Y = (self.Y - self.output_mean) / self.output_std
            if self.has_collocation:
                self.Y_col = (self.Y_col - self.output_mean) / self.output_std
        else:
            self.input_mean = torch.zeros(6)
            self.input_std = torch.ones(6)
            self.output_mean = torch.zeros(4)
            self.output_std = torch.ones(4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_collocation:
            return self.X[idx], self.Y[idx], self.z_frac[idx], self.Y_col[idx], self.X_raw[idx]
        else:
            return self.X[idx], self.Y[idx], self.X_raw[idx]

    def get_norm_stats(self):
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
        }


# =============================================================================
# Loss Functions
# =============================================================================

def mlp_loss(model, X, Y, **kwargs):
    """Standard MSE loss for MLP / Compositional (endpoint only)."""
    pred = model(X)
    loss = F.mse_loss(pred, Y)
    return loss, {'total': loss.item()}


def pinn_loss(model, X, Y, z_frac, Y_col,
              lambda_ic=10.0, lambda_endpoint=1.0, lambda_collocation=1.0, **kwargs):
    """PINN loss with IC, endpoint, and supervised collocation (for Quadratic, ZFrac)."""
    B = X.size(0)

    # IC loss at z_frac=0
    z0 = torch.zeros(B, device=X.device)
    pred_ic = model(X, z0)
    target_ic = X[:, :4]
    loss_ic = F.mse_loss(pred_ic, target_ic)

    # Endpoint loss at z_frac=1
    z1 = torch.ones(B, device=X.device)
    pred_end = model(X, z1)
    loss_end = F.mse_loss(pred_end, Y)

    # Supervised collocation loss
    pred_col = model(X, z_frac)
    loss_col = F.mse_loss(pred_col, Y_col)

    total = lambda_ic * loss_ic + lambda_endpoint * loss_end + lambda_collocation * loss_col

    return total, {
        'total': total.item(),
        'ic': loss_ic.item(),
        'endpoint': loss_end.item(),
        'collocation': loss_col.item(),
    }


def pde_loss_pure(model, X, Y, X_raw, norm_stats, field_model,
                  lambda_ic=10.0, lambda_endpoint=1.0, lambda_pde=1.0,
                  n_pde_points=8, **kwargs):
    """
    PDE-residual loss (pure mode): endpoint MSE + IC + physics ODE residual.

    No supervised collocation — the physics loss comes from comparing
    autograd derivatives d(state)/d(z_frac) against the Lorentz force ODE
    evaluated at the predicted positions using the real magnetic field.

    The z_frac values are sampled uniformly in [0.05, 0.95] per batch.

    Physics ODE (z-parameterized):
        dx/dz  = tx
        dy/dz  = ty
        dtx/dz = κ·N·[tx·ty·Bx − (1+tx²)·By + ty·Bz]
        dty/dz = κ·N·[(1+ty²)·Bx − tx·ty·By − tx·Bz]

    where κ = C_LIGHT · qop,  N = sqrt(1 + tx² + ty²)
    """
    device = X.device
    B = X.size(0)

    # ---- IC loss ----
    z0 = torch.zeros(B, device=device)
    pred_ic = model(X, z0)
    target_ic = X[:, :4]
    loss_ic = F.mse_loss(pred_ic, target_ic)

    # ---- Endpoint loss ----
    z1 = torch.ones(B, device=device)
    pred_end = model(X, z1)
    loss_end = F.mse_loss(pred_end, Y)

    # ---- PDE residual loss ----
    # Sample z_frac values for PDE evaluation (requires grad for autograd)
    z_frac_pde = torch.rand(B, n_pde_points, device=device) * 0.9 + 0.05  # [0.05, 0.95]
    z_frac_pde.requires_grad_(True)

    # Forward at PDE collocation points: [B, n_pde_points, 4]
    pred_pde = model(X, z_frac_pde)  # [B, n_pde_points, 4]

    # Compute d(pred)/d(z_frac) via autograd
    # pred_pde has shape [B, n_pde_points, 4], z_frac_pde has shape [B, n_pde_points]
    # We need d(pred_pde[:,:,i]) / d(z_frac_pde) for each output i
    dpred_dzfrac = torch.zeros_like(pred_pde)  # [B, n_pde_points, 4]
    for i in range(4):
        grad_outputs = torch.ones(B, n_pde_points, device=device)
        grads = torch.autograd.grad(
            pred_pde[:, :, i], z_frac_pde,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]  # [B, n_pde_points]
        dpred_dzfrac[:, :, i] = grads

    # Convert from d/d(z_frac) to d/dz (physical)
    # z = z_start + z_frac * dz, so dz_frac/dz = 1/dz, thus d/dz = (1/dz) * d/d(z_frac)
    # But predictions are in normalized output space — we need to denormalize first.
    #
    # pred_physical = pred_normalized * output_std + output_mean
    # d(pred_physical)/dz = output_std * d(pred_normalized)/d(z_frac) * (1/dz_physical)
    #
    # Get physical dz from raw inputs (X_raw[:, 5] = dz in mm)
    dz_phys = X_raw[:, 5].unsqueeze(-1)  # [B, 1]

    output_std = norm_stats['output_std'].to(device)  # [4]
    output_mean = norm_stats['output_mean'].to(device)  # [4]
    input_std = norm_stats['input_std'].to(device)  # [6]
    input_mean = norm_stats['input_mean'].to(device)  # [6]

    # Physical derivatives: d(state_physical)/dz
    # dpred_dzfrac is d(normalized_pred)/d(z_frac)
    # d(physical)/dz = output_std * dpred_dzfrac / dz_phys
    dstate_dz = output_std.unsqueeze(0).unsqueeze(0) * dpred_dzfrac / dz_phys.unsqueeze(-1)
    # dstate_dz: [B, n_pde_points, 4] = [dx/dz, dy/dz, dtx/dz, dty/dz] in physical units

    # ---- Get predicted physical state at PDE points ----
    pred_phys = pred_pde * output_std.unsqueeze(0).unsqueeze(0) + output_mean.unsqueeze(0).unsqueeze(0)
    # pred_phys: [B, n_pde_points, 4] = [x, y, tx, ty] in physical units

    x_pred = pred_phys[:, :, 0]    # [B, n_pde_points]
    y_pred = pred_phys[:, :, 1]
    tx_pred = pred_phys[:, :, 2]
    ty_pred = pred_phys[:, :, 3]

    # Get physical z positions:  z = z_start + z_frac * dz
    # z_start comes from the RK4 data generation — it's the z position of the origin.
    # In our data, z_start is implicit. For track extrapolation in the LHCb magnet,
    # z typically ranges from ~2000 to ~14000 mm. However, z_start is NOT stored
    # in our data arrays. We approximate: z_start ≈ some reference.
    #
    # IMPORTANT: The training data was generated with variable z_start.
    # We need z_start to compute physical z for B-field lookup.
    # Since z_start isn't in the data, we use a fixed reference z=2500mm
    # (upstream T-station entry) and dz up to 12000mm covers the magnet.
    #
    # TODO: If more precision is needed, extend data generation to include z_start.
    z_ref = 2500.0  # Approximate upstream reference position in mm
    z_phys = z_ref + z_frac_pde * dz_phys  # [B, n_pde_points]

    # ---- Evaluate magnetic field at predicted positions ----
    with torch.no_grad():
        # Field evaluation doesn't need gradients (it's the "ground truth" physics)
        Bx, By, Bz = field_model(
            x_pred.detach().flatten(),
            y_pred.detach().flatten(),
            z_phys.detach().flatten()
        )
    Bx = Bx.view(B, n_pde_points)
    By = By.view(B, n_pde_points)
    Bz = Bz.view(B, n_pde_points)

    # ---- Compute Lorentz force ODE target ----
    qop = X_raw[:, 4].unsqueeze(-1)     # [B, 1]  q/p in 1/MeV
    kappa = C_LIGHT * qop               # [B, 1]
    N = torch.sqrt(1.0 + tx_pred ** 2 + ty_pred ** 2)  # [B, n_pde_points]

    # Expected derivatives from Lorentz force
    dx_dz_expected = tx_pred
    dy_dz_expected = ty_pred
    dtx_dz_expected = kappa * N * (tx_pred * ty_pred * Bx - (1.0 + tx_pred ** 2) * By + ty_pred * Bz)
    dty_dz_expected = kappa * N * ((1.0 + ty_pred ** 2) * Bx - tx_pred * ty_pred * By - tx_pred * Bz)

    # ---- PDE residual ----
    residual_x  = dstate_dz[:, :, 0] - dx_dz_expected
    residual_y  = dstate_dz[:, :, 1] - dy_dz_expected
    residual_tx = dstate_dz[:, :, 2] - dtx_dz_expected
    residual_ty = dstate_dz[:, :, 3] - dty_dz_expected

    loss_pde = (residual_x ** 2 + residual_y ** 2 +
                residual_tx ** 2 + residual_ty ** 2).mean()

    total = lambda_ic * loss_ic + lambda_endpoint * loss_end + lambda_pde * loss_pde

    return total, {
        'total': total.item(),
        'ic': loss_ic.item(),
        'endpoint': loss_end.item(),
        'pde': loss_pde.item(),
    }


def pde_loss_hybrid(model, X, Y, z_frac, Y_col, X_raw, norm_stats, field_model,
                    lambda_ic=10.0, lambda_endpoint=1.0, lambda_collocation=1.0,
                    lambda_pde=0.1, n_pde_points=8, **kwargs):
    """
    Hybrid PDE-residual loss: supervised collocation + physics ODE residual.

    Combines the supervised PINN loss (with pre-computed collocation ground truth)
    and the PDE-residual loss (autograd + Lorentz force) for maximum constraint.
    """
    device = X.device
    B = X.size(0)

    # ---- Supervised PINN losses (IC + endpoint + collocation) ----
    z0 = torch.zeros(B, device=device)
    pred_ic = model(X, z0)
    target_ic = X[:, :4]
    loss_ic = F.mse_loss(pred_ic, target_ic)

    z1 = torch.ones(B, device=device)
    pred_end = model(X, z1)
    loss_end = F.mse_loss(pred_end, Y)

    pred_col = model(X, z_frac)
    loss_col = F.mse_loss(pred_col, Y_col)

    # ---- PDE residual loss (same as pure mode) ----
    z_frac_pde = torch.rand(B, n_pde_points, device=device) * 0.9 + 0.05
    z_frac_pde.requires_grad_(True)

    pred_pde = model(X, z_frac_pde)

    dpred_dzfrac = torch.zeros_like(pred_pde)
    for i in range(4):
        grad_outputs = torch.ones(B, n_pde_points, device=device)
        grads = torch.autograd.grad(
            pred_pde[:, :, i], z_frac_pde,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]
        dpred_dzfrac[:, :, i] = grads

    dz_phys = X_raw[:, 5].unsqueeze(-1)
    output_std = norm_stats['output_std'].to(device)
    output_mean = norm_stats['output_mean'].to(device)

    dstate_dz = output_std.unsqueeze(0).unsqueeze(0) * dpred_dzfrac / dz_phys.unsqueeze(-1)

    pred_phys = pred_pde * output_std.unsqueeze(0).unsqueeze(0) + output_mean.unsqueeze(0).unsqueeze(0)
    x_pred = pred_phys[:, :, 0]
    y_pred = pred_phys[:, :, 1]
    tx_pred = pred_phys[:, :, 2]
    ty_pred = pred_phys[:, :, 3]

    z_ref = 2500.0
    z_phys = z_ref + z_frac_pde * dz_phys

    with torch.no_grad():
        Bx, By, Bz = field_model(
            x_pred.detach().flatten(),
            y_pred.detach().flatten(),
            z_phys.detach().flatten()
        )
    Bx = Bx.view(B, n_pde_points)
    By = By.view(B, n_pde_points)
    Bz = Bz.view(B, n_pde_points)

    qop = X_raw[:, 4].unsqueeze(-1)
    kappa = C_LIGHT * qop
    N = torch.sqrt(1.0 + tx_pred ** 2 + ty_pred ** 2)

    dx_dz_expected = tx_pred
    dy_dz_expected = ty_pred
    dtx_dz_expected = kappa * N * (tx_pred * ty_pred * Bx - (1.0 + tx_pred ** 2) * By + ty_pred * Bz)
    dty_dz_expected = kappa * N * ((1.0 + ty_pred ** 2) * Bx - tx_pred * ty_pred * By - tx_pred * Bz)

    residual_x  = dstate_dz[:, :, 0] - dx_dz_expected
    residual_y  = dstate_dz[:, :, 1] - dy_dz_expected
    residual_tx = dstate_dz[:, :, 2] - dtx_dz_expected
    residual_ty = dstate_dz[:, :, 3] - dty_dz_expected

    loss_pde = (residual_x ** 2 + residual_y ** 2 +
                residual_tx ** 2 + residual_ty ** 2).mean()

    total = (lambda_ic * loss_ic + lambda_endpoint * loss_end +
             lambda_collocation * loss_col + lambda_pde * loss_pde)

    return total, {
        'total': total.item(),
        'ic': loss_ic.item(),
        'endpoint': loss_end.item(),
        'collocation': loss_col.item(),
        'pde': loss_pde.item(),
    }


def compositional_pinn_loss(model, X, Y, z_frac=None, Y_col=None,
                            lambda_ic=10.0, lambda_endpoint=1.0,
                            lambda_collocation=1.0, use_collocation=False, **kwargs):
    """
    Loss for CompositionalPINN.

    If use_collocation=True and z_frac/Y_col are provided, uses supervised
    collocation loss (like standard PINN loss). Otherwise, endpoint MSE only.
    """
    B = X.size(0)
    pred_end = model(X)  # Endpoint prediction via full chain
    loss_end = F.mse_loss(pred_end, Y)

    if use_collocation and z_frac is not None and Y_col is not None:
        # IC loss
        z0 = torch.zeros(B, device=X.device)
        pred_ic = model(X, z0)
        target_ic = X[:, :4]
        loss_ic = F.mse_loss(pred_ic, target_ic)

        # Supervised collocation
        pred_col = model(X, z_frac)
        loss_col = F.mse_loss(pred_col, Y_col)

        total = lambda_ic * loss_ic + lambda_endpoint * loss_end + lambda_collocation * loss_col
        return total, {
            'total': total.item(),
            'ic': loss_ic.item(),
            'endpoint': loss_end.item(),
            'collocation': loss_col.item(),
        }
    else:
        return loss_end, {'total': loss_end.item()}


# =============================================================================
# Training Loop
# =============================================================================

def train(config: Dict):
    """Main training function."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    model_type = config['model']['type']
    is_pinn = model_type in ('quadratic', 'zfrac')
    is_pde = model_type == 'pde'
    is_compositional = model_type == 'compositional'
    pde_mode = config.get('pde_mode', 'pure')  # 'pure' or 'hybrid'
    use_collocation_data = is_pinn or (is_pde and pde_mode == 'hybrid') or \
                           (is_compositional and config.get('use_collocation', False))

    # ---- Load data ----
    data_path = config['data']['train_path']
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    X_all = data['X']
    Y_all = data['Y']

    if use_collocation_data:
        z_frac_all = data['z_frac']
        Y_col_all = data['Y_col']
        print(f"  X: {X_all.shape}, Y: {Y_all.shape}")
        print(f"  z_frac: {z_frac_all.shape}, Y_col: {Y_col_all.shape}")
    else:
        z_frac_all = None
        Y_col_all = None
        print(f"  X: {X_all.shape}, Y: {Y_all.shape}")

    # ---- Split ----
    n = len(X_all)
    val_frac = config['data'].get('val_split', 0.1)
    n_val = int(n * val_frac)
    n_train = n - n_val

    np.random.seed(config.get('seed', 42))
    idx = np.random.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    # Create datasets
    if use_collocation_data:
        train_ds = V5Dataset(X_all[train_idx], Y_all[train_idx],
                             z_frac_all[train_idx], Y_col_all[train_idx])
        val_ds = V5Dataset(X_all[val_idx], Y_all[val_idx],
                           z_frac_all[val_idx], Y_col_all[val_idx])
    else:
        train_ds = V5Dataset(X_all[train_idx], Y_all[train_idx])
        val_ds = V5Dataset(X_all[val_idx], Y_all[val_idx])

    norm_stats = train_ds.get_norm_stats()

    print(f"  Train: {n_train:,}, Val: {n_val:,}")

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ---- Load magnetic field for PDE models ----
    field_model = None
    if is_pde:
        field_map_path = config.get('field_map_path', None)
        polarity = config.get('polarity', 1)
        print(f"\nLoading magnetic field for PDE-residual loss...")
        field_model = get_field_torch(field_map_path, polarity=polarity, device=str(device))
        field_model.eval()
        for p in field_model.parameters():
            p.requires_grad_(False)
        print(f"  Field model loaded on {device}")

    # ---- Create model ----
    hidden_dims = config['model']['hidden_dims']
    activation = config['model'].get('activation', 'silu')

    model_kwargs = {}
    if is_compositional:
        model_kwargs['n_steps'] = config['model'].get('n_steps', 8)

    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(hidden_dims=hidden_dims, activation=activation, **model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_type} {hidden_dims}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Activation: {activation}")
    if is_compositional:
        print(f"  N steps: {model_kwargs.get('n_steps', 8)}")
    if is_pde:
        print(f"  PDE mode: {pde_mode}")

    # ---- Optimizer & Scheduler ----
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    wd = config['training'].get('weight_decay', 1e-4)
    warmup = config['training'].get('warmup_epochs', 5)
    grad_clip = config['training'].get('grad_clip', 1.0)
    patience = config['training'].get('patience', 20)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])

    # ---- Loss config ----
    loss_config = config.get('loss', {})
    lambda_ic = loss_config.get('lambda_ic', 10.0)
    lambda_end = loss_config.get('lambda_endpoint', 1.0)
    lambda_col = loss_config.get('lambda_collocation', 1.0)
    lambda_pde = loss_config.get('lambda_pde', 1.0)
    n_pde_points = loss_config.get('n_pde_points', 8)

    if is_pinn or (is_pde and pde_mode == 'hybrid'):
        print(f"  Loss weights: λ_ic={lambda_ic}, λ_end={lambda_end}, λ_col={lambda_col}")
    if is_pde:
        print(f"  PDE loss: λ_pde={lambda_pde}, n_pde_points={n_pde_points}")

    # ---- Output ----
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if config['output'].get('tensorboard', False) and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / 'tb_logs')

    # ---- Training loop ----
    print(f"\nTraining for {epochs} epochs (patience={patience})...")
    print("=" * 80)

    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': []}
    no_improve = 0

    for epoch in range(epochs):
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_losses = {}
        n_batch = 0

        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()

            if is_pinn:
                X, Y, zf, Yc, X_raw = batch
                loss, losses = pinn_loss(model, X, Y, zf, Yc,
                                         lambda_ic, lambda_end, lambda_col)
            elif is_pde:
                if pde_mode == 'hybrid':
                    X, Y, zf, Yc, X_raw = batch
                    loss, losses = pde_loss_hybrid(
                        model, X, Y, zf, Yc, X_raw, norm_stats, field_model,
                        lambda_ic=lambda_ic, lambda_endpoint=lambda_end,
                        lambda_collocation=lambda_col, lambda_pde=lambda_pde,
                        n_pde_points=n_pde_points)
                else:  # pure
                    X, Y, X_raw = batch
                    loss, losses = pde_loss_pure(
                        model, X, Y, X_raw, norm_stats, field_model,
                        lambda_ic=lambda_ic, lambda_endpoint=lambda_end,
                        lambda_pde=lambda_pde, n_pde_points=n_pde_points)
            elif is_compositional:
                use_col = config.get('use_collocation', False)
                if use_col and use_collocation_data:
                    X, Y, zf, Yc, X_raw = batch
                    loss, losses = compositional_pinn_loss(
                        model, X, Y, zf, Yc,
                        lambda_ic=lambda_ic, lambda_endpoint=lambda_end,
                        lambda_collocation=lambda_col, use_collocation=True)
                else:
                    X, Y, X_raw = batch
                    loss, losses = mlp_loss(model, X, Y)
            else:  # mlp
                X, Y, X_raw = batch
                loss, losses = mlp_loss(model, X, Y)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k, v in losses.items():
                train_losses[k] = train_losses.get(k, 0) + v
            n_batch += 1

        train_losses = {k: v / n_batch for k, v in train_losses.items()}

        # ---- Validate ----
        model.eval()
        val_losses = {}
        n_batch = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]

                if is_pinn:
                    X, Y, zf, Yc, X_raw = batch
                    _, losses = pinn_loss(model, X, Y, zf, Yc,
                                          lambda_ic, lambda_end, lambda_col)
                elif is_pde:
                    # For validation, use endpoint MSE only (PDE loss needs grad)
                    if pde_mode == 'hybrid':
                        X, Y, zf, Yc, X_raw = batch
                        z1 = torch.ones(X.size(0), device=device)
                        pred_end = model(X, z1)
                        loss_end = F.mse_loss(pred_end, Y)
                        losses = {'total': loss_end.item(), 'endpoint': loss_end.item()}
                    else:
                        X, Y, X_raw = batch
                        z1 = torch.ones(X.size(0), device=device)
                        pred_end = model(X, z1)
                        loss_end = F.mse_loss(pred_end, Y)
                        losses = {'total': loss_end.item(), 'endpoint': loss_end.item()}
                elif is_compositional:
                    use_col = config.get('use_collocation', False)
                    if use_col and use_collocation_data:
                        X, Y, zf, Yc, X_raw = batch
                        _, losses = compositional_pinn_loss(
                            model, X, Y, zf, Yc,
                            lambda_ic=lambda_ic, lambda_endpoint=lambda_end,
                            lambda_collocation=lambda_col, use_collocation=True)
                    else:
                        X, Y, X_raw = batch
                        _, losses = mlp_loss(model, X, Y)
                else:
                    X, Y, X_raw = batch
                    _, losses = mlp_loss(model, X, Y)

                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v
                n_batch += 1

        val_losses = {k: v / n_batch for k, v in val_losses.items()}

        scheduler.step()
        history['train'].append(train_losses)
        history['val'].append(val_losses)

        elapsed = time.time() - t0

        # TensorBoard
        if writer:
            for k, v in train_losses.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_losses.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            loss_parts = []
            for k, v in train_losses.items():
                if k != 'total':
                    loss_parts.append(f"{k}={v:.4f}")
            parts_str = f" ({', '.join(loss_parts)})" if loss_parts else ""

            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train: {train_losses['total']:.6f}{parts_str} | "
                  f"Val: {val_losses['total']:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'norm_stats': norm_stats,
                'model_type': model_type,
                'hidden_dims': hidden_dims,
            }, output_dir / 'best_model.pt')
        else:
            no_improve += 1

        # Periodic checkpoint
        save_every = config['output'].get('save_every', 10)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'norm_stats': norm_stats,
            }, output_dir / f'checkpoint_epoch{epoch + 1}.pt')

        # Early stopping
        if patience > 0 and no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    # ---- Save final model ----
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'norm_stats': norm_stats,
        'history': history,
        'model_type': model_type,
        'hidden_dims': hidden_dims,
    }, output_dir / 'final_model.pt')

    # Save config copy
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        hist_ser = {
            'train': [{k: float(v) for k, v in h.items()} for h in history['train']],
            'val': [{k: float(v) for k, v in h.items()} for h in history['val']],
        }
        json.dump(hist_ser, f, indent=2)

    if writer:
        writer.close()

    print("=" * 80)
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
    print(f"  Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='V5 Training Script')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print("=" * 80)
    print(f"V5 Training: {config.get('description', args.config)}")
    print("=" * 80)

    train(config)


if __name__ == '__main__':
    main()
