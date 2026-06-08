#!/usr/bin/env python3
"""
Fix I — detector-resolution-weighted endpoint loss.

The Gen-3 training loss (gen3_protocol.tex §4.3) replaces plain MSE with

    L_det = (1/N) Σ_i Σ_{k ∈ {x,y,tx,ty}} ((ŷ_ik - y_ik) / σ_k^det(z_f^i))²

where ``σ_k^det(z_f)`` is the detector resolution of the station nearest to
``z_f`` (the predicted endpoint z).

Stations and resolutions (from the protocol):

    VELO   ( z ≲ 1500 mm ):  σ_x = σ_y = 12 µm,   σ_tx = σ_ty = 1e-4
    UT     (1500-4000 mm  ):  σ_x = 50 µm, σ_y =  5 µm, σ_tx = σ_ty = 1e-4
    SciFi  ( z ≳ 4000 mm ):  σ_x = 60 µm, σ_y = 500 µm, σ_tx = σ_ty = 3e-5

The 5th output column (qop_f) is vacuum-conserved in NeuralRK4 (identity from
the 5th input); we include it with a benign large ``σ_qop = 1.0`` so its
contribution to the loss is essentially zero and does not mask the four
kinematic residuals.
"""

from __future__ import annotations

import torch


# Station break points (mm)
_VELO_MAX_Z = 1500.0
_UT_MAX_Z = 4000.0

# Resolutions — all in native units (mm for positions, dimensionless for slopes)
_SIGMA_VELO = (0.012, 0.012, 1.0e-4, 1.0e-4)
_SIGMA_UT   = (0.050, 0.005, 1.0e-4, 1.0e-4)
_SIGMA_SCIFI = (0.060, 0.500, 3.0e-5, 3.0e-5)

# qop pass-through weight (output col 4). Keep small gradient contribution.
_SIGMA_QOP = 1.0


def detector_sigma(z_f: torch.Tensor) -> torch.Tensor:
    """Return per-sample σ vector of shape ``[B, 5]`` for z_f in mm.

    Station assignment is piecewise-constant:

        z_f < 1500         -> VELO
        1500 ≤ z_f < 4000  -> UT
        z_f ≥ 4000         -> SciFi
    """
    device = z_f.device
    dtype = z_f.dtype
    B = z_f.shape[0]

    sigma = torch.empty((B, 5), device=device, dtype=dtype)
    is_velo = z_f < _VELO_MAX_Z
    is_ut = (~is_velo) & (z_f < _UT_MAX_Z)
    is_scifi = z_f >= _UT_MAX_Z

    for col, (v, u, s) in enumerate(zip(_SIGMA_VELO, _SIGMA_UT, _SIGMA_SCIFI)):
        sigma[:, col] = torch.where(
            is_velo, torch.tensor(v, device=device, dtype=dtype),
            torch.where(is_ut, torch.tensor(u, device=device, dtype=dtype),
                        torch.tensor(s, device=device, dtype=dtype)),
        )
    sigma[:, 4] = _SIGMA_QOP
    return sigma


def detector_sigma_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    z_f: torch.Tensor,
    per_component: bool = False,
) -> torch.Tensor | dict:
    """Detector-σ-weighted squared error.

    Args:
        y_pred: ``[B, 5]`` predicted output.
        y_true: ``[B, 5]`` ground-truth output.
        z_f: ``[B]`` endpoint z in mm (``z_start + dz``).
        per_component: if True, also return the per-column mean squared error
            (unweighted) as a dict.

    Returns:
        Scalar loss (default) or a dict with the scalar loss and the 5 per-
        column unweighted MSEs when ``per_component=True``.
    """
    sigma = detector_sigma(z_f)                        # [B, 5]
    scaled = (y_pred - y_true) / sigma                 # [B, 5]
    loss = scaled.pow(2).mean()                        # scalar

    if per_component:
        with torch.no_grad():
            raw = (y_pred - y_true).pow(2).mean(dim=0)  # [5]
            return {
                "loss": loss,
                "mse_x": raw[0].item(),
                "mse_y": raw[1].item(),
                "mse_tx": raw[2].item(),
                "mse_ty": raw[3].item(),
                "mse_qop": raw[4].item(),
            }
    return loss
