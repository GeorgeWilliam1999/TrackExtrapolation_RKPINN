#!/usr/bin/env python3
"""
================================================================================
Neural Network Architectures for LHCb Track Extrapolation
================================================================================

This module provides three neural network architectures for track extrapolation:

1. MLP (Multi-Layer Perceptron)
   - Standard feedforward network
   - Trained with data loss only (supervised learning)
   - Fastest inference, simplest to train
   - No physics constraints

2. PINN (Physics-Informed Neural Network)
   - Network + Lorentz force PDE residual loss via autodiff
   - Learns trajectory y(z) that satisfies governing equations
   - Enforces: initial condition + PDE at collocation points + endpoint
   - Physics-constrained, better generalization

3. RK_PINN (Runge-Kutta Physics-Informed Neural Network)
   - Multi-stage architecture inspired by RK4 numerical integrator
   - Shared backbone + 4 stage-specific heads at z fractions [0.25, 0.5, 0.75, 1.0]
   - Learnable combination weights (initialized to RK4: [1,2,2,1]/6)
   - Physics loss at each stage

Physics Background
------------------
Charged particles in a magnetic field follow the Lorentz force:

    F = q(v × B)

In z-parameterization (as used in LHCb), the equations of motion are:

    dx/dz  = tx
    dy/dz  = ty  
    dtx/dz = κ · N · [tx·ty·Bx - (1+tx²)·By + ty·Bz]
    dty/dz = κ · N · [(1+ty²)·Bx - tx·ty·By - tx·Bz]

where:
    - κ = (q/p) × c_light, with c_light = 2.99792458×10⁻⁴
    - N = √(1 + tx² + ty²) is the normalization factor
    - Bx, By, Bz are magnetic field components in Tesla
    - q/p is charge over momentum in 1/MeV

The dominant field component in LHCb is By (vertical), which causes
horizontal (x) bending of tracks.

Usage Examples
--------------
>>> from architectures import MLP, PINN, RK_PINN, create_model
>>> 
>>> # Create MLP model
>>> mlp = MLP(hidden_dims=[256, 256, 128], activation='silu')
>>> 
>>> # Create PINN model with physics loss
>>> pinn = PINN(hidden_dims=[256, 256, 128], lambda_pde=1.0, n_collocation=10)
>>> 
>>> # Create model from registry
>>> model = create_model('rk_pinn', hidden_dims=[256, 256, 128])
>>> 
>>> # Forward pass
>>> x = torch.randn(32, 6)  # [x, y, tx, ty, q/p, dz]
>>> y = model(x)            # [x_f, y_f, tx_f, ty_f]

Architecture Presets
--------------------
    tiny:   [64, 64]           ~5k params    - Quick debugging
    small:  [128, 128]         ~20k params   - Fast training baseline
    medium: [256, 256, 128]    ~100k params  - Balanced performance
    large:  [512, 512, 256]    ~300k params  - High accuracy
    wide:   [512, 512, 256, 128] ~500k params - Maximum accuracy

Author: G. Scriven
Date: January 2026
LHCb Track Extrapolation Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings


# =============================================================================
# Physical Constants
# =============================================================================

# Speed of light factor for Lorentz force calculation
# Converts (q/p in 1/MeV) × (B in Tesla) → curvature in 1/mm
# This is the critical constant that matches LHCb's C++ extrapolators
C_LIGHT = 2.99792458e-4  # c in mm/ns × conversion factors


# =============================================================================
# Activation Function Registry
# =============================================================================

ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'silu': nn.SiLU,      # Swish - recommended for most cases
    'gelu': nn.GELU,      # Gaussian Error Linear Unit
    'elu': nn.ELU,
    'mish': nn.Mish,
}


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation name ('relu', 'tanh', 'silu', 'gelu', etc.)
        
    Returns:
        Activation module instance
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]()


# =============================================================================
# Magnetic Field Models
# =============================================================================

DEFAULT_FIELD_PARAMS = {
    'B0': -1.0182,        # Peak field strength (Tesla)
    'z_center': 5007.0,   # Center of dipole magnet (mm)
    'z_width': 1744.0,    # Gaussian width parameter (mm)
}


class MagneticField(nn.Module):
    """
    Simplified LHCb dipole magnetic field model (differentiable).
    
    Models field as Gaussian profile: By(z) = B0 × exp(-0.5 × ((z - z_center) / z_width)²)
    
    Field Parameters (fitted from twodip.rtf):
        B0 = -1.0182 T      Peak field (negative = pointing down in y)
        z_center = 5007 mm  Center of magnet
        z_width = 1744 mm   Characteristic Gaussian width
    """
    
    def __init__(
        self,
        B0: float = DEFAULT_FIELD_PARAMS['B0'],
        z_center: float = DEFAULT_FIELD_PARAMS['z_center'],
        z_width: float = DEFAULT_FIELD_PARAMS['z_width'],
    ):
        super().__init__()
        self.register_buffer('B0', torch.tensor(B0, dtype=torch.float32))
        self.register_buffer('z_center', torch.tensor(z_center, dtype=torch.float32))
        self.register_buffer('z_width', torch.tensor(z_width, dtype=torch.float32))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute magnetic field at position(s) z. Returns (Bx, By, Bz)."""
        z_rel = (z - self.z_center) / self.z_width
        By = self.B0 * torch.exp(-0.5 * z_rel**2)
        Bx = torch.zeros_like(By)
        Bz = torch.zeros_like(By)
        return Bx, By, Bz


# =============================================================================
# Base Model Class
# =============================================================================

class BaseTrackExtrapolator(nn.Module):
    """
    Base class for all track extrapolation neural networks.
    
    Provides:
    - Input/output normalization (z-score standardization)
    - Normalization persistence (save/load)
    - Parameter counting
    - Configuration export
    
    Input Format [6]: [x, y, tx, ty, q/p, dz]
    Output Format [4]: [x_f, y_f, tx_f, ty_f]
    """
    
    def __init__(self, input_dim: int = 6, output_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
        self._normalization_set = False
    
    def set_normalization(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> None:
        """Compute normalization from training data."""
        self.input_mean = X.mean(dim=0)
        self.input_std = X.std(dim=0) + eps
        self.output_mean = Y.mean(dim=0)
        self.output_std = Y.std(dim=0) + eps
        self._normalization_set = True
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.output_std + self.output_mean
    
    def save_normalization(self, filepath: str) -> None:
        norm_dict = {
            'input_mean': self.input_mean.cpu().tolist(),
            'input_std': self.input_std.cpu().tolist(),
            'output_mean': self.output_mean.cpu().tolist(),
            'output_std': self.output_std.cpu().tolist(),
        }
        with open(filepath, 'w') as f:
            json.dump(norm_dict, f, indent=2)
    
    def load_normalization(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            norm_dict = json.load(f)
        self.input_mean = torch.tensor(norm_dict['input_mean'])
        self.input_std = torch.tensor(norm_dict['input_std'])
        self.output_mean = torch.tensor(norm_dict['output_mean'])
        self.output_std = torch.tensor(norm_dict['output_std'])
        self._normalization_set = True
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> dict:
        raise NotImplementedError("Subclasses must implement get_config()")


# =============================================================================
# MLP - Multi-Layer Perceptron
# =============================================================================

class MLP(BaseTrackExtrapolator):
    """
    Multi-Layer Perceptron for Track Extrapolation.
    
    Standard feedforward network trained with data loss only.
    
    Architecture:
        Input [6] → Normalize → [FC→Act→(Dropout)]×N → FC → Denormalize → Output [4]
    
    Training Loss:
        L = MSE(prediction, ground_truth)
    
    Example:
        >>> model = MLP(hidden_dims=[256, 256, 128], activation='silu')
        >>> y = model(x)  # x: [batch, 6], y: [batch, 4]
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = 'silu',
        dropout: float = 0.0,
        input_dim: int = 6,
        output_dim: int = 4,
    ):
        """
        Args:
            hidden_dims: List of hidden layer sizes
            activation: Activation function ('relu', 'silu', 'gelu', 'tanh', etc.)
            dropout: Dropout probability (0 = no dropout)
            input_dim: Number of input features (default 6)
            output_dim: Number of output features (default 4)
        """
        super().__init__(input_dim, output_dim)
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [batch, 6] → [batch, 4]"""
        x_norm = self.normalize_input(x)
        y_norm = self.network(x_norm)
        return self.denormalize_output(y_norm)
    
    def get_config(self) -> dict:
        return {
            'model_type': 'MLP',
            'hidden_dims': self.hidden_dims,
            'activation': self.activation_name,
            'dropout': self.dropout_rate,
            'parameters': self.count_parameters(),
        }


# =============================================================================
# PINN - Physics-Informed Neural Network
# =============================================================================

class PINN(BaseTrackExtrapolator):
    """
    Physics-Informed Neural Network for Track Extrapolation.
    
    Uses autodiff to enforce Lorentz force equations as PDE constraint.
    Network learns continuous trajectory y(z) satisfying governing equations.
    
    Training Loss:
        L = L_data + λ_ic · L_ic + λ_pde · L_pde
        
        L_data : MSE at endpoint vs ground truth
        L_ic   : MSE at z=0 vs initial condition
        L_pde  : Σ ||∂y/∂z - F(y, B)||² at collocation points
    
    Physics Equations (Lorentz force):
        dx/dz  = tx
        dy/dz  = ty
        dtx/dz = κ · N · [tx·ty·Bx - (1+tx²)·By + ty·Bz]
        dty/dz = κ · N · [(1+ty²)·Bx - tx·ty·By - tx·Bz]
    
    Example:
        >>> pinn = PINN(hidden_dims=[256, 256, 128], lambda_pde=1.0)
        >>> y = pinn(x)
        >>> physics_loss = pinn.compute_physics_loss(x, y)
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = 'tanh',
        dropout: float = 0.0,
        lambda_pde: float = 1.0,
        lambda_ic: float = 1.0,
        n_collocation: int = 10,
        field_model: Optional[nn.Module] = None,
        z_start: float = 0.0,
        z_end: float = 2300.0,
    ):
        """
        Args:
            hidden_dims: Hidden layer sizes
            activation: Activation function ('tanh' recommended for PINNs)
            dropout: Dropout probability
            lambda_pde: Weight for PDE residual loss
            lambda_ic: Weight for initial condition loss
            n_collocation: Number of collocation points for PDE loss
            field_model: Magnetic field model (default: MagneticField)
            z_start, z_end: Propagation range (mm)
        """
        super().__init__(input_dim=6, output_dim=4)
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.n_collocation = n_collocation
        self.z_start = z_start
        self.z_end = z_end
        
        self.field = field_model if field_model is not None else MagneticField()
        
        # Build network
        layers = []
        prev_dim = 6
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass at endpoint (z_norm=1)."""
        batch_size = x.shape[0]
        x_query = x.clone()
        x_query[:, 5] = 1.0  # z_normalized = 1 for endpoint
        x_norm = self.normalize_input(x_query)
        y_norm = self.network(x_norm)
        return self.denormalize_output(y_norm)
    
    def forward_at_z(self, x0: torch.Tensor, z_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass at arbitrary z position."""
        if z_norm.dim() == 0:
            z_norm = z_norm.unsqueeze(0)
        if z_norm.dim() == 1:
            z_norm = z_norm.unsqueeze(1)
        if z_norm.shape[0] == 1 and x0.shape[0] > 1:
            z_norm = z_norm.expand(x0.shape[0], 1)
        
        x_full = torch.cat([x0, z_norm], dim=1)
        x_norm = self.normalize_input(x_full)
        y_norm = self.network(x_norm)
        return self.denormalize_output(y_norm)
    
    def compute_physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss: IC + PDE residuals.
        
        Args:
            x: Input [batch, 6] = [x0, y0, tx0, ty0, qop, dz]
            y_pred: Model prediction (unused, recomputed with gradients)
            
        Returns:
            Dict with 'ic' and 'pde' loss components (scaled by λ)
        """
        batch_size = x.shape[0]
        device = x.device
        
        x0 = x[:, :5]
        initial_state = x[:, :4]
        qop = x[:, 4]
        dz = x[:, 5]
        kappa = qop * C_LIGHT
        
        # Initial condition loss
        z_zero = torch.zeros((batch_size, 1), device=device)
        y_at_z0 = self.forward_at_z(x0, z_zero)
        ic_loss = F.mse_loss(y_at_z0, initial_state)
        
        # PDE residual loss at collocation points
        z_fracs = torch.linspace(0.1, 1.0, self.n_collocation, device=device)
        total_residual = torch.tensor(0.0, device=device)
        
        for z_frac in z_fracs:
            z_tensor = torch.full((batch_size, 1), z_frac.item(), device=device, requires_grad=True)
            x_full = torch.cat([x0, z_tensor], dim=1)
            x_full.requires_grad_(True)
            
            x_norm = self.normalize_input(x_full)
            y_norm = self.network(x_norm)
            y = self.denormalize_output(y_norm)
            
            # Compute gradients ∂y/∂z
            dy_dz = []
            for i in range(4):
                grad = torch.autograd.grad(
                    outputs=y[:, i].sum(),
                    inputs=z_tensor,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                dy_dz.append(grad.squeeze())
            dy_dz = torch.stack(dy_dz, dim=1)
            
            # Physical position and field
            z_physical = self.z_start + z_frac * dz
            Bx, By, Bz = self.field(z_physical)
            
            tx_pred = y[:, 2]
            ty_pred = y[:, 3]
            sqrt_term = torch.sqrt(1 + tx_pred**2 + ty_pred**2)
            
            # Expected derivatives from Lorentz equations
            dx_dz_expected = tx_pred
            dy_dz_expected = ty_pred
            dtx_dz_expected = kappa * sqrt_term * (
                tx_pred * ty_pred * Bx - (1 + tx_pred**2) * By + ty_pred * Bz
            )
            dty_dz_expected = kappa * sqrt_term * (
                (1 + ty_pred**2) * Bx - tx_pred * ty_pred * By - tx_pred * Bz
            )
            
            dy_dz_physical = dy_dz / dz.unsqueeze(1)
            
            residual = (
                (dy_dz_physical[:, 0] - dx_dz_expected)**2 +
                (dy_dz_physical[:, 1] - dy_dz_expected)**2 +
                (dy_dz_physical[:, 2] - dtx_dz_expected)**2 +
                (dy_dz_physical[:, 3] - dty_dz_expected)**2
            ).mean()
            
            total_residual = total_residual + residual
        
        pde_loss = total_residual / self.n_collocation
        
        return {
            'ic': self.lambda_ic * ic_loss,
            'pde': self.lambda_pde * pde_loss,
        }
    
    def get_config(self) -> dict:
        return {
            'model_type': 'PINN',
            'hidden_dims': self.hidden_dims,
            'activation': self.activation_name,
            'dropout': self.dropout_rate,
            'lambda_pde': self.lambda_pde,
            'lambda_ic': self.lambda_ic,
            'n_collocation': self.n_collocation,
            'parameters': self.count_parameters(),
        }


# =============================================================================
# RK_PINN - Runge-Kutta Physics-Informed Neural Network
# =============================================================================

class RK_PINN(BaseTrackExtrapolator):
    """
    Runge-Kutta Physics-Informed Neural Network.
    
    Multi-stage architecture inspired by RK4 numerical integrator:
    - Shared backbone extracts features from initial state
    - 4 stage heads predict state at z = 0.25, 0.5, 0.75, 1.0
    - Learnable weights combine stages (initialized to RK4: [1,2,2,1]/6)
    - Physics loss enforced at each stage position
    
    Architecture:
        Input → Backbone → [Head₁, Head₂, Head₃, Head₄] → Weighted Sum → Output
    
    Training Loss:
        L = L_data + λ_ic · L_ic + λ_pde · L_pde
    
    Example:
        >>> model = RK_PINN(hidden_dims=[256, 256, 128])
        >>> y = model(x)
        >>> stage_preds = model.get_stage_predictions(x)  # 4 predictions
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256, 128],
        activation: str = 'tanh',
        n_stages: int = 4,
        lambda_pde: float = 1.0,
        lambda_ic: float = 1.0,
        field_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            hidden_dims: Hidden layer sizes for backbone
            activation: Activation function
            n_stages: Number of RK stages (default 4)
            lambda_pde: Weight for PDE residual loss
            lambda_ic: Weight for initial condition loss
            field_model: Magnetic field model
        """
        super().__init__(input_dim=6, output_dim=4)
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.n_stages = n_stages
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        
        self.field = field_model if field_model is not None else MagneticField()
        
        # Stage positions
        stage_fracs = torch.linspace(1.0/n_stages, 1.0, n_stages)
        self.register_buffer('stage_fractions', stage_fracs)
        
        # Shared backbone
        backbone_layers = []
        prev_dim = 6
        for dim in hidden_dims[:-1]:
            backbone_layers.append(nn.Linear(prev_dim, dim))
            backbone_layers.append(get_activation(activation))
            prev_dim = dim
        self.backbone = nn.Sequential(*backbone_layers)
        self.backbone_out_dim = prev_dim
        
        # Stage heads
        head_hidden_dim = hidden_dims[-1] if hidden_dims else 64
        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim + 1, head_hidden_dim),
                get_activation(activation),
                nn.Linear(head_hidden_dim, 4)
            )
            for _ in range(n_stages)
        ])
        
        # Learnable weights (initialized to RK4)
        rk4_weights = torch.tensor([1.0, 2.0, 2.0, 1.0]) / 6.0
        if n_stages != 4:
            rk4_weights = torch.ones(n_stages) / n_stages
        self.stage_weights = nn.Parameter(rk4_weights[:n_stages])
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: weighted combination of stage predictions."""
        x_norm = self.normalize_input(x)
        features = self.backbone(x_norm)
        
        stage_outputs = []
        for head, z_frac in zip(self.stage_heads, self.stage_fractions):
            z_input = torch.full((features.shape[0], 1), z_frac.item(), device=features.device)
            head_input = torch.cat([features, z_input], dim=1)
            stage_outputs.append(head(head_input))
        
        weights = F.softmax(self.stage_weights, dim=0)
        y_norm = sum(w * out for w, out in zip(weights, stage_outputs))
        return self.denormalize_output(y_norm)
    
    def get_stage_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get individual stage predictions."""
        x_norm = self.normalize_input(x)
        features = self.backbone(x_norm)
        
        predictions = []
        for head, z_frac in zip(self.stage_heads, self.stage_fractions):
            z_input = torch.full((features.shape[0], 1), z_frac.item(), device=features.device)
            head_input = torch.cat([features, z_input], dim=1)
            predictions.append(self.denormalize_output(head(head_input)))
        return predictions
    
    def compute_physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute physics loss at each RK stage."""
        batch_size = x.shape[0]
        device = x.device
        
        initial_state = x[:, :4]
        qop = x[:, 4]
        dz = x[:, 5]
        kappa = qop * C_LIGHT
        
        x_norm = self.normalize_input(x)
        features = self.backbone(x_norm)
        
        # Initial condition loss
        z_zero = torch.zeros((batch_size, 1), device=device)
        head_input_z0 = torch.cat([features, z_zero], dim=1)
        y_at_z0_list = [self.denormalize_output(head(head_input_z0)) for head in self.stage_heads]
        y_at_z0 = torch.stack(y_at_z0_list).mean(dim=0)
        ic_loss = F.mse_loss(y_at_z0, initial_state)
        
        # PDE residual at each stage
        total_residual = torch.tensor(0.0, device=device)
        
        for head, z_frac in zip(self.stage_heads, self.stage_fractions):
            z_tensor = torch.full((batch_size, 1), z_frac.item(), device=device, requires_grad=True)
            head_input = torch.cat([features.detach(), z_tensor], dim=1)
            y_stage_norm = head(head_input)
            y_stage = self.denormalize_output(y_stage_norm)
            
            dy_dz = []
            for j in range(4):
                grad = torch.autograd.grad(
                    outputs=y_stage[:, j].sum(),
                    inputs=z_tensor,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                dy_dz.append(grad.squeeze())
            dy_dz = torch.stack(dy_dz, dim=1)
            
            z_physical = z_frac * dz
            Bx, By, Bz = self.field(z_physical)
            
            tx_pred = y_stage[:, 2]
            ty_pred = y_stage[:, 3]
            sqrt_term = torch.sqrt(1 + tx_pred**2 + ty_pred**2)
            
            dx_dz_expected = tx_pred
            dy_dz_expected = ty_pred
            dtx_dz_expected = kappa * sqrt_term * (
                tx_pred * ty_pred * Bx - (1 + tx_pred**2) * By + ty_pred * Bz
            )
            dty_dz_expected = kappa * sqrt_term * (
                (1 + ty_pred**2) * Bx - tx_pred * ty_pred * By - tx_pred * Bz
            )
            
            dy_dz_physical = dy_dz / dz.unsqueeze(1)
            
            residual = (
                (dy_dz_physical[:, 0] - dx_dz_expected)**2 +
                (dy_dz_physical[:, 1] - dy_dz_expected)**2 +
                (dy_dz_physical[:, 2] - dtx_dz_expected)**2 +
                (dy_dz_physical[:, 3] - dty_dz_expected)**2
            ).mean()
            
            total_residual = total_residual + residual
        
        pde_loss = total_residual / self.n_stages
        
        return {
            'ic': self.lambda_ic * ic_loss,
            'pde': self.lambda_pde * pde_loss,
        }
    
    def get_config(self) -> dict:
        weights = F.softmax(self.stage_weights, dim=0)
        return {
            'model_type': 'RK_PINN',
            'hidden_dims': self.hidden_dims,
            'activation': self.activation_name,
            'n_stages': self.n_stages,
            'lambda_pde': self.lambda_pde,
            'lambda_ic': self.lambda_ic,
            'stage_fractions': self.stage_fractions.cpu().tolist(),
            'stage_weights': weights.detach().cpu().tolist(),
            'parameters': self.count_parameters(),
        }


# =============================================================================
# Model Registry and Factory
# =============================================================================

MODEL_REGISTRY = {
    'mlp': MLP,
    'pinn': PINN,
    'rk_pinn': RK_PINN,
}


def create_model(model_type: str, **kwargs) -> BaseTrackExtrapolator:
    """Create model from registry. model_type: 'mlp', 'pinn', or 'rk_pinn'."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)


# =============================================================================
# Architecture Presets
# =============================================================================

ARCHITECTURE_PRESETS = {
    'tiny': {'hidden_dims': [64, 64], 'description': '~5k params'},
    'small': {'hidden_dims': [128, 128], 'description': '~20k params'},
    'medium': {'hidden_dims': [256, 256, 128], 'description': '~100k params'},
    'large': {'hidden_dims': [512, 512, 256], 'description': '~400k params'},
    'wide': {'hidden_dims': [512, 512, 256, 128], 'description': '~500k params'},
}


def get_preset_config(preset: str) -> dict:
    """Get architecture configuration from preset name."""
    if preset not in ARCHITECTURE_PRESETS:
        raise ValueError(f"Unknown preset: '{preset}'. Available: {list(ARCHITECTURE_PRESETS.keys())}")
    return ARCHITECTURE_PRESETS[preset].copy()


def list_presets() -> None:
    """Print available architecture presets."""
    print("Architecture Presets:")
    for name, config in ARCHITECTURE_PRESETS.items():
        print(f"  {name:8s}: {config['hidden_dims']} - {config['description']}")


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Neural Network Architectures - Self Test")
    print("=" * 60)
    
    batch_size = 32
    x = torch.randn(batch_size, 6)
    
    for model_name in MODEL_REGISTRY.keys():
        print(f"\n{model_name.upper()}:")
        model = create_model(model_name, hidden_dims=[64, 32])
        y = model(x)
        config = model.get_config()
        print(f"  Input:  {tuple(x.shape)} → Output: {tuple(y.shape)}")
        print(f"  Params: {config['parameters']:,}")
        
        if hasattr(model, 'compute_physics_loss'):
            losses = model.compute_physics_loss(x, y)
            print(f"  Physics: IC={losses['ic']:.4f}, PDE={losses['pde']:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
