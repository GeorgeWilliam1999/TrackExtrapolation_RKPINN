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
import sys
from pathlib import Path

# Add utils directory to path for unified imports
_utils_dir = Path(__file__).parent.parent / 'utils'
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

# Import unified magnetic field module - SINGLE SOURCE OF TRUTH
# This ensures consistent field model between data generation and PINN training
try:
    from magnetic_field import (
        GaussianFieldTorch, InterpolatedFieldTorch,
        get_field_torch, C_LIGHT
    )
    UNIFIED_FIELD_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"Could not import unified magnetic_field module: {e}\n"
        f"Using local field definitions. This may cause inconsistencies!"
    )
    UNIFIED_FIELD_AVAILABLE = False
    # Fallback constant
    C_LIGHT = 2.99792458e-4


# =============================================================================
# Physical Constants (re-exported from magnetic_field module for compatibility)
# =============================================================================

# Speed of light factor for Lorentz force calculation
# Converts (q/p in 1/MeV) × (B in Tesla) → curvature in 1/mm
# NOTE: C_LIGHT is imported from magnetic_field.py for consistency


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
# Magnetic Field Models (UNIFIED - imported from utils/magnetic_field.py)
# =============================================================================

# IMPORTANT: All field models are now imported from the unified magnetic_field module
# to ensure consistency between data generation and PINN training.
# 
# The unified module is the SINGLE SOURCE OF TRUTH for:
# - Data generation (RK4 integrator in generate_cpp_data.py)
# - PINN physics loss computation (this file)
# - Model validation and testing
#
# Using different field implementations will cause systematic errors!

DEFAULT_FIELD_PARAMS = {
    'B0': -1.0182,        # Peak field strength (Tesla)
    'z_center': 5007.0,   # Center of dipole magnet (mm)
    'z_width': 1744.0,    # Gaussian width parameter (mm)
}


if UNIFIED_FIELD_AVAILABLE:
    # Use unified module - RECOMMENDED
    GaussianMagneticField = GaussianFieldTorch
    InterpolatedMagneticField = InterpolatedFieldTorch
    
    # Default field for PINN physics loss - MUST match data generation!
    # Data is generated with InterpolatedFieldNumpy (twodip.rtf)
    # So PINN physics loss must use InterpolatedFieldTorch for consistency
    def _get_default_field():
        """Get default field model (interpolated, matching data generation)."""
        return get_field_torch(use_interpolated=True)
    
    # Alias for backward compatibility (Gaussian for simple cases)
    MagneticField = GaussianFieldTorch
    
    def create_magnetic_field(field_type: str = 'gaussian', field_map_path: str = None, **kwargs) -> nn.Module:
        """
        Factory function to create magnetic field model.
        
        IMPORTANT: Uses unified magnetic_field module for consistency.
        
        Args:
            field_type: 'gaussian' for analytical approximation, 'interpolated' for field map
            field_map_path: Path to field map file (required if field_type='interpolated')
            **kwargs: Additional arguments passed to field constructor
            
        Returns:
            Magnetic field module from unified module
        """
        return get_field_torch(use_interpolated=(field_type == 'interpolated'), 
                               field_map_path=field_map_path, **kwargs)

else:
    # Fallback definitions - only used if unified module import fails
    warnings.warn(
        "Using LOCAL field definitions because unified module is unavailable. "
        "This may cause inconsistencies with data generation!"
    )
    
    class GaussianMagneticField(nn.Module):
        """
        Simplified Gaussian approximation of LHCb dipole magnetic field (FALLBACK).
        
        WARNING: This is a fallback. Prefer importing from unified magnetic_field module.
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
        
        def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            z_rel = (z - self.z_center) / self.z_width
            By = self.B0 * torch.exp(-0.5 * z_rel**2)
            Bx = torch.zeros_like(By)
            Bz = torch.zeros_like(By)
            return Bx, By, Bz
    
    
    class InterpolatedMagneticField(nn.Module):
        """
        Interpolated 3D magnetic field from tabulated field map (FALLBACK).
        
        WARNING: This is a fallback. Prefer importing from unified magnetic_field module.
        """
        
        def __init__(self, field_map_path: str, device: str = 'cpu'):
            super().__init__()
            self._load_field_map(field_map_path, device)
        
        def _load_field_map(self, path: str, device: str) -> None:
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
            
            self.register_buffer('dx', torch.tensor(x_unique[1] - x_unique[0], dtype=torch.float32))
            self.register_buffer('dy', torch.tensor(y_unique[1] - y_unique[0], dtype=torch.float32))
            self.register_buffer('dz', torch.tensor(z_unique[1] - z_unique[0], dtype=torch.float32))
            
            # Note: File ordering is y-fastest, then x, then z -> reshape order is (nz, nx, ny)
            Bx_grid = Bx.reshape(nz, nx, ny).transpose(1, 2, 0)  # [nx, ny, nz]
            By_grid = By.reshape(nz, nx, ny).transpose(1, 2, 0)
            Bz_grid = Bz.reshape(nz, nx, ny).transpose(1, 2, 0)
            
            B_grid = np.stack([Bx_grid, By_grid, Bz_grid], axis=0)
            B_grid = np.transpose(B_grid, (0, 3, 2, 1))  # [3, nz, ny, nx]
            B_grid = B_grid[np.newaxis, ...]  # [1, 3, nz, ny, nx]
            
            self.register_buffer('B_grid', torch.tensor(B_grid, dtype=torch.float32))
            self.grid_shape = (nz, ny, nx)
        
        def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
            y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
            z_norm = 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0
            
            x_norm = torch.clamp(x_norm, -1.0, 1.0)
            y_norm = torch.clamp(y_norm, -1.0, 1.0)
            z_norm = torch.clamp(z_norm, -1.0, 1.0)
            
            original_shape = x.shape
            grid = torch.stack([x_norm.flatten(), y_norm.flatten(), z_norm.flatten()], dim=-1)
            grid = grid.view(1, 1, 1, -1, 3)
            
            B_interp = torch.nn.functional.grid_sample(
                self.B_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            
            B_interp = B_interp.squeeze().T
            
            Bx = B_interp[:, 0].view(original_shape)
            By = B_interp[:, 1].view(original_shape)
            Bz = B_interp[:, 2].view(original_shape)
            
            return Bx, By, Bz
    
    # Alias for backward compatibility
    MagneticField = GaussianMagneticField
    
    def create_magnetic_field(field_type: str = 'gaussian', field_map_path: str = None, **kwargs) -> nn.Module:
        """Factory function to create magnetic field model (FALLBACK)."""
        if field_type == 'gaussian':
            return GaussianMagneticField(**kwargs)
        elif field_type == 'interpolated':
            if field_map_path is None:
                raise ValueError("field_map_path required for interpolated field")
            return InterpolatedMagneticField(field_map_path, **kwargs)
        else:
            raise ValueError(f"Unknown field_type: {field_type}")


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
    Physics-Informed Neural Network with Residual Formulation for Track Extrapolation.
    
    KEY INNOVATION: Uses skip connections that guarantee initial condition (IC) satisfaction.
    
    ARCHITECTURE:
    -------------
    Output = InitialState + z_frac * NetworkCorrection
    
    At z=0: Output = InitialState (IC automatically satisfied!)
    At z=1: Output = InitialState + NetworkCorrection (learned displacement)
    
    The correction is multiplied by z_frac, so the network CANNOT ignore z.
    
    Training Loss:
        L = L_data + λ_ic · L_ic + λ_pde · L_pde
        
        L_data : MSE at endpoint vs ground truth
        L_ic   : MSE at z=0 vs initial condition (should be ~0 by construction)
        L_pde  : Σ ||∂y/∂z - F(y, B)||² at collocation points
    
    Physics Equations (Lorentz force):
        dx/dz  = tx
        dy/dz  = ty
        dtx/dz = κ · N · [tx·ty·Bx - (1+tx²)·By + ty·Bz]
        dty/dz = κ · N · [(1+ty²)·Bx - tx·ty·By - tx·Bz]
    
    Example:
        >>> pinn = PINN(hidden_dims=[256, 256], lambda_pde=1.0)
        >>> y = pinn(x)
        >>> # At z=0, y should equal initial state (guaranteed!)
        >>> y_z0 = pinn.forward_at_z(x[:, :5], torch.zeros(1))
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'tanh',
        dropout: float = 0.0,
        lambda_pde: float = 1.0,
        lambda_ic: float = 0.1,  # Lower weight since IC is satisfied by construction
        n_collocation: int = 10,
        field_model: Optional[nn.Module] = None,
        z_start: float = 0.0,
        z_end: float = 8000.0,
    ):
        """
        Args:
            hidden_dims: Hidden layer sizes for feature encoder
            activation: Activation function ('tanh' recommended for smooth gradients)
            dropout: Dropout probability
            lambda_pde: Weight for PDE residual loss
            lambda_ic: Weight for initial condition loss (low since IC is automatic)
            n_collocation: Number of collocation points for PDE loss
            field_model: Magnetic field model
            z_start, z_end: Propagation range (mm) - z_end is dz (step size)
        """
        super().__init__(input_dim=6, output_dim=4)
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.n_collocation = n_collocation
        self.z_start = z_start
        self.z_end = z_end  # This is dz (8000 mm typically)
        
        # Magnetic field model
        if field_model is not None:
            self.field = field_model
        elif UNIFIED_FIELD_AVAILABLE:
            self.field = _get_default_field()
        else:
            self.field = MagneticField()
        
        # Feature encoder: [x0, y0, tx0, ty0, qop] → features
        # Note: z_frac is NOT input to encoder - it's used in skip connection
        encoder_layers = []
        prev_dim = 5  # Only initial state, no z_frac
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(get_activation(activation))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Correction head: features → [Δtx, Δty, Δx_extra, Δy_extra]
        # Network predicts CORRECTIONS to physics-based baseline
        self.correction_head = nn.Linear(prev_dim, 4)
        
        # Learnable scale for magnetic field correction (helps with training stability)
        self.field_scale = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for stable training with residual formulation."""
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                # Standard initialization for encoder
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Correction head: start with small corrections
        nn.init.uniform_(self.correction_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.correction_head.bias)
    
    def set_normalization(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> None:
        """Compute normalization from training data.
        
        For PINN with residual formulation, we normalize the initial state 
        (columns 0-4) but NOT the z_frac column (5). The network operates 
        in normalized space but outputs are in physical units.
        """
        self.input_mean = X.mean(dim=0).clone()
        self.input_std = X.std(dim=0).clone() + eps
        self.output_mean = Y.mean(dim=0)
        self.output_std = Y.std(dim=0) + eps
        
        # Store dz for physics calculations
        self.register_buffer('dz', torch.tensor(X[:, 5].mean().item()))
        
        # Don't normalize z_frac (column 5) - it will be replaced with 0-1 values
        self.input_mean[5] = 0.0
        self.input_std[5] = 1.0
        
        self._normalization_set = True
    
    def _compute_baseline_trajectory(
        self, 
        x0: torch.Tensor, 
        y0: torch.Tensor,
        tx0: torch.Tensor,
        ty0: torch.Tensor, 
        z_frac: torch.Tensor,
        dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute physics-based baseline trajectory (straight line extrapolation).
        
        For a charged particle, straight-line propagation gives:
            x(z) = x0 + tx0 * Δz
            y(z) = y0 + ty0 * Δz
            tx(z) = tx0  (constant in no-field limit)
            ty(z) = ty0
        
        The neural network will learn CORRECTIONS to this baseline.
        """
        delta_z = z_frac * dz  # Physical distance traveled
        
        x_baseline = x0 + tx0 * delta_z
        y_baseline = y0 + ty0 * delta_z
        tx_baseline = tx0
        ty_baseline = ty0
        
        return x_baseline, y_baseline, tx_baseline, ty_baseline
    
    def forward_at_z(self, x0: torch.Tensor, z_frac: torch.Tensor) -> torch.Tensor:
        """
        Forward pass at arbitrary z position.
        
        Args:
            x0: Initial state [batch, 5] = [x0, y0, tx0, ty0, qop]
            z_frac: Fractional z position [batch, 1] or scalar, in range [0, 1]
        
        Returns:
            y: State at z [batch, 4] = [x, y, tx, ty]
        """
        batch_size = x0.shape[0]
        device = x0.device
        dtype = x0.dtype
        
        # Handle z_frac dimensions
        if z_frac.dim() == 0:
            z_frac = z_frac.unsqueeze(0)
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(1)
        if z_frac.shape[0] == 1 and batch_size > 1:
            z_frac = z_frac.expand(batch_size, 1)
        
        # Extract initial state components
        x0_pos = x0[:, 0:1]
        y0_pos = x0[:, 1:2]
        tx0 = x0[:, 2:3]
        ty0 = x0[:, 3:4]
        qop = x0[:, 4:5]
        
        # Get dz (step size)
        if hasattr(self, 'dz'):
            dz = self.dz
        else:
            dz = torch.tensor(self.z_end, device=device, dtype=dtype)
        
        # Normalize initial state for encoder (not z_frac)
        x0_norm = (x0 - self.input_mean[:5]) / self.input_std[:5]
        
        # Encode initial state
        features = self.encoder(x0_norm)
        
        # Get correction predictions [Δtx, Δty, Δx_extra, Δy_extra]
        corrections = self.correction_head(features)
        
        # Physics baseline (straight-line extrapolation)
        x_base, y_base, tx_base, ty_base = self._compute_baseline_trajectory(
            x0_pos.squeeze(1), y0_pos.squeeze(1),
            tx0.squeeze(1), ty0.squeeze(1),
            z_frac.squeeze(1), dz
        )
        
        # Apply corrections with z_frac modulation
        # Key: corrections are SCALED by z_frac, ensuring IC is satisfied at z=0
        delta_tx = corrections[:, 0] * z_frac.squeeze(1)  # Slope correction
        delta_ty = corrections[:, 1] * z_frac.squeeze(1)
        delta_x = corrections[:, 2] * z_frac.squeeze(1) * dz  # Position correction (scaled by dz)
        delta_y = corrections[:, 3] * z_frac.squeeze(1) * dz
        
        # Final output: baseline + z_frac * corrections
        x_out = x_base + delta_x
        y_out = y_base + delta_y
        tx_out = tx_base + delta_tx
        ty_out = ty_base + delta_ty
        
        return torch.stack([x_out, y_out, tx_out, ty_out], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass at endpoint (z_frac=1)."""
        x0 = x[:, :5]
        z_frac = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        return self.forward_at_z(x0, z_frac)
    
    def compute_physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss: IC + PDE residuals.
        
        Note: Due to the residual formulation, IC loss should naturally be
        very low (ideally zero). This serves as a sanity check.
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        x0 = x[:, :5]
        initial_state = x[:, :4]  # [x0, y0, tx0, ty0]
        qop = x[:, 4]
        
        if hasattr(self, 'dz'):
            dz = self.dz
        else:
            dz = torch.tensor(self.z_end, device=device, dtype=dtype)
        
        kappa = qop * C_LIGHT
        
        # Characteristic scales for normalization
        pos_scale = 500.0   # mm
        slope_scale = 0.1
        
        # =====================================================================
        # Initial Condition Loss (should be ~0 due to residual formulation)
        # =====================================================================
        z_zero = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        y_at_z0 = self.forward_at_z(x0, z_zero)
        
        ic_pos_err = ((y_at_z0[:, :2] - initial_state[:, :2]) / pos_scale).pow(2).mean()
        ic_slope_err = ((y_at_z0[:, 2:] - initial_state[:, 2:]) / slope_scale).pow(2).mean()
        ic_loss = ic_pos_err + ic_slope_err
        
        if torch.isnan(ic_loss) or torch.isinf(ic_loss):
            ic_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        # =====================================================================
        # PDE Residual Loss at Collocation Points
        # =====================================================================
        z_fracs = torch.linspace(0.1, 1.0, self.n_collocation, device=device, dtype=dtype)
        total_residual = torch.tensor(0.0, device=device, dtype=dtype)
        valid_points = 0
        
        for z_frac_val in z_fracs:
            z_tensor = torch.full((batch_size, 1), z_frac_val.item(), device=device, dtype=dtype, requires_grad=True)
            
            # Forward at this z with gradient tracking
            y = self.forward_at_z(x0, z_tensor)
            
            # Clamp to physical range
            y_clamped = y.clone()
            y_clamped[:, 0] = y[:, 0].clamp(-3000, 3000)
            y_clamped[:, 1] = y[:, 1].clamp(-3000, 3000)
            y_clamped[:, 2] = y[:, 2].clamp(-1, 1)
            y_clamped[:, 3] = y[:, 3].clamp(-1, 1)
            
            # Compute gradients ∂y/∂z_frac
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
            
            # Physical z position and field evaluation
            z_physical = self.z_start + z_frac_val * dz
            Bx, By, Bz = self.field(y_clamped[:, 0], y_clamped[:, 1], z_physical.expand(batch_size))
            
            tx_pred = y_clamped[:, 2]
            ty_pred = y_clamped[:, 3]
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
            
            # Convert autograd gradient: ∂y/∂z_physical = (∂y/∂z_frac) / dz
            dz_safe = max(dz.item() if torch.is_tensor(dz) else dz, 100.0)
            dy_dz_physical = dy_dz / dz_safe
            
            # Normalized residuals
            curvature_scale = 1e-5
            
            residual_x = (dy_dz_physical[:, 0] - dx_dz_expected).pow(2)
            residual_y = (dy_dz_physical[:, 1] - dy_dz_expected).pow(2)
            residual_tx = ((dy_dz_physical[:, 2] - dtx_dz_expected) / curvature_scale).pow(2)
            residual_ty = ((dy_dz_physical[:, 3] - dty_dz_expected) / curvature_scale).pow(2)
            
            residual = (residual_x + residual_y + residual_tx + residual_ty).mean()
            
            if not (torch.isnan(residual) or torch.isinf(residual)):
                total_residual = total_residual + residual
                valid_points += 1
        
        if valid_points > 0:
            pde_loss = total_residual / valid_points
        else:
            pde_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        if torch.isnan(pde_loss) or torch.isinf(pde_loss):
            pde_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
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
            'z_end': self.z_end,
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
        
        # Use interpolated field by default to match data generation
        if field_model is not None:
            self.field = field_model
        elif UNIFIED_FIELD_AVAILABLE:
            self.field = _get_default_field()  # InterpolatedFieldTorch
        else:
            self.field = MagneticField()  # Fallback to Gaussian
        
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
            
            # Physical position and field
            # Note: y_stage contains [x_pos, y_pos, tx, ty]
            z_physical = z_frac * dz
            x_pos = y_stage[:, 0]  # x position at this stage
            y_pos = y_stage[:, 1]  # y position at this stage
            Bx, By, Bz = self.field(x_pos, y_pos, z_physical)
            
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
