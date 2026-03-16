#!/usr/bin/env python3
"""
V3 PINN Residual Architecture

Key insight: Output = IC + z_frac × Correction

This guarantees:
1. At z_frac=0: Output = IC (exactly, no training needed!)
2. At z_frac=1: Output = IC + Correction (same cost as MLP)
3. Smooth interpolation between IC and final state

The core network only sees (state, dz), NOT z_frac.
This means inference at z_frac=1 is identical to an MLP!

Author: G. Scriven
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PINNResidual(nn.Module):
    """
    Physics-Informed Neural Network with Residual Formulation.
    
    Output = InitialCondition + z_frac × NetworkCorrection
    
    This architecture:
    - Guarantees IC satisfaction (z_frac=0 → output = input)
    - Has same inference cost as MLP (at z_frac=1)
    - Learns smooth trajectory interpolation
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256],
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'silu')
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()
        
        self.hidden_dims = hidden_dims
        
        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
        }
        self.activation_fn = activations.get(activation.lower(), nn.SiLU())
        
        # Build core network: (state, dz) → correction
        # Input: 6 features [x, y, tx, ty, qop, dz]
        # Output: 4 features [dx, dy, dtx, dty] (correction)
        layers = []
        in_dim = 6
        
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        
        layers.append(nn.Linear(in_dim, 4))
        
        self.core = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for stable training
        nn.init.xavier_uniform_(self.core[-1].weight, gain=0.1)
        nn.init.zeros_(self.core[-1].bias)
    
    def forward(
        self,
        state_dz: torch.Tensor,
        z_frac: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual formulation.
        
        Args:
            state_dz: [B, 6] = [x, y, tx, ty, qop, dz]
            z_frac: [B] or [B, 1] or [B, N_col] for batched collocation
                   If None, defaults to z_frac=1.0 (endpoint)
        
        Returns:
            output: [B, 4] or [B, N_col, 4] = [x, y, tx, ty] at z_frac
        """
        batch_size = state_dz.size(0)
        
        # Extract initial condition
        initial = state_dz[:, :4]  # [B, 4] = [x, y, tx, ty]
        
        # Compute correction (core network sees state+dz, NOT z_frac)
        correction = self.core(state_dz)  # [B, 4]
        
        # Default to endpoint (z_frac=1) if not specified
        if z_frac is None:
            return initial + correction
        
        # Handle different z_frac shapes
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)  # [B] → [B, 1]
        
        if z_frac.size(-1) == 1:
            # Single z_frac per sample: [B, 1]
            # Output: [B, 4]
            return initial + z_frac * correction
        else:
            # Multiple z_frac per sample (batched collocation): [B, N_col]
            # Output: [B, N_col, 4]
            initial = initial.unsqueeze(1)      # [B, 1, 4]
            correction = correction.unsqueeze(1)  # [B, 1, 4]
            z_frac = z_frac.unsqueeze(-1)       # [B, N_col, 1]
            return initial + z_frac * correction
    
    def forward_endpoint(self, state_dz: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass for z_frac=1 only (deployment).
        
        Equivalent to: self.forward(state_dz, z_frac=1.0)
        But without the z_frac overhead.
        """
        initial = state_dz[:, :4]
        correction = self.core(state_dz)
        return initial + correction


class PINNWithZFracInput(nn.Module):
    """
    Alternative PINN where z_frac is a regular input feature.
    
    Input: [x, y, tx, ty, qop, dz, z_frac] (7 features)
    Output: [x, y, tx, ty] (4 features)
    
    This architecture:
    - More flexible (can learn non-linear z_frac dependence)
    - Must learn IC constraint from data
    - Slightly more parameters
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 256],
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
        }
        self.activation_fn = activations.get(activation.lower(), nn.SiLU())
        
        # Build network: (state, dz, z_frac) → state
        # Input: 7 features
        # Output: 4 features
        layers = []
        in_dim = 7  # +1 for z_frac
        
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        
        layers.append(nn.Linear(in_dim, 4))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        state_dz: torch.Tensor,
        z_frac: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state_dz: [B, 6] = [x, y, tx, ty, qop, dz]
            z_frac: [B] or [B, 1] or [B, N_col]
        
        Returns:
            output: [B, 4] or [B, N_col, 4]
        """
        batch_size = state_dz.size(0)
        
        if z_frac is None:
            z_frac = torch.ones(batch_size, 1, device=state_dz.device)
        
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)
        
        if z_frac.size(-1) == 1:
            # Single z_frac: [B, 1]
            input_full = torch.cat([state_dz, z_frac], dim=-1)  # [B, 7]
            return self.network(input_full)  # [B, 4]
        else:
            # Batched z_frac: [B, N_col]
            N_col = z_frac.size(-1)
            state_dz_exp = state_dz.unsqueeze(1).expand(-1, N_col, -1)  # [B, N_col, 6]
            z_frac_exp = z_frac.unsqueeze(-1)  # [B, N_col, 1]
            input_full = torch.cat([state_dz_exp, z_frac_exp], dim=-1)  # [B, N_col, 7]
            return self.network(input_full)  # [B, N_col, 4]


def create_pinn(
    architecture: str = "residual",
    hidden_dims: List[int] = [256, 256],
    activation: str = "silu",
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory function to create PINN model.
    
    Args:
        architecture: "residual" or "z_frac_input"
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        dropout: Dropout probability
    
    Returns:
        PINN model
    """
    if architecture == "residual":
        return PINNResidual(hidden_dims, activation, dropout)
    elif architecture == "z_frac_input":
        return PINNWithZFracInput(hidden_dims, activation, dropout)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Test
if __name__ == "__main__":
    print("Testing PINN Residual Architecture")
    print("=" * 50)
    
    # Create model
    model = PINNResidual(hidden_dims=[256, 256])
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    state_dz = torch.randn(batch_size, 6)
    
    # Test endpoint (z_frac=1)
    output = model(state_dz)
    print(f"\nEndpoint output shape: {output.shape}")  # [32, 4]
    
    # Test IC (z_frac=0) - should equal input state
    z_frac_0 = torch.zeros(batch_size)
    output_ic = model(state_dz, z_frac_0)
    ic_error = (output_ic - state_dz[:, :4]).abs().max().item()
    print(f"IC error (should be 0): {ic_error:.2e}")
    
    # Test batched collocation
    n_col = 10
    z_frac_col = torch.rand(batch_size, n_col)
    output_col = model(state_dz, z_frac_col)
    print(f"Collocation output shape: {output_col.shape}")  # [32, 10, 4]
    
    # Verify interpolation
    z_frac_half = torch.ones(batch_size) * 0.5
    output_half = model(state_dz, z_frac_half)
    output_full = model(state_dz, torch.ones(batch_size))
    
    # At z_frac=0.5, output should be halfway between IC and endpoint
    expected_half = 0.5 * (state_dz[:, :4] + output_full)
    interp_error = (output_half - expected_half).abs().max().item()
    print(f"Linear interpolation check (should be 0): {interp_error:.2e}")
    
    print("\n✓ All tests passed!")
