#!/usr/bin/env python3
"""
Train a TRUE Physics-Informed Neural Network (PINN) for track extrapolation.

This implements a proper PINN with physics loss that enforces the Lorentz force
equations during training. This is fundamentally different from the data-driven
MLP in train_pinn.py which only uses supervised loss.

Key differences from data-driven MLP:
1. Network predicts track state as a FUNCTION of z (not just final state)
2. Physics loss enforces d(state)/dz = f(state, B) at collocation points
3. Loss = L_data + λ * L_physics

The physics loss enforces:
    dx/dz = tx
    dy/dz = ty  
    dtx/dz = κ√(1+tx²+ty²) [tx·ty·Bx - (1+tx²)By + ty·Bz]
    dty/dz = κ√(1+tx²+ty²) [(1+ty²)Bx - tx·ty·By - tx·Bz]
    d(q/p)/dz = 0

where κ = c · (q/p) with c = 299.792458 mm/ns

Author: G. Scriven
Date: 2025-12-20
"""

import numpy as np
import struct
import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# LHCb Magnetic Field Model (differentiable)
# =============================================================================

class LHCbMagneticFieldTorch(nn.Module):
    """
    Differentiable LHCb dipole magnetic field model for use in PINN training.
    
    The LHCb magnet is a warm dipole with:
    - Main field component By (vertical)
    - Field integral ~4 Tm for full magnet
    - Field varies with position, strongest in center
    
    This is a simplified analytical model that is differentiable for backprop.
    """
    
    def __init__(self, polarity: int = 1):
        """
        Initialize field model.
        
        Args:
            polarity: +1 for MagUp, -1 for MagDown
        """
        super().__init__()
        self.polarity = polarity
        # Field parameters (approximate LHCb values)
        self.B0 = 1.0  # Tesla, peak field
        self.z_center = 5250.0  # mm, center of magnet
        self.z_halfwidth = 2500.0  # mm, half-width of field region
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get magnetic field at position (x, y, z).
        
        Args:
            x, y, z: Position tensors (can be batched)
            
        Returns:
            (Bx, By, Bz) in Tesla
        """
        # Gaussian-like z profile
        z_rel = (z - self.z_center) / self.z_halfwidth
        By_profile = torch.exp(-0.5 * z_rel**2)
        
        # Small x,y dependence (fringe fields)
        r_trans = torch.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        
        By = self.polarity * self.B0 * By_profile * fringe_factor
        
        # Small Bx component from field non-uniformity
        Bx = -0.01 * By * (x / 1000.0)
        
        # Bz is very small
        Bz = torch.zeros_like(x)
        
        return Bx, By, Bz


# =============================================================================
# True PINN Network Architecture
# =============================================================================

class TrackPINN(nn.Module):
    """
    True Physics-Informed Neural Network for track propagation.
    
    This network takes initial state + z and outputs the track state at z.
    The physics loss then enforces that the derivatives match the Lorentz force.
    
    Input: [x0, y0, tx0, ty0, qop, z] (6 parameters)
    Output: [x(z), y(z), tx(z), ty(z)] (4 parameters)
    
    For inference, we can evaluate at z=z_final to get the extrapolated state.
    During training, we evaluate at collocation points and compute physics loss.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dims: list = [128, 128, 64], 
                 output_dim: int = 4, activation: str = 'tanh'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sin':
                layers.append(SineActivation())  # Good for PINNs
            elif activation == 'swish':
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Tanh())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Normalization parameters
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
        
        # Initialize weights for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Xavier/Glorot for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with normalization.
        
        Args:
            x: Input tensor of shape (batch, 6) = [x0, y0, tx0, ty0, qop, z]
            
        Returns:
            Output tensor of shape (batch, 4) = [x, y, tx, ty] at z
        """
        # Normalize input
        x_norm = (x - self.input_mean) / self.input_std
        
        # Network forward
        out = self.network(x_norm)
        
        # Denormalize output
        return out * self.output_std + self.output_mean
    
    def forward_with_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns gradients w.r.t. z for physics loss.
        
        This is the key method for PINN - we need d(output)/dz to enforce physics.
        
        Args:
            x: Input tensor with requires_grad=True for z component
            
        Returns:
            output: [x, y, tx, ty] at z
            doutput_dz: [dx/dz, dy/dz, dtx/dz, dty/dz] derivatives
        """
        # Ensure z has gradient tracking
        x = x.clone()
        x.requires_grad_(True)
        
        output = self.forward(x)
        
        # Compute gradients of each output w.r.t. input (specifically z, which is input[:, 5])
        batch_size = x.shape[0]
        doutput_dz = torch.zeros_like(output)
        
        for i in range(self.output_dim):
            grad_outputs = torch.zeros_like(output)
            grad_outputs[:, i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            # z is the 6th input (index 5)
            doutput_dz[:, i] = grads[:, 5]
        
        return output, doutput_dz
    
    def save_binary(self, filepath: str):
        """Save model in binary format for C++ loading."""
        with open(filepath, 'wb') as f:
            # Count linear layers
            linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]
            f.write(struct.pack('i', len(linear_layers)))
            
            # Each layer
            for layer in linear_layers:
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                rows, cols = W.shape
                f.write(struct.pack('ii', rows, cols))
                f.write(W.astype(np.float64).tobytes())
                f.write(b.astype(np.float64).tobytes())
            
            # Input normalization
            input_mean = self.input_mean.cpu().numpy()
            input_std = self.input_std.cpu().numpy()
            f.write(struct.pack('i', len(input_mean)))
            f.write(input_mean.astype(np.float64).tobytes())
            f.write(input_std.astype(np.float64).tobytes())
            
            # Output normalization
            output_mean = self.output_mean.cpu().numpy()
            output_std = self.output_std.cpu().numpy()
            f.write(struct.pack('i', len(output_mean)))
            f.write(output_mean.astype(np.float64).tobytes())
            f.write(output_std.astype(np.float64).tobytes())
        
        print(f"Model saved to {filepath}")


class SineActivation(nn.Module):
    """Sine activation function - often better for PINNs."""
    def forward(self, x):
        return torch.sin(x)


# =============================================================================
# Physics Loss Computation
# =============================================================================

class PhysicsLoss(nn.Module):
    """
    Physics loss that enforces the Lorentz force equations.
    
    The Lorentz force equations in track parameters:
        dx/dz = tx
        dy/dz = ty
        dtx/dz = κ√(1+tx²+ty²) [tx·ty·Bx - (1+tx²)By + ty·Bz]
        dty/dz = κ√(1+tx²+ty²) [(1+ty²)Bx - tx·ty·By - tx·Bz]
        
    where κ = c · (q/p) with c = 299.792458 mm/ns
    """
    
    def __init__(self, field: LHCbMagneticFieldTorch):
        super().__init__()
        self.field = field
        self.c_light = 299.792458  # mm/ns
        
    def forward(self, x_in: torch.Tensor, state_pred: torch.Tensor, 
                dstate_dz_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics residuals.
        
        Args:
            x_in: Input tensor [x0, y0, tx0, ty0, qop, z] (batch, 6)
            state_pred: Predicted state [x, y, tx, ty] at z (batch, 4)
            dstate_dz_pred: Predicted derivatives [dx/dz, dy/dz, dtx/dz, dty/dz] (batch, 4)
            
        Returns:
            Dictionary with:
                - total: Total physics loss
                - position: Loss from position equations
                - slope: Loss from slope (momentum) equations
        """
        # Extract variables
        x0, y0, tx0, ty0, qop, z = x_in.unbind(dim=1)
        x, y, tx, ty = state_pred.unbind(dim=1)
        dx_dz, dy_dz, dtx_dz, dty_dz = dstate_dz_pred.unbind(dim=1)
        
        # Get magnetic field at current position
        Bx, By, Bz = self.field(x, y, z)
        
        # Compute κ = c · (q/p) (converting units appropriately)
        # qop is in 1/MeV, need to convert to 1/(GeV*mm/ns) for field in Tesla
        kappa = self.c_light * 1e-3 * qop  # Approximate unit conversion
        
        # Geometric factor
        norm = torch.sqrt(1.0 + tx**2 + ty**2)
        
        # Physics residuals
        # Position equations: dx/dz = tx, dy/dz = ty
        res_x = dx_dz - tx
        res_y = dy_dz - ty
        
        # Slope equations (Lorentz force)
        # dtx/dz = κ·norm·[tx·ty·Bx - (1+tx²)·By + ty·Bz]
        dtx_dz_physics = kappa * norm * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        res_tx = dtx_dz - dtx_dz_physics
        
        # dty/dz = κ·norm·[(1+ty²)·Bx - tx·ty·By - tx·Bz]
        dty_dz_physics = kappa * norm * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        res_ty = dty_dz - dty_dz_physics
        
        # Compute losses (MSE of residuals)
        loss_position = torch.mean(res_x**2 + res_y**2)
        loss_slope = torch.mean(res_tx**2 + res_ty**2)
        
        return {
            'total': loss_position + loss_slope,
            'position': loss_position,
            'slope': loss_slope,
            'residuals': {
                'x': torch.mean(res_x**2),
                'y': torch.mean(res_y**2),
                'tx': torch.mean(res_tx**2),
                'ty': torch.mean(res_ty**2)
            }
        }


# =============================================================================
# Training Functions
# =============================================================================

def generate_collocation_points(n_points: int, z_range: Tuple[float, float],
                                x_range: Tuple[float, float] = (-900, 900),
                                y_range: Tuple[float, float] = (-750, 750),
                                tx_range: Tuple[float, float] = (-0.3, 0.3),
                                ty_range: Tuple[float, float] = (-0.25, 0.25),
                                qop_range: Tuple[float, float] = (-4e-4, 4e-4),
                                seed: int = 42) -> torch.Tensor:
    """
    Generate collocation points for physics loss evaluation.
    
    These are points throughout the domain where we enforce the physics equations.
    
    Args:
        n_points: Number of collocation points
        z_range: Range of z values (start, end)
        *_range: Ranges for other parameters
        seed: Random seed
        
    Returns:
        Tensor of shape (n_points, 6) = [x0, y0, tx0, ty0, qop, z]
    """
    torch.manual_seed(seed)
    
    x0 = torch.rand(n_points) * (x_range[1] - x_range[0]) + x_range[0]
    y0 = torch.rand(n_points) * (y_range[1] - y_range[0]) + y_range[0]
    tx0 = torch.rand(n_points) * (tx_range[1] - tx_range[0]) + tx_range[0]
    ty0 = torch.rand(n_points) * (ty_range[1] - ty_range[0]) + ty_range[0]
    qop = torch.rand(n_points) * (qop_range[1] - qop_range[0]) + qop_range[0]
    z = torch.rand(n_points) * (z_range[1] - z_range[0]) + z_range[0]
    
    return torch.stack([x0, y0, tx0, ty0, qop, z], dim=1)


def load_mlp_training_data(data_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV file."""
    import csv
    
    X_list = []
    Y_list = []
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x0 = float(row['x_in'])
            y0 = float(row['y_in'])
            tx0 = float(row['tx_in'])
            ty0 = float(row['ty_in'])
            qop = float(row['qop'])
            dz = float(row['dz'])
            
            x_out = float(row['x_out'])
            y_out = float(row['y_out'])
            tx_out = float(row['tx_out'])
            ty_out = float(row['ty_out'])
            
            X_list.append([x0, y0, tx0, ty0, qop, dz])
            Y_list.append([x_out, y_out, tx_out, ty_out])
    
    return np.array(X_list), np.array(Y_list)


def train_true_pinn(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    z_start: float = 3000.0,
    z_end: float = 7000.0,
    hidden_dims: list = [128, 128, 64],
    epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 64,
    lambda_physics: float = 0.1,
    n_collocation: int = 5000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Tuple[TrackPINN, Dict]:
    """
    Train a true Physics-Informed Neural Network.
    
    Args:
        X_data: Input data [x0, y0, tx0, ty0, qop, dz] 
        Y_data: Output data [x_out, y_out, tx_out, ty_out] (at z=z_end)
        z_start, z_end: Z range for extrapolation
        hidden_dims: Hidden layer dimensions
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for data loss
        lambda_physics: Weight for physics loss
        n_collocation: Number of collocation points for physics loss
        device: Device to train on
        verbose: Print progress
        
    Returns:
        Trained model and training history
    """
    print("=" * 70)
    print("Training TRUE Physics-Informed Neural Network")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Lambda physics: {lambda_physics}")
    print(f"Collocation points: {n_collocation}")
    print()
    
    # Convert data - need to add z coordinate
    # For boundary conditions: at z=z_start, state = initial state
    # At z=z_end, state = final state (from data)
    X_start = np.hstack([X_data[:, :5], np.full((len(X_data), 1), z_start)])  # Initial conditions
    X_end = np.hstack([X_data[:, :5], np.full((len(X_data), 1), z_end)])      # Final z
    Y_end = Y_data  # Final state
    
    # Initial state (boundary condition at z=z_start)
    Y_start = X_data[:, :4]  # [x0, y0, tx0, ty0] at z=z_start
    
    # Convert to tensors
    X_start_t = torch.FloatTensor(X_start).to(device)
    Y_start_t = torch.FloatTensor(Y_start).to(device)
    X_end_t = torch.FloatTensor(X_end).to(device)
    Y_end_t = torch.FloatTensor(Y_end).to(device)
    
    # Create model
    model = TrackPINN(input_dim=6, hidden_dims=hidden_dims, output_dim=4)
    model.to(device)
    
    # Set normalization from combined data
    X_all = torch.cat([X_start_t, X_end_t], dim=0)
    Y_all = torch.cat([Y_start_t, Y_end_t], dim=0)
    model.input_mean = X_all.mean(dim=0)
    model.input_std = X_all.std(dim=0) + 1e-8
    model.output_mean = Y_all.mean(dim=0)
    model.output_std = Y_all.std(dim=0) + 1e-8
    
    # Create field model and physics loss
    field = LHCbMagneticFieldTorch(polarity=1)
    field.to(device)
    physics_loss_fn = PhysicsLoss(field)
    physics_loss_fn.to(device)
    
    # Create data loader for boundary conditions
    dataset_start = TensorDataset(X_start_t, Y_start_t)
    dataset_end = TensorDataset(X_end_t, Y_end_t)
    loader_start = DataLoader(dataset_start, batch_size=batch_size, shuffle=True)
    loader_end = DataLoader(dataset_end, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5, verbose=verbose)
    mse_loss = nn.MSELoss()
    
    # Training history
    history = {
        'total_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'boundary_loss': [],
        'learning_rate': []
    }
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0
        data_loss_epoch = 0
        physics_loss_epoch = 0
        boundary_loss_epoch = 0
        n_batches = 0
        
        # Iterate over boundary condition batches
        for (x_start_batch, y_start_batch), (x_end_batch, y_end_batch) in zip(loader_start, loader_end):
            optimizer.zero_grad()
            
            # ===== Data Loss (Boundary Conditions) =====
            # Initial boundary: at z=z_start, predict initial state
            pred_start = model(x_start_batch)
            loss_boundary = mse_loss(pred_start, y_start_batch)
            
            # Final boundary: at z=z_end, predict final state
            pred_end = model(x_end_batch)
            loss_data = mse_loss(pred_end, y_end_batch)
            
            # ===== Physics Loss (Collocation Points) =====
            # Generate collocation points for this batch
            n_coll = min(n_collocation // (len(loader_start)), batch_size * 2)
            collocation = generate_collocation_points(
                n_coll, (z_start, z_end), seed=epoch * 1000 + n_batches
            ).to(device)
            
            # Forward pass with gradients
            state_coll, dstate_dz_coll = model.forward_with_grad(collocation)
            
            # Physics loss
            physics_losses = physics_loss_fn(collocation, state_coll, dstate_dz_coll)
            loss_physics = physics_losses['total']
            
            # ===== Total Loss =====
            loss_total = loss_data + loss_boundary + lambda_physics * loss_physics
            
            # Backward and optimize
            loss_total.backward()
            optimizer.step()
            
            # Accumulate
            total_loss_epoch += loss_total.item()
            data_loss_epoch += loss_data.item()
            physics_loss_epoch += loss_physics.item()
            boundary_loss_epoch += loss_boundary.item()
            n_batches += 1
        
        # Average losses
        avg_total = total_loss_epoch / n_batches
        avg_data = data_loss_epoch / n_batches
        avg_physics = physics_loss_epoch / n_batches
        avg_boundary = boundary_loss_epoch / n_batches
        
        # Record history
        history['total_loss'].append(avg_total)
        history['data_loss'].append(avg_data)
        history['physics_loss'].append(avg_physics)
        history['boundary_loss'].append(avg_boundary)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step
        scheduler.step(avg_total)
        
        # Track best
        if avg_total < best_loss:
            best_loss = avg_total
        
        # Print progress
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}: "
                  f"Total={avg_total:.6f}, Data={avg_data:.6f}, "
                  f"Physics={avg_physics:.6f}, Boundary={avg_boundary:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nBest total loss: {best_loss:.6f}")
    
    return model, history


def evaluate_model(model: TrackPINN, X_test: np.ndarray, Y_test: np.ndarray, 
                   z_end: float = 7000.0, device: str = 'cpu') -> Dict:
    """Evaluate model on test data."""
    model.eval()
    model.to(device)
    
    # Prepare input (add z_end as z coordinate)
    X_eval = np.hstack([X_test[:, :5], np.full((len(X_test), 1), z_end)])
    X_t = torch.FloatTensor(X_eval).to(device)
    
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    
    errors = pred - Y_test
    
    results = {
        'mean_error_x': np.mean(errors[:, 0]),
        'mean_error_y': np.mean(errors[:, 1]),
        'std_error_x': np.std(errors[:, 0]),
        'std_error_y': np.std(errors[:, 1]),
        'mean_abs_error_x': np.mean(np.abs(errors[:, 0])),
        'mean_abs_error_y': np.mean(np.abs(errors[:, 1])),
        'mean_radial_error': np.mean(np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)),
        'max_radial_error': np.max(np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)),
        'predictions': pred,
        'errors': errors
    }
    
    return results


def save_config(filepath: str, model: TrackPINN, history: Dict, 
                training_params: Dict, results: Dict):
    """Save training configuration and results."""
    config = {
        "model_type": "PINN",
        "description": "True Physics-Informed Neural Network with Lorentz force loss",
        "date": datetime.now().isoformat(),
        "architecture": {
            "type": "PINN",
            "input_dim": model.input_dim,
            "hidden_dims": training_params.get('hidden_dims', [128, 128, 64]),
            "output_dim": model.output_dim,
            "activation": "tanh"
        },
        "training": {
            "epochs": training_params['epochs'],
            "batch_size": training_params['batch_size'],
            "learning_rate": training_params['lr'],
            "lambda_physics": training_params['lambda_physics'],
            "n_collocation": training_params['n_collocation'],
            "final_total_loss": history['total_loss'][-1],
            "final_data_loss": history['data_loss'][-1],
            "final_physics_loss": history['physics_loss'][-1]
        },
        "results": {
            "mean_radial_error_mm": results['mean_radial_error'],
            "max_radial_error_mm": results['max_radial_error'],
            "mean_abs_error_x_mm": results['mean_abs_error_x'],
            "mean_abs_error_y_mm": results['mean_abs_error_y']
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {filepath}")


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train True PINN for track extrapolation')
    parser.add_argument('--data', type=str, default=None, help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-physics', type=float, default=0.1, help='Physics loss weight')
    parser.add_argument('--n-collocation', type=int, default=5000, help='Number of collocation points')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--hidden-dims', type=str, default='128,128,64', help='Hidden dimensions')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "ml_models" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    print("=" * 70)
    print("TRUE PINN Training for Track Extrapolation")
    print("=" * 70)
    print()
    print("This is a TRUE Physics-Informed Neural Network!")
    print("Loss = L_data + λ × L_physics")
    print()
    print("Physics loss enforces Lorentz force equations:")
    print("  dx/dz = tx")
    print("  dy/dz = ty")
    print("  dtx/dz = κ√(1+tx²+ty²) [tx·ty·Bx - (1+tx²)By + ty·Bz]")
    print("  dty/dz = κ√(1+tx²+ty²) [(1+ty²)Bx - tx·ty·By - tx·Bz]")
    print()
    
    # Load or generate data
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        X, Y = load_mlp_training_data(args.data)
    else:
        print("Generating synthetic training data...")
        # Use the same data generation as the MLP for fair comparison
        from train_pinn import generate_ground_truth_data
        X, Y = generate_ground_truth_data(num_samples=5000, step_size=1.0)
    
    # Split
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Training parameters
    training_params = {
        'hidden_dims': hidden_dims,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': 64,
        'lambda_physics': args.lambda_physics,
        'n_collocation': args.n_collocation
    }
    
    # Train
    model, history = train_true_pinn(
        X_train, Y_train,
        **training_params,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation on Test Set")
    print("=" * 70)
    results = evaluate_model(model, X_test, Y_test)
    
    print(f"\nTest Results:")
    print(f"  Mean radial error:     {results['mean_radial_error']:.2f} mm")
    print(f"  Max radial error:      {results['max_radial_error']:.2f} mm")
    print(f"  Mean |error| in x:     {results['mean_abs_error_x']:.2f} mm")
    print(f"  Mean |error| in y:     {results['mean_abs_error_y']:.2f} mm")
    
    # Save model
    model_path = output_dir / "pinn_model_true.bin"
    model.save_binary(str(model_path))
    
    # Save config
    config_path = output_dir / "pinn_config.json"
    save_config(str(config_path), model, history, training_params, results)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Config saved to: {config_path}")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs_range = range(1, len(history['total_loss']) + 1)
        
        axes[0].semilogy(epochs_range, history['total_loss'], label='Total')
        axes[0].semilogy(epochs_range, history['data_loss'], label='Data')
        axes[0].semilogy(epochs_range, history['physics_loss'], label='Physics')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(epochs_range, history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        
        # Error distribution
        radial_errors = np.sqrt(results['errors'][:, 0]**2 + results['errors'][:, 1]**2)
        axes[2].hist(radial_errors, bins=50, edgecolor='black')
        axes[2].axvline(results['mean_radial_error'], color='r', linestyle='--', 
                       label=f'Mean: {results["mean_radial_error"]:.2f} mm')
        axes[2].set_xlabel('Radial Error (mm)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Error Distribution')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plot_path = output_dir.parent / "plots" / "pinn_training_history.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        print(f"Training plot saved to: {plot_path}")
        plt.close()
        
    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    main()
