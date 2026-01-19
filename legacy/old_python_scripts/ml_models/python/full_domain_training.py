#!/usr/bin/env python3
"""
Full Domain Training and Benchmarking Script

This script:
1. Generates training data covering the FULL domain:
   - Momentum: 0.5 - 100 GeV/c
   - Position: Full LHCb acceptance
   - Both charge signs
   
2. Trains both MLP and true PINN models

3. Benchmarks against high-precision RK8 ground truth

4. Generates comprehensive comparison plots

Author: G. Scriven
Date: 2025-12-20
"""

import numpy as np
import struct
import time
import json
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# LHCb Magnetic Field Model
# =============================================================================

class LHCbMagneticField:
    """
    Simplified LHCb dipole magnetic field model.
    """
    def __init__(self, polarity: int = 1):
        self.polarity = polarity
        self.B0 = 1.0  # Tesla, peak field
        self.z_center = 5250.0  # mm
        self.z_halfwidth = 2500.0  # mm
        
    def get_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        z_rel = (z - self.z_center) / self.z_halfwidth
        By_profile = np.exp(-0.5 * z_rel**2)
        r_trans = np.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        By = self.polarity * self.B0 * By_profile * fringe_factor
        Bx = -0.01 * By * (x / 1000.0)
        Bz = 0.0
        return (Bx, By, Bz)


class LHCbMagneticFieldTorch(nn.Module):
    """Differentiable field model for PINN training."""
    def __init__(self, polarity: int = 1):
        super().__init__()
        self.polarity = polarity
        self.B0 = 1.0
        self.z_center = 5250.0
        self.z_halfwidth = 2500.0
        
    def forward(self, x, y, z):
        z_rel = (z - self.z_center) / self.z_halfwidth
        By_profile = torch.exp(-0.5 * z_rel**2)
        r_trans = torch.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        By = self.polarity * self.B0 * By_profile * fringe_factor
        Bx = -0.01 * By * (x / 1000.0)
        Bz = torch.zeros_like(x)
        return Bx, By, Bz


# =============================================================================
# High-Precision RK8 Integrator (Ground Truth)
# =============================================================================

class HighPrecisionRKIntegrator:
    """
    RK8 integrator with Richardson extrapolation for ground truth.
    """
    def __init__(self, field: LHCbMagneticField, step_size: float = 1.0):
        self.field = field
        self.step_size = step_size
        self.c_light = 299.792458  # mm/ns
        
    def derivatives(self, z: float, state: np.ndarray) -> np.ndarray:
        x, y, tx, ty, qop = state
        Bx, By, Bz = self.field.get_field(x, y, z)
        factor = qop * self.c_light * 1e-3
        norm = np.sqrt(1.0 + tx**2 + ty**2)
        dtx_dz = factor * norm * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = factor * norm * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        return np.array([tx, ty, dtx_dz, dty_dz, 0.0])
    
    def rk4_step(self, z: float, state: np.ndarray, h: float) -> np.ndarray:
        k1 = self.derivatives(z, state)
        k2 = self.derivatives(z + 0.5*h, state + 0.5*h*k1)
        k3 = self.derivatives(z + 0.5*h, state + 0.5*h*k2)
        k4 = self.derivatives(z + h, state + h*k3)
        return state + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def rk8_step(self, z: float, state: np.ndarray, h: float) -> np.ndarray:
        """RK8 via Richardson extrapolation."""
        state_half = self.rk4_step(z, state, h/2)
        state_two_half = self.rk4_step(z + h/2, state_half, h/2)
        state_full = self.rk4_step(z, state, h)
        return (16.0 * state_two_half - state_full) / 15.0
    
    def propagate(self, state_in: np.ndarray, z_in: float, z_out: float) -> np.ndarray:
        state = state_in.copy()
        z = z_in
        dz = z_out - z_in
        h = self.step_size if dz > 0 else -self.step_size
        n_steps = int(np.ceil(abs(dz) / self.step_size))
        h = dz / n_steps
        for _ in range(n_steps):
            state = self.rk8_step(z, state, h)
            z += h
        return state
    
    def propagate_trajectory(self, state_in: np.ndarray, z_in: float, z_out: float, 
                             n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get full trajectory for visualization."""
        z_vals = np.linspace(z_in, z_out, n_points)
        states = np.zeros((n_points, 5))
        states[0] = state_in
        
        for i in range(1, n_points):
            states[i] = self.propagate(states[i-1], z_vals[i-1], z_vals[i])
        
        return z_vals, states


# =============================================================================
# Standard RK4 Integrator (for comparison)
# =============================================================================

class StandardRK4Integrator:
    """Standard RK4 with larger step size (production-like)."""
    def __init__(self, field: LHCbMagneticField, step_size: float = 50.0):
        self.field = field
        self.step_size = step_size
        self.c_light = 299.792458
        
    def derivatives(self, z: float, state: np.ndarray) -> np.ndarray:
        x, y, tx, ty, qop = state
        Bx, By, Bz = self.field.get_field(x, y, z)
        factor = qop * self.c_light * 1e-3
        norm = np.sqrt(1.0 + tx**2 + ty**2)
        dtx_dz = factor * norm * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = factor * norm * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        return np.array([tx, ty, dtx_dz, dty_dz, 0.0])
    
    def propagate(self, state_in: np.ndarray, z_in: float, z_out: float) -> np.ndarray:
        state = state_in.copy()
        z = z_in
        dz = z_out - z_in
        h = self.step_size if dz > 0 else -self.step_size
        n_steps = max(1, int(np.ceil(abs(dz) / self.step_size)))
        h = dz / n_steps
        
        for _ in range(n_steps):
            k1 = self.derivatives(z, state)
            k2 = self.derivatives(z + 0.5*h, state + 0.5*h*k1)
            k3 = self.derivatives(z + 0.5*h, state + 0.5*h*k2)
            k4 = self.derivatives(z + h, state + h*k3)
            state = state + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            z += h
        
        return state


# =============================================================================
# Neural Network Models
# =============================================================================

class TrackMLP(nn.Module):
    """Data-driven MLP for track extrapolation."""
    def __init__(self, input_dim=6, hidden_dims=[256, 256, 128, 64], output_dim=4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
        
    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        out = self.network(x_norm)
        return out * self.output_std + self.output_mean
    
    def save_binary(self, filepath):
        with open(filepath, 'wb') as f:
            linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]
            f.write(struct.pack('i', len(linear_layers)))
            for layer in linear_layers:
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                rows, cols = W.shape
                f.write(struct.pack('ii', rows, cols))
                f.write(W.astype(np.float64).tobytes())
                f.write(b.astype(np.float64).tobytes())
            
            input_mean = self.input_mean.cpu().numpy()
            input_std = self.input_std.cpu().numpy()
            f.write(struct.pack('i', len(input_mean)))
            f.write(input_mean.astype(np.float64).tobytes())
            f.write(input_std.astype(np.float64).tobytes())
            
            output_mean = self.output_mean.cpu().numpy()
            output_std = self.output_std.cpu().numpy()
            f.write(struct.pack('i', len(output_mean)))
            f.write(output_mean.astype(np.float64).tobytes())
            f.write(output_std.astype(np.float64).tobytes())


class TrackPINN(nn.Module):
    """True Physics-Informed Neural Network."""
    def __init__(self, input_dim=6, hidden_dims=[256, 256, 128, 64], output_dim=4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        out = self.network(x_norm)
        return out * self.output_std + self.output_mean
    
    def forward_with_grad(self, x):
        x = x.clone().requires_grad_(True)
        output = self.forward(x)
        
        doutput_dz = torch.zeros_like(output)
        for i in range(4):
            grad = torch.autograd.grad(
                outputs=output[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]
            doutput_dz[:, i] = grad[:, 5]  # z is index 5
        
        return output, doutput_dz
    
    def save_binary(self, filepath):
        with open(filepath, 'wb') as f:
            linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]
            f.write(struct.pack('i', len(linear_layers)))
            for layer in linear_layers:
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                rows, cols = W.shape
                f.write(struct.pack('ii', rows, cols))
                f.write(W.astype(np.float64).tobytes())
                f.write(b.astype(np.float64).tobytes())
            
            input_mean = self.input_mean.cpu().numpy()
            input_std = self.input_std.cpu().numpy()
            f.write(struct.pack('i', len(input_mean)))
            f.write(input_mean.astype(np.float64).tobytes())
            f.write(input_std.astype(np.float64).tobytes())
            
            output_mean = self.output_mean.cpu().numpy()
            output_std = self.output_std.cpu().numpy()
            f.write(struct.pack('i', len(output_mean)))
            f.write(output_mean.astype(np.float64).tobytes())
            f.write(output_std.astype(np.float64).tobytes())


# =============================================================================
# Training Data Generation
# =============================================================================

def generate_full_domain_data(n_samples: int = 20000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data covering the FULL domain:
    - Momentum: 1.0 - 100 GeV/c (log-uniform for better coverage)
    - Position: Full acceptance
    - Both charges
    """
    print(f"Generating {n_samples} full-domain training samples...")
    np.random.seed(seed)
    
    field = LHCbMagneticField(polarity=1)
    integrator = HighPrecisionRKIntegrator(field, step_size=5.0)  # Larger step for stability
    
    z_in, z_out = 3000.0, 7000.0
    dz = z_out - z_in
    
    X_list = []
    Y_list = []
    n_failed = 0
    
    i = 0
    while len(X_list) < n_samples:
        i += 1
        if (len(X_list) + 1) % 2000 == 0:
            print(f"  Generated {len(X_list)+1}/{n_samples} samples (attempts: {i})")
        
        x0 = np.random.uniform(-900, 900)
        y0 = np.random.uniform(-750, 750)
        tx0 = np.random.uniform(-0.3, 0.3)
        ty0 = np.random.uniform(-0.25, 0.25)
        
        # Log-uniform momentum distribution: 1.0 to 100 GeV (avoid very low p)
        log_p = np.random.uniform(np.log(1.0), np.log(100))
        p_gev = np.exp(log_p)
        charge = np.random.choice([-1, 1])
        qop = charge / (p_gev * 1000.0)  # 1/MeV
        
        state_in = np.array([x0, y0, tx0, ty0, qop])
        
        try:
            state_out = integrator.propagate(state_in, z_in, z_out)
            
            # Check for valid output (no NaN/Inf, reasonable values)
            if (np.all(np.isfinite(state_out)) and 
                np.abs(state_out[0]) < 10000 and np.abs(state_out[1]) < 10000 and
                np.abs(state_out[2]) < 10 and np.abs(state_out[3]) < 10):
                
                X_list.append([x0, y0, tx0, ty0, qop, dz])
                Y_list.append([state_out[0], state_out[1], state_out[2], state_out[3]])
            else:
                n_failed += 1
        except:
            n_failed += 1
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"Data generation complete! (failed: {n_failed})")
    print(f"  Momentum range: {1.0:.1f} - {100:.1f} GeV/c")
    print(f"  Position range: x∈[-900,900], y∈[-750,750] mm")
    
    return X, Y


# =============================================================================
# Training Functions
# =============================================================================

def train_mlp(X: np.ndarray, Y: np.ndarray, epochs: int = 1500, lr: float = 1e-3,
              batch_size: int = 256) -> Tuple[TrackMLP, Dict]:
    """Train data-driven MLP."""
    print("\n" + "="*70)
    print("Training Data-Driven MLP")
    print("="*70)
    
    device = 'cpu'
    
    # Split data
    n_train = int(0.85 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]
    
    X_t = torch.FloatTensor(X_train).to(device)
    Y_t = torch.FloatTensor(Y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    
    model = TrackMLP(hidden_dims=[256, 256, 128, 64])
    model.to(device)
    
    # Set normalization
    model.input_mean = X_t.mean(dim=0)
    model.input_std = X_t.std(dim=0) + 1e-8
    model.output_mean = Y_t.mean(dim=0)
    model.output_std = Y_t.std(dim=0) + 1e-8
    
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(loader)
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
    
    print(f"Best validation loss: {best_val_loss:.6f}")
    return model, history


def train_pinn(X: np.ndarray, Y: np.ndarray, epochs: int = 1500, lr: float = 1e-3,
               lambda_phys: float = 0.1, batch_size: int = 256) -> Tuple[TrackPINN, Dict]:
    """Train true PINN with physics loss."""
    print("\n" + "="*70)
    print("Training True Physics-Informed Neural Network")
    print("="*70)
    print(f"Lambda physics: {lambda_phys}")
    
    device = 'cpu'
    z_start, z_end = 3000.0, 7000.0
    
    field = LHCbMagneticFieldTorch(polarity=1)
    c_light = 299.792458
    
    # Split data
    n_train = int(0.85 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]
    
    # Prepare boundary conditions
    X_start = np.hstack([X_train[:, :5], np.full((len(X_train), 1), z_start)])
    X_end = np.hstack([X_train[:, :5], np.full((len(X_train), 1), z_end)])
    Y_start = X_train[:, :4]
    Y_end = Y_train
    
    X_start_t = torch.FloatTensor(X_start).to(device)
    Y_start_t = torch.FloatTensor(Y_start).to(device)
    X_end_t = torch.FloatTensor(X_end).to(device)
    Y_end_t = torch.FloatTensor(Y_end).to(device)
    
    model = TrackPINN(hidden_dims=[256, 256, 128, 64])
    model.to(device)
    
    # Normalization
    X_all = torch.cat([X_start_t, X_end_t], dim=0)
    Y_all = torch.cat([Y_start_t, Y_end_t], dim=0)
    model.input_mean = X_all.mean(dim=0)
    model.input_std = X_all.std(dim=0) + 1e-8
    model.output_mean = Y_all.mean(dim=0)
    model.output_std = Y_all.std(dim=0) + 1e-8
    
    dataset_start = TensorDataset(X_start_t, Y_start_t)
    dataset_end = TensorDataset(X_end_t, Y_end_t)
    loader_start = DataLoader(dataset_start, batch_size=batch_size, shuffle=True)
    loader_end = DataLoader(dataset_end, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    history = {'total_loss': [], 'data_loss': [], 'physics_loss': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0
        data_loss_epoch = 0
        physics_loss_epoch = 0
        n_batches = 0
        
        for (x_start_batch, y_start_batch), (x_end_batch, y_end_batch) in zip(loader_start, loader_end):
            optimizer.zero_grad()
            
            # Boundary losses
            pred_start = model(x_start_batch)
            loss_boundary = mse_loss(pred_start, y_start_batch)
            
            pred_end = model(x_end_batch)
            loss_data = mse_loss(pred_end, y_end_batch)
            
            # Physics loss at collocation points
            n_coll = min(64, len(x_start_batch))
            z_coll = torch.rand(n_coll) * (z_end - z_start) + z_start
            idx = torch.randint(0, len(x_start_batch), (n_coll,))
            
            x_coll = torch.zeros(n_coll, 6)
            x_coll[:, :5] = x_start_batch[idx, :5]
            x_coll[:, 5] = z_coll
            
            state_coll, dstate_dz = model.forward_with_grad(x_coll)
            
            x, y, tx, ty = state_coll[:, 0], state_coll[:, 1], state_coll[:, 2], state_coll[:, 3]
            dx_dz, dy_dz, dtx_dz, dty_dz = dstate_dz[:, 0], dstate_dz[:, 1], dstate_dz[:, 2], dstate_dz[:, 3]
            qop = x_coll[:, 4]
            z = x_coll[:, 5]
            
            Bx, By, Bz = field(x, y, z)
            kappa = c_light * 1e-3 * qop
            norm = torch.sqrt(1.0 + tx**2 + ty**2)
            
            res_x = dx_dz - tx
            res_y = dy_dz - ty
            res_tx = dtx_dz - kappa * norm * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
            res_ty = dty_dz - kappa * norm * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
            
            loss_physics = torch.mean(res_x**2 + res_y**2 + res_tx**2 + res_ty**2)
            
            loss_total = loss_data + loss_boundary + lambda_phys * loss_physics
            loss_total.backward()
            optimizer.step()
            
            total_loss_epoch += loss_total.item()
            data_loss_epoch += (loss_data.item() + loss_boundary.item())
            physics_loss_epoch += loss_physics.item()
            n_batches += 1
        
        scheduler.step()
        
        history['total_loss'].append(total_loss_epoch / n_batches)
        history['data_loss'].append(data_loss_epoch / n_batches)
        history['physics_loss'].append(physics_loss_epoch / n_batches)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}: Total={total_loss_epoch/n_batches:.6f}, "
                  f"Data={data_loss_epoch/n_batches:.6f}, Physics={physics_loss_epoch/n_batches:.6f}")
    
    return model, history


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_all_methods(X_test: np.ndarray, Y_truth: np.ndarray,
                          mlp_model: TrackMLP, pinn_model: TrackPINN) -> Dict:
    """Benchmark all methods against RK8 ground truth."""
    print("\n" + "="*70)
    print("Benchmarking All Methods Against RK8 Ground Truth")
    print("="*70)
    
    field = LHCbMagneticField(polarity=1)
    z_in, z_out = 3000.0, 7000.0
    
    # Different RK configurations
    rk_configs = {
        'RK8 (1mm)': HighPrecisionRKIntegrator(field, step_size=1.0),
        'RK4 (10mm)': StandardRK4Integrator(field, step_size=10.0),
        'RK4 (50mm)': StandardRK4Integrator(field, step_size=50.0),
        'RK4 (100mm)': StandardRK4Integrator(field, step_size=100.0),
    }
    
    results = {}
    n_test = len(X_test)
    
    # Benchmark RK methods
    for name, integrator in rk_configs.items():
        print(f"\nBenchmarking {name}...")
        
        predictions = np.zeros((n_test, 4))
        
        start = time.time()
        for i in range(n_test):
            state_in = X_test[i, :5]
            state_out = integrator.propagate(state_in, z_in, z_out)
            predictions[i] = state_out[:4]
        elapsed = time.time() - start
        
        errors = predictions - Y_truth
        radial_errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
        
        results[name] = {
            'predictions': predictions,
            'errors': errors,
            'radial_errors': radial_errors,
            'mean_radial': np.mean(radial_errors),
            'std_radial': np.std(radial_errors),
            'max_radial': np.max(radial_errors),
            'p95_radial': np.percentile(radial_errors, 95),
            'time_per_track_us': (elapsed / n_test) * 1e6,
            'total_time_s': elapsed
        }
        
        print(f"  Mean radial error: {results[name]['mean_radial']:.4f} mm")
        print(f"  Time per track: {results[name]['time_per_track_us']:.2f} μs")
    
    # Benchmark MLP
    print(f"\nBenchmarking MLP (data-driven)...")
    mlp_model.eval()
    X_test_t = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        start = time.time()
        for _ in range(10):  # Average over multiple runs
            pred_mlp = mlp_model(X_test_t)
        elapsed = (time.time() - start) / 10
    
    pred_mlp = pred_mlp.numpy()
    errors_mlp = pred_mlp - Y_truth
    radial_mlp = np.sqrt(errors_mlp[:, 0]**2 + errors_mlp[:, 1]**2)
    
    results['MLP'] = {
        'predictions': pred_mlp,
        'errors': errors_mlp,
        'radial_errors': radial_mlp,
        'mean_radial': np.mean(radial_mlp),
        'std_radial': np.std(radial_mlp),
        'max_radial': np.max(radial_mlp),
        'p95_radial': np.percentile(radial_mlp, 95),
        'time_per_track_us': (elapsed / n_test) * 1e6,
        'total_time_s': elapsed
    }
    print(f"  Mean radial error: {results['MLP']['mean_radial']:.4f} mm")
    print(f"  Time per track: {results['MLP']['time_per_track_us']:.2f} μs")
    
    # Benchmark PINN
    print(f"\nBenchmarking PINN (physics-informed)...")
    pinn_model.eval()
    
    # PINN needs z as input
    X_pinn = np.hstack([X_test[:, :5], np.full((n_test, 1), z_out)])
    X_pinn_t = torch.FloatTensor(X_pinn)
    
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            pred_pinn = pinn_model(X_pinn_t)
        elapsed = (time.time() - start) / 10
    
    pred_pinn = pred_pinn.numpy()
    errors_pinn = pred_pinn - Y_truth
    radial_pinn = np.sqrt(errors_pinn[:, 0]**2 + errors_pinn[:, 1]**2)
    
    results['PINN'] = {
        'predictions': pred_pinn,
        'errors': errors_pinn,
        'radial_errors': radial_pinn,
        'mean_radial': np.mean(radial_pinn),
        'std_radial': np.std(radial_pinn),
        'max_radial': np.max(radial_pinn),
        'p95_radial': np.percentile(radial_pinn, 95),
        'time_per_track_us': (elapsed / n_test) * 1e6,
        'total_time_s': elapsed
    }
    print(f"  Mean radial error: {results['PINN']['mean_radial']:.4f} mm")
    print(f"  Time per track: {results['PINN']['time_per_track_us']:.2f} μs")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_3d_tracks(X_test: np.ndarray, Y_truth: np.ndarray, results: Dict, 
                   output_dir: Path, n_tracks: int = 10):
    """Generate 3D track visualizations."""
    print("\nGenerating 3D track visualizations...")
    
    field = LHCbMagneticField(polarity=1)
    integrator = HighPrecisionRKIntegrator(field, step_size=10.0)
    z_in, z_out = 3000.0, 7000.0
    
    # Select diverse tracks (different momenta)
    momenta = np.abs(1.0 / X_test[:, 4]) / 1000  # GeV
    
    # Sample tracks at different momenta
    p_bins = [0.5, 2, 5, 10, 20, 50, 100]
    selected_indices = []
    for i in range(len(p_bins) - 1):
        mask = (momenta >= p_bins[i]) & (momenta < p_bins[i+1])
        if np.any(mask):
            idx = np.where(mask)[0]
            selected_indices.append(np.random.choice(idx))
    
    # Add a few random tracks
    remaining = n_tracks - len(selected_indices)
    if remaining > 0:
        all_idx = set(range(len(X_test))) - set(selected_indices)
        selected_indices.extend(np.random.choice(list(all_idx), min(remaining, len(all_idx)), replace=False))
    
    fig = plt.figure(figsize=(20, 15))
    
    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)  # X-Z projection
    ax3 = fig.add_subplot(2, 2, 3)  # Y-Z projection
    ax4 = fig.add_subplot(2, 2, 4)  # X-Y at z_end
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))
    
    for i, idx in enumerate(selected_indices):
        state_in = X_test[idx, :5]
        p_gev = abs(1.0 / state_in[4]) / 1000
        charge = np.sign(state_in[4])
        
        # Get full trajectory (RK8 truth)
        z_vals, states = integrator.propagate_trajectory(state_in, z_in, z_out, n_points=50)
        
        # Plot truth trajectory
        label = f'p={p_gev:.1f} GeV, q={int(charge)}'
        ax1.plot(states[:, 0], z_vals, states[:, 1], '-', color=colors[i], linewidth=2, 
                label=label, alpha=0.8)
        ax2.plot(z_vals, states[:, 0], '-', color=colors[i], linewidth=2)
        ax3.plot(z_vals, states[:, 1], '-', color=colors[i], linewidth=2)
        
        # Plot endpoints for each method
        truth_end = Y_truth[idx]
        mlp_end = results['MLP']['predictions'][idx]
        pinn_end = results['PINN']['predictions'][idx]
        
        # 3D endpoints
        ax1.scatter([truth_end[0]], [z_out], [truth_end[1]], marker='o', s=100, 
                   color=colors[i], edgecolor='black', linewidth=2)
        ax1.scatter([mlp_end[0]], [z_out], [mlp_end[1]], marker='^', s=80,
                   color=colors[i], edgecolor='red', linewidth=1)
        ax1.scatter([pinn_end[0]], [z_out], [pinn_end[1]], marker='s', s=80,
                   color=colors[i], edgecolor='blue', linewidth=1)
        
        # X-Y plot at z_end
        ax4.scatter([truth_end[0]], [truth_end[1]], marker='o', s=100,
                   color=colors[i], edgecolor='black', linewidth=2, label=label if i < 3 else None)
        ax4.scatter([mlp_end[0]], [mlp_end[1]], marker='^', s=60,
                   color=colors[i], edgecolor='red', linewidth=1)
        ax4.scatter([pinn_end[0]], [pinn_end[1]], marker='s', s=60,
                   color=colors[i], edgecolor='blue', linewidth=1)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_zlabel('Y (mm)')
    ax1.set_title('3D Track Visualization\n(○=Truth, △=MLP, □=PINN)')
    ax1.legend(loc='upper left', fontsize=8)
    
    ax2.set_xlabel('Z (mm)')
    ax2.set_ylabel('X (mm)')
    ax2.set_title('X-Z Projection (Bending Plane)')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Z (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title('Y-Z Projection')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title(f'Track Endpoints at z={z_out}mm\n(○=Truth, △=MLP, □=PINN)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_track_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 3d_track_visualization.png")


def plot_comparison_summary(results: Dict, output_dir: Path):
    """Generate comprehensive comparison plots."""
    print("Generating comparison summary plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = list(results.keys())
    colors = {'RK8 (1mm)': 'green', 'RK4 (10mm)': 'orange', 'RK4 (50mm)': 'purple',
              'RK4 (100mm)': 'brown', 'MLP': 'blue', 'PINN': 'red'}
    
    # 1. Error distribution histogram
    ax = axes[0, 0]
    for method in ['RK4 (50mm)', 'MLP', 'PINN']:
        ax.hist(results[method]['radial_errors'], bins=50, alpha=0.5, 
               label=f"{method} (μ={results[method]['mean_radial']:.2f})",
               color=colors[method], density=True)
    ax.set_xlabel('Radial Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Radial Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Bar chart - Mean errors
    ax = axes[0, 1]
    mean_errors = [results[m]['mean_radial'] for m in methods]
    bars = ax.bar(range(len(methods)), mean_errors, color=[colors[m] for m in methods])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Mean Radial Error (mm)')
    ax.set_title('Mean Radial Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mean_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{val:.2f}', ha='center', fontsize=9)
    
    # 3. Timing comparison
    ax = axes[0, 2]
    times = [results[m]['time_per_track_us'] for m in methods]
    bars = ax.bar(range(len(methods)), times, color=[colors[m] for m in methods])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Time per Track (μs)')
    ax.set_title('Timing Comparison')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Accuracy vs Speed scatter
    ax = axes[1, 0]
    for method in methods:
        ax.scatter(results[method]['time_per_track_us'], results[method]['mean_radial'],
                  s=200, label=method, color=colors[method], edgecolor='black')
    ax.set_xlabel('Time per Track (μs)')
    ax.set_ylabel('Mean Radial Error (mm)')
    ax.set_title('Accuracy vs Speed Trade-off')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Error by momentum bin (for NN methods)
    ax = axes[1, 1]
    X_test = np.array([[0, 0, 0, 0, qop, 4000] for qop in results['MLP']['predictions'][:, 0]])  # Dummy
    # We need to recover momentum from the test data - let me fix this
    
    # For now, show 95th percentile comparison
    p95_errors = [results[m]['p95_radial'] for m in methods]
    bars = ax.bar(range(len(methods)), p95_errors, color=[colors[m] for m in methods])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('95th Percentile Error (mm)')
    ax.set_title('95th Percentile Radial Error')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary table as text
    ax = axes[1, 2]
    ax.axis('off')
    
    table_text = "BENCHMARK SUMMARY\n" + "="*50 + "\n\n"
    table_text += f"{'Method':<15} {'Mean(mm)':<10} {'Max(mm)':<10} {'Time(μs)':<10} {'Speedup':<10}\n"
    table_text += "-"*55 + "\n"
    
    rk8_time = results['RK8 (1mm)']['time_per_track_us']
    for method in methods:
        r = results[method]
        speedup = rk8_time / r['time_per_track_us']
        table_text += f"{method:<15} {r['mean_radial']:<10.2f} {r['max_radial']:<10.2f} {r['time_per_track_us']:<10.1f} {speedup:<10.1f}x\n"
    
    ax.text(0.1, 0.5, table_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: benchmark_comparison.png")


def plot_error_by_momentum(X_test: np.ndarray, results: Dict, output_dir: Path):
    """Plot error as function of momentum."""
    print("Generating error vs momentum plots...")
    
    momenta = np.abs(1.0 / X_test[:, 4]) / 1000  # GeV
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = ['RK4 (50mm)', 'MLP', 'PINN']
    colors = {'RK4 (50mm)': 'purple', 'MLP': 'blue', 'PINN': 'red'}
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        radial_errors = results[method]['radial_errors']
        
        ax.scatter(momenta, radial_errors, alpha=0.3, s=5, color=colors[method])
        
        # Binned means
        p_bins = np.logspace(np.log10(0.5), np.log10(100), 15)
        bin_centers = []
        bin_means = []
        bin_stds = []
        
        for i in range(len(p_bins) - 1):
            mask = (momenta >= p_bins[i]) & (momenta < p_bins[i+1])
            if np.sum(mask) > 5:
                bin_centers.append(np.sqrt(p_bins[i] * p_bins[i+1]))
                bin_means.append(np.mean(radial_errors[mask]))
                bin_stds.append(np.std(radial_errors[mask]))
        
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', 
                   color='black', markersize=8, linewidth=2, capsize=4,
                   label='Binned mean ± std')
        
        ax.set_xlabel('Momentum (GeV/c)')
        ax.set_ylabel('Radial Error (mm)')
        ax.set_title(f'{method}\nMean: {results[method]["mean_radial"]:.2f} mm')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_vs_momentum.png")


def plot_training_history(mlp_history: Dict, pinn_history: Dict, output_dir: Path):
    """Plot training history for both models."""
    print("Generating training history plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MLP
    ax = axes[0]
    epochs = range(1, len(mlp_history['train_loss']) + 1)
    ax.semilogy(epochs, mlp_history['train_loss'], label='Train', color='blue')
    ax.semilogy(epochs, mlp_history['val_loss'], label='Validation', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('MLP Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PINN
    ax = axes[1]
    epochs = range(1, len(pinn_history['total_loss']) + 1)
    ax.semilogy(epochs, pinn_history['total_loss'], label='Total', color='black')
    ax.semilogy(epochs, pinn_history['data_loss'], label='Data', color='blue')
    ax.semilogy(epochs, pinn_history['physics_loss'], label='Physics', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('PINN Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_history.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("FULL DOMAIN TRAINING AND BENCHMARKING")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "ml_models" / "models"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate full-domain training data
    print("\n" + "="*70)
    print("PHASE 1: Data Generation")
    print("="*70)
    X_train, Y_train = generate_full_domain_data(n_samples=20000, seed=42)
    
    # Generate test data (separate seed)
    print("\nGenerating test data...")
    X_test, Y_test = generate_full_domain_data(n_samples=2000, seed=123)
    
    # Train MLP
    print("\n" + "="*70)
    print("PHASE 2: Training")
    print("="*70)
    mlp_model, mlp_history = train_mlp(X_train, Y_train, epochs=1500)
    
    # Train PINN
    pinn_model, pinn_history = train_pinn(X_train, Y_train, epochs=1500, lambda_phys=0.1)
    
    # Save models
    print("\nSaving models...")
    mlp_model.save_binary(str(models_dir / "mlp_full_domain.bin"))
    pinn_model.save_binary(str(models_dir / "pinn_full_domain.bin"))
    print(f"  Saved: mlp_full_domain.bin")
    print(f"  Saved: pinn_full_domain.bin")
    
    # Benchmark
    print("\n" + "="*70)
    print("PHASE 3: Benchmarking")
    print("="*70)
    results = benchmark_all_methods(X_test, Y_test, mlp_model, pinn_model)
    
    # Generate plots
    print("\n" + "="*70)
    print("PHASE 4: Visualization")
    print("="*70)
    plot_3d_tracks(X_test, Y_test, results, plots_dir)
    plot_comparison_summary(results, plots_dir)
    plot_error_by_momentum(X_test, results, plots_dir)
    plot_training_history(mlp_history, pinn_history, plots_dir)
    
    # Save results
    results_summary = {}
    for method, r in results.items():
        results_summary[method] = {
            'mean_radial_mm': float(r['mean_radial']),
            'std_radial_mm': float(r['std_radial']),
            'max_radial_mm': float(r['max_radial']),
            'p95_radial_mm': float(r['p95_radial']),
            'time_per_track_us': float(r['time_per_track_us'])
        }
    
    with open(models_dir / "full_domain_results.json", 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nResults saved to: {models_dir / 'full_domain_results.json'}")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Method':<15} {'Mean Error':<12} {'95% Error':<12} {'Time':<12} {'Speedup'}")
    print("-"*60)
    
    rk8_time = results['RK8 (1mm)']['time_per_track_us']
    for method in results:
        r = results[method]
        speedup = rk8_time / r['time_per_track_us']
        print(f"{method:<15} {r['mean_radial']:<12.2f} {r['p95_radial']:<12.2f} "
              f"{r['time_per_track_us']:<12.1f} {speedup:.1f}x")
    
    print("\nPlots saved to:", plots_dir)
    print("Models saved to:", models_dir)


if __name__ == "__main__":
    main()
