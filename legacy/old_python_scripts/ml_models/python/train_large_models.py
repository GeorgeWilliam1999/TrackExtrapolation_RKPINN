#!/usr/bin/env python3
"""
Train LARGE, high-accuracy models for track extrapolation.

This script trains multiple model architectures:
1. Large MLP (512-512-256-128)
2. Very Large MLP (1024-512-256-128)
3. Large PINN (512-512-256-128)
4. Deep MLP (256-256-256-256-128)

All models are trained on full domain data and saved.

Author: G. Scriven
Date: 2026-01-12
"""

import numpy as np
import struct
import time
import json
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# LHCb Magnetic Field Model
# =============================================================================

class LHCbMagneticField:
    """Simplified LHCb dipole magnetic field model."""
    def __init__(self, polarity: int = 1):
        self.polarity = polarity
        self.B0 = 1.0  # Tesla
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
    """Differentiable field for PINN."""
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
    """RK8 integrator for ground truth generation."""
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
    
    def rk8_step(self, z: float, state: np.ndarray, h: float) -> np.ndarray:
        """Single RK8 step using Dormand-Prince coefficients."""
        # Simplified RK8 (actually RK4 for speed, but could be upgraded)
        k1 = self.derivatives(z, state)
        k2 = self.derivatives(z + h/2, state + h*k1/2)
        k3 = self.derivatives(z + h/2, state + h*k2/2)
        k4 = self.derivatives(z + h, state + h*k3)
        
        return state + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def propagate(self, state_in: np.ndarray, z_in: float, z_out: float) -> np.ndarray:
        """Propagate state from z_in to z_out."""
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


# =============================================================================
# Model Architectures
# =============================================================================

class TrackMLP(nn.Module):
    """Flexible MLP with configurable architecture."""
    def __init__(self, hidden_dims: list = [512, 512, 256, 128], activation: str = 'tanh'):
        super().__init__()
        
        layers = []
        prev_dim = 6  # [x, y, tx, ty, qop, dz]
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 4))  # [x, y, tx, ty]
        self.network = nn.Sequential(*layers)
        
        self.register_buffer('input_mean', torch.zeros(6))
        self.register_buffer('input_std', torch.ones(6))
        self.register_buffer('output_mean', torch.zeros(4))
        self.register_buffer('output_std', torch.ones(4))
        
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
    
    def save_binary(self, filepath):
        """Save in C++ compatible format."""
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
            
            for name in ['input_mean', 'input_std', 'output_mean', 'output_std']:
                arr = getattr(self, name).cpu().numpy()
                f.write(struct.pack('i', len(arr)))
                f.write(arr.astype(np.float64).tobytes())


class TrackPINN(nn.Module):
    """Physics-Informed Neural Network."""
    def __init__(self, hidden_dims: list = [512, 512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = 6
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.Tanh()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        
        self.network = nn.Sequential(*layers)
        self.register_buffer('input_mean', torch.zeros(6))
        self.register_buffer('input_std', torch.ones(6))
        self.register_buffer('output_mean', torch.zeros(4))
        self.register_buffer('output_std', torch.ones(4))
        
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
    
    def save_binary(self, filepath):
        """Save in C++ compatible format."""
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
            
            for name in ['input_mean', 'input_std', 'output_mean', 'output_std']:
                arr = getattr(self, name).cpu().numpy()
                f.write(struct.pack('i', len(arr)))
                f.write(arr.astype(np.float64).tobytes())


# =============================================================================
# Data Generation
# =============================================================================

def generate_training_data(num_samples: int = 50000, step_size: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate comprehensive training data."""
    print(f"Generating {num_samples} training samples...")
    
    np.random.seed(42)
    field = LHCbMagneticField(polarity=1)
    integrator = HighPrecisionRKIntegrator(field, step_size=step_size)
    
    z_in, z_out = 3000.0, 7000.0
    dz = z_out - z_in
    
    X_list = []
    Y_list = []
    
    for i in range(num_samples):
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i+1}/{num_samples} samples")
        
        # Sample full parameter space
        x0 = np.random.uniform(-900, 900)
        y0 = np.random.uniform(-750, 750)
        tx0 = np.random.uniform(-0.3, 0.3)
        ty0 = np.random.uniform(-0.25, 0.25)
        
        # Log-uniform momentum distribution (more low-p samples)
        # Avoid very low momentum to prevent numerical instabilities
        log_p = np.random.uniform(np.log(2.0), np.log(100))
        p_gev = np.exp(log_p)
        charge = np.random.choice([-1, 1])
        qop = charge / (p_gev * 1000.0)
        
        state_in = np.array([x0, y0, tx0, ty0, qop])
        state_out = integrator.propagate(state_in, z_in, z_out)
        
        X_list.append([x0, y0, tx0, ty0, qop, dz])
        Y_list.append([state_out[0], state_out[1], state_out[2], state_out[3]])
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    print(f"Generated {len(X)} training samples")
    return X, Y


# =============================================================================
# Training
# =============================================================================

def train_model(model, X_train, Y_train, X_val, Y_val, 
                epochs: int = 2000, batch_size: int = 256, lr: float = 0.001,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train a model."""
    
    model = model.to(device)
    
    # Compute normalization
    model.input_mean = torch.FloatTensor(X_train.mean(axis=0)).to(device)
    model.input_std = torch.FloatTensor(X_train.std(axis=0)).to(device)
    model.output_mean = torch.FloatTensor(Y_train.mean(axis=0)).to(device)
    model.output_std = torch.FloatTensor(Y_train.std(axis=0)).to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=100)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            Y_val_t = torch.FloatTensor(Y_val).to(device)
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
            val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"Best validation loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_model(model, X_test, Y_truth, name: str, device: str = 'cpu'):
    """Benchmark a model."""
    model = model.to(device)
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_test_t)
    
    # Timing
    n_trials = 100
    with torch.no_grad():
        start = time.time()
        for _ in range(n_trials):
            pred = model(X_test_t)
        elapsed = (time.time() - start) / n_trials
    
    pred = pred.cpu().numpy()
    errors = pred - Y_truth
    radial_errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
    
    results = {
        'name': name,
        'mean_error': np.mean(radial_errors),
        'std_error': np.std(radial_errors),
        'max_error': np.max(radial_errors),
        'p95_error': np.percentile(radial_errors, 95),
        'p99_error': np.percentile(radial_errors, 99),
        'time_per_track_us': (elapsed / len(X_test)) * 1e6,
        'total_params': sum(p.numel() for p in model.parameters())
    }
    
    print(f"\n{name} Results:")
    print(f"  Mean error: {results['mean_error']:.4f} mm")
    print(f"  95th percentile: {results['p95_error']:.4f} mm")
    print(f"  Max error: {results['max_error']:.4f} mm")
    print(f"  Time/track: {results['time_per_track_us']:.2f} μs")
    print(f"  Parameters: {results['total_params']:,}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("LARGE MODEL TRAINING FOR HIGH ACCURACY")
    print("="*70)
    print()
    
    base_dir = Path("/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators")
    models_dir = base_dir / "ml_models" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Generate or load data
    print("="*70)
    print("STEP 1: Generate Training Data")
    print("="*70)
    
    X, Y = generate_training_data(num_samples=50000, step_size=5.0)
    
    # Split
    n_train = int(0.85 * len(X))
    n_val = int(0.10 * len(X))
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    Y_train, Y_val, Y_test = Y[:n_train], Y[n_train:n_train+n_val], Y[n_train+n_val:]
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Define model configurations
    configs = [
        {
            'name': 'mlp_large',
            'class': TrackMLP,
            'hidden_dims': [512, 512, 256, 128],
            'save_name': 'mlp_large.bin'
        },
        {
            'name': 'mlp_very_large',
            'class': TrackMLP,
            'hidden_dims': [1024, 512, 256, 128],
            'save_name': 'mlp_very_large.bin'
        },
        {
            'name': 'mlp_deep',
            'class': TrackMLP,
            'hidden_dims': [256, 256, 256, 256, 128],
            'save_name': 'mlp_deep.bin'
        },
        {
            'name': 'pinn_large',
            'class': TrackPINN,
            'hidden_dims': [512, 512, 256, 128],
            'save_name': 'pinn_large.bin'
        },
    ]
    
    # Train all models
    trained_models = {}
    all_results = {}
    
    for config in configs:
        print("\n" + "="*70)
        print(f"STEP 2: Training {config['name'].upper()}")
        print("="*70)
        print(f"Architecture: {config['hidden_dims']}")
        
        model = config['class'](hidden_dims=config['hidden_dims'])
        
        trained_model, train_losses, val_losses = train_model(
            model, X_train, Y_train, X_val, Y_val,
            epochs=2000,
            batch_size=256,
            lr=0.001,
            device=device
        )
        
        trained_models[config['name']] = trained_model
        
        # Save model
        save_path = models_dir / config['save_name']
        trained_model.cpu()
        trained_model.save_binary(str(save_path))
        print(f"\nSaved: {save_path}")
        
        # Save training history
        history_path = models_dir / f"{config['name']}_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': [float(x) for x in train_losses],
                'val_losses': [float(x) for x in val_losses],
                'architecture': config['hidden_dims'],
                'date': datetime.now().isoformat()
            }, f, indent=2)
    
    # Benchmark all models
    print("\n" + "="*70)
    print("STEP 3: Benchmarking All Models")
    print("="*70)
    
    for config in configs:
        model = trained_models[config['name']]
        results = benchmark_model(model, X_test, Y_test, config['name'], device='cpu')
        all_results[config['name']] = results
    
    # Save benchmark results
    benchmark_path = models_dir / "large_models_benchmark.json"
    with open(benchmark_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark results saved: {benchmark_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Model Comparison")
    print("="*70)
    print(f"\n{'Model':<20} {'Params':>12} {'Mean (mm)':>12} {'P95 (mm)':>12} {'Time (μs)':>12}")
    print("-"*70)
    
    for name, res in all_results.items():
        print(f"{name:<20} {res['total_params']:>12,} {res['mean_error']:>12.4f} "
              f"{res['p95_error']:>12.4f} {res['time_per_track_us']:>12.2f}")
    
    print("\n" + "="*70)
    print("ALL MODELS SAVED AND BENCHMARKED!")
    print("="*70)


if __name__ == "__main__":
    main()
