#!/usr/bin/env python3
"""
Weighted Loss Training for Track Extrapolation
===============================================
Train MLP with momentum-weighted loss to improve low-p performance.
Uses conservative momentum range (2-100 GeV) to avoid numerical issues.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import struct
import json
import time
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path('/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators')
MODELS_DIR = BASE_DIR / 'ml_models' / 'models'
EXPERIMENT_DIR = BASE_DIR / 'experiments' / 'weighted_loss'
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("WEIGHTED LOSS TRAINING EXPERIMENT")
print("=" * 60)

# =============================================================================
# Magnetic Field Model
# =============================================================================
class LHCbMagneticField:
    """Simplified LHCb dipole magnetic field."""
    def __init__(self, polarity=1):
        self.polarity = polarity
        self.B0 = 1.0  # Tesla
        self.z_center = 5250.0  # mm
        self.z_halfwidth = 2500.0  # mm
        
    def get_field(self, x, y, z):
        z_rel = (z - self.z_center) / self.z_halfwidth
        By_profile = np.exp(-0.5 * z_rel**2)
        r_trans = np.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        By = self.polarity * self.B0 * By_profile * fringe_factor
        Bx = -0.01 * By * (x / 1000.0)
        return (Bx, By, 0.0)

# =============================================================================
# RK Integrator
# =============================================================================
class RKIntegrator:
    """RK4/RK8 integrator for track propagation."""
    def __init__(self, field, step_size=10.0, use_rk8=False):
        self.field = field
        self.step_size = step_size
        self.c_light = 299.792458
        self.use_rk8 = use_rk8
        
    def derivatives(self, z, state):
        x, y, tx, ty, qop = state
        Bx, By, Bz = self.field.get_field(x, y, z)
        factor = qop * self.c_light * 1e-3
        norm = np.sqrt(1.0 + tx**2 + ty**2)
        dtx_dz = factor * norm * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = factor * norm * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        return np.array([tx, ty, dtx_dz, dty_dz, 0.0])
    
    def rk4_step(self, z, state, h):
        k1 = self.derivatives(z, state)
        k2 = self.derivatives(z + 0.5*h, state + 0.5*h*k1)
        k3 = self.derivatives(z + 0.5*h, state + 0.5*h*k2)
        k4 = self.derivatives(z + h, state + h*k3)
        return state + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def rk8_step(self, z, state, h):
        state_half = self.rk4_step(z, state, h/2)
        state_two_half = self.rk4_step(z + h/2, state_half, h/2)
        state_full = self.rk4_step(z, state, h)
        return (16.0 * state_two_half - state_full) / 15.0
    
    def propagate(self, state_in, z_in, z_out):
        state = state_in.copy()
        dz = z_out - z_in
        n_steps = max(1, int(np.ceil(abs(dz) / self.step_size)))
        h = dz / n_steps
        z = z_in
        step_fn = self.rk8_step if self.use_rk8 else self.rk4_step
        for _ in range(n_steps):
            state = step_fn(z, state, h)
            z += h
            # Early termination if unstable
            if not np.all(np.isfinite(state)) or np.abs(state[2]) > 10:
                return np.full(5, np.nan)
        return state

# =============================================================================
# Neural Network Model
# =============================================================================
class TrackMLP(nn.Module):
    """Track extrapolation MLP."""
    def __init__(self, hidden_dims=[256, 256, 128, 64]):
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
                f.write(struct.pack('ii', *W.shape))
                f.write(W.astype(np.float64).tobytes())
                f.write(b.astype(np.float64).tobytes())
            for arr in [self.input_mean, self.input_std]:
                a = arr.cpu().numpy()
                f.write(struct.pack('i', len(a)))
                f.write(a.astype(np.float64).tobytes())
            for arr in [self.output_mean, self.output_std]:
                a = arr.cpu().numpy()
                f.write(struct.pack('i', len(a)))
                f.write(a.astype(np.float64).tobytes())

# =============================================================================
# Data Generation
# =============================================================================
def generate_data(n_samples, min_p=2.0, max_p=100.0, seed=42):
    """Generate training data with momentum weighting info."""
    np.random.seed(seed)
    
    field = LHCbMagneticField(polarity=1)
    integrator = RKIntegrator(field, step_size=10.0, use_rk8=True)
    
    z_in, z_out = 3000.0, 7000.0
    dz = z_out - z_in
    
    X_list, Y_list, P_list = [], [], []
    n_attempts = 0
    max_attempts = n_samples * 3
    
    while len(X_list) < n_samples and n_attempts < max_attempts:
        n_attempts += 1
        
        x0 = np.random.uniform(-900, 900)
        y0 = np.random.uniform(-750, 750)
        tx0 = np.random.uniform(-0.3, 0.3)
        ty0 = np.random.uniform(-0.25, 0.25)
        
        # Log-uniform momentum
        p_gev = np.exp(np.random.uniform(np.log(min_p), np.log(max_p)))
        charge = np.random.choice([-1, 1])
        qop = charge / (p_gev * 1000.0)
        
        state_in = np.array([x0, y0, tx0, ty0, qop])
        state_out = integrator.propagate(state_in, z_in, z_out)
        
        # Validity checks
        if (np.all(np.isfinite(state_out)) and 
            np.abs(state_out[2]) < 5 and 
            np.abs(state_out[3]) < 5 and
            np.abs(state_out[0]) < 5000 and
            np.abs(state_out[1]) < 5000):
            X_list.append([x0, y0, tx0, ty0, qop, dz])
            Y_list.append(state_out[:4])
            P_list.append(p_gev)
        
        if len(X_list) % 2000 == 0 and len(X_list) > 0:
            print(f"  Generated {len(X_list)}/{n_samples} samples")
    
    return np.array(X_list), np.array(Y_list), np.array(P_list)

# =============================================================================
# Training Functions
# =============================================================================
def train_uniform(X, Y, epochs=1000, lr=1e-3, batch_size=256):
    """Train with uniform loss (baseline)."""
    n_train = int(0.85 * len(X))
    X_t = torch.FloatTensor(X[:n_train])
    Y_t = torch.FloatTensor(Y[:n_train])
    X_val = torch.FloatTensor(X[n_train:])
    Y_val = torch.FloatTensor(Y[n_train:])
    
    model = TrackMLP()
    model.input_mean = X_t.mean(dim=0)
    model.input_std = X_t.std(dim=0) + 1e-8
    model.output_mean = Y_t.mean(dim=0)
    model.output_std = Y_t.std(dim=0) + 1e-8
    
    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader)
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    return model, history


def train_weighted(X, Y, P, epochs=1000, lr=1e-3, batch_size=256, weight_power=2.0):
    """Train with momentum-weighted loss (upweight low-p tracks)."""
    n_train = int(0.85 * len(X))
    X_t = torch.FloatTensor(X[:n_train])
    Y_t = torch.FloatTensor(Y[:n_train])
    P_t = torch.FloatTensor(P[:n_train])
    X_val = torch.FloatTensor(X[n_train:])
    Y_val = torch.FloatTensor(Y[n_train:])
    
    model = TrackMLP()
    model.input_mean = X_t.mean(dim=0)
    model.input_std = X_t.std(dim=0) + 1e-8
    model.output_mean = Y_t.mean(dim=0)
    model.output_std = Y_t.std(dim=0) + 1e-8
    
    # Compute weights: w = (p_max / p)^power, normalized
    weights = (P_t.max() / P_t) ** weight_power
    weights = weights / weights.mean()  # Normalize to mean=1
    
    dataset = TensorDataset(X_t, Y_t, weights.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by, bw in loader:
            optimizer.zero_grad()
            pred = model(bx)
            # Weighted MSE loss
            loss = (bw * (pred - by)**2).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader)
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = ((model(X_val) - Y_val)**2).mean().item()
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    return model, history

# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(model, X, Y, P):
    """Evaluate model performance by momentum bins."""
    model.eval()
    X_t = torch.FloatTensor(X)
    
    with torch.no_grad():
        preds = model(X_t).numpy()
    
    errors = preds - Y
    radial = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
    
    # Overall stats
    results = {
        'overall': {
            'mean': float(np.mean(radial)),
            'std': float(np.std(radial)),
            'p95': float(np.percentile(radial, 95)),
            'max': float(np.max(radial))
        }
    }
    
    # By momentum bin
    p_bins = [(2, 5), (5, 10), (10, 20), (20, 50), (50, 100)]
    for p_low, p_high in p_bins:
        mask = (P >= p_low) & (P < p_high)
        if mask.sum() > 0:
            results[f'{p_low}-{p_high}GeV'] = {
                'n': int(mask.sum()),
                'mean': float(np.mean(radial[mask])),
                'std': float(np.std(radial[mask])),
                'p95': float(np.percentile(radial[mask], 95))
            }
    
    return results

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # Generate data
    print("\n[1/4] Generating training data...")
    X_train, Y_train, P_train = generate_data(20000, min_p=2.0, max_p=100.0, seed=42)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Momentum range: {P_train.min():.1f} - {P_train.max():.1f} GeV/c")
    
    print("\n[2/4] Generating test data...")
    X_test, Y_test, P_test = generate_data(3000, min_p=2.0, max_p=100.0, seed=123)
    print(f"  Test samples: {len(X_test)}")
    
    # Train uniform baseline
    print("\n[3/4] Training UNIFORM loss model...")
    t0 = time.time()
    model_uniform, hist_uniform = train_uniform(X_train, Y_train, epochs=1000)
    t_uniform = time.time() - t0
    print(f"  Training time: {t_uniform:.1f}s")
    
    # Train weighted
    print("\n[4/4] Training WEIGHTED loss model (power=2)...")
    t0 = time.time()
    model_weighted, hist_weighted = train_weighted(X_train, Y_train, P_train, epochs=1000, weight_power=2.0)
    t_weighted = time.time() - t0
    print(f"  Training time: {t_weighted:.1f}s")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    results_uniform = evaluate_model(model_uniform, X_test, Y_test, P_test)
    results_weighted = evaluate_model(model_weighted, X_test, Y_test, P_test)
    
    print(f"\n{'Momentum Bin':<15} {'Uniform Mean':<15} {'Weighted Mean':<15} {'Improvement':<12}")
    print("-" * 57)
    
    for key in results_uniform:
        u = results_uniform[key]['mean']
        w = results_weighted[key]['mean']
        imp = (u - w) / u * 100 if u > 0 else 0
        print(f"{key:<15} {u:<15.3f} {w:<15.3f} {imp:>+.1f}%")
    
    # Save models
    print("\nSaving models...")
    model_uniform.save_binary(str(EXPERIMENT_DIR / 'mlp_uniform.bin'))
    model_weighted.save_binary(str(EXPERIMENT_DIR / 'mlp_weighted.bin'))
    
    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'min_p_gev': 2.0,
            'max_p_gev': 100.0,
            'epochs': 1000,
            'weight_power': 2.0
        },
        'uniform': results_uniform,
        'weighted': results_weighted,
        'training_time_s': {
            'uniform': t_uniform,
            'weighted': t_weighted
        }
    }
    
    with open(EXPERIMENT_DIR / 'weighted_loss_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {EXPERIMENT_DIR}")
    print("Done!")
