#!/usr/bin/env python3
"""
Weighted Loss Training for Track Extrapolation

Experiment: Train MLP with momentum-weighted loss to improve low-momentum performance.

Key idea: Low momentum tracks have larger errors because they bend more sharply.
By weighting the loss inversely proportional to momentum, we can encourage the
network to focus more on these difficult cases.

Author: Auto-generated experiment
Date: 2024-12-21
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


class LHCbMagneticField:
    """Simplified LHCb dipole magnetic field model."""
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


class RKIntegrator:
    """Runge-Kutta integrator."""
    def __init__(self, field, step_size=5.0, use_rk8=True):
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
        return state


class TrackMLPWeighted(nn.Module):
    """MLP with weighted loss support."""
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
        x = (x - self.input_mean) / self.input_std
        out = self.network(x)
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


def generate_data(n_samples, seed=42):
    """Generate training data."""
    np.random.seed(seed)
    field = LHCbMagneticField(polarity=1)
    integrator = RKIntegrator(field, step_size=5.0, use_rk8=True)
    z_in, z_out = 3000.0, 7000.0
    dz = z_out - z_in
    
    X_list, Y_list = [], []
    for i in range(n_samples):
        if (i + 1) % 5000 == 0:
            print(f'  Generated {i+1}/{n_samples}')
        
        x0 = np.random.uniform(-900, 900)
        y0 = np.random.uniform(-750, 750)
        tx0 = np.random.uniform(-0.3, 0.3)
        ty0 = np.random.uniform(-0.25, 0.25)
        p_gev = np.exp(np.random.uniform(np.log(1.0), np.log(100)))
        charge = np.random.choice([-1, 1])
        qop = charge / (p_gev * 1000.0)
        
        state_in = np.array([x0, y0, tx0, ty0, qop])
        state_out = integrator.propagate(state_in, z_in, z_out)
        
        if np.all(np.isfinite(state_out)) and np.abs(state_out[2]) < 5:
            X_list.append([x0, y0, tx0, ty0, qop, dz])
            Y_list.append(state_out[:4])
    
    return np.array(X_list), np.array(Y_list)


def compute_weights(qop, weight_type='inverse_p', power=1.0):
    """
    Compute sample weights based on momentum.
    
    Args:
        qop: q/p values in 1/MeV
        weight_type: 'inverse_p', 'inverse_p_squared', 'log_p', or 'uniform'
        power: exponent for inverse_p weighting
    
    Returns:
        weights: normalized weights summing to len(qop)
    """
    p_gev = np.abs(1.0 / qop) / 1000  # Convert to GeV
    
    if weight_type == 'uniform':
        weights = np.ones_like(p_gev)
    elif weight_type == 'inverse_p':
        weights = 1.0 / (p_gev ** power)
    elif weight_type == 'inverse_p_squared':
        weights = 1.0 / (p_gev ** 2)
    elif weight_type == 'log_p':
        weights = 1.0 / (1.0 + np.log(p_gev))
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # Normalize so weights sum to N (average weight = 1)
    weights = weights * len(weights) / weights.sum()
    return weights


def train_weighted_mlp(X, Y, epochs=1000, lr=1e-3, batch_size=256, 
                       weight_type='inverse_p', weight_power=1.0):
    """Train MLP with weighted loss."""
    n_train = int(0.85 * len(X))
    
    X_t = torch.FloatTensor(X[:n_train])
    Y_t = torch.FloatTensor(Y[:n_train])
    X_val = torch.FloatTensor(X[n_train:])
    Y_val = torch.FloatTensor(Y[n_train:])
    
    # Compute weights for training data
    qop_train = X[:n_train, 4]
    weights = compute_weights(qop_train, weight_type, weight_power)
    weights_t = torch.FloatTensor(weights)
    
    model = TrackMLPWeighted()
    model.input_mean = X_t.mean(dim=0)
    model.input_std = X_t.std(dim=0) + 1e-8
    model.output_mean = Y_t.mean(dim=0)
    model.output_std = Y_t.std(dim=0) + 1e-8
    
    # Create dataset with weights
    dataset = TensorDataset(X_t, Y_t, weights_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train': [], 'val': [], 'train_weighted': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        weighted_loss = 0
        n_batches = 0
        
        for bx, by, bw in loader:
            optimizer.zero_grad()
            pred = model(bx)
            
            # Weighted MSE loss
            sq_errors = (pred - by) ** 2
            sample_losses = sq_errors.mean(dim=1)  # Mean over output dims
            loss = (sample_losses * bw).mean()  # Weighted average
            
            loss.backward()
            optimizer.step()
            
            weighted_loss += loss.item()
            train_loss += sample_losses.mean().item()  # Unweighted for comparison
            n_batches += 1
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = ((val_pred - Y_val) ** 2).mean().item()
        
        history['train'].append(train_loss / n_batches)
        history['train_weighted'].append(weighted_loss / n_batches)
        history['val'].append(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1:4d}: Train={train_loss/n_batches:.6f}, '
                  f'Weighted={weighted_loss/n_batches:.6f}, Val={val_loss:.6f}')
    
    return model, history


def evaluate_by_momentum(model, X, Y, momentum_bins=[1, 5, 20, 100]):
    """Evaluate model performance by momentum bin."""
    model.eval()
    X_t = torch.FloatTensor(X)
    
    with torch.no_grad():
        pred = model(X_t).numpy()
    
    errors = pred - Y
    radial = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
    momenta = np.abs(1.0 / X[:, 4]) / 1000
    
    results = {}
    for i in range(len(momentum_bins) - 1):
        p_low, p_high = momentum_bins[i], momentum_bins[i+1]
        mask = (momenta >= p_low) & (momenta < p_high)
        if mask.sum() > 0:
            results[f'{p_low}-{p_high} GeV'] = {
                'n': int(mask.sum()),
                'mean': float(radial[mask].mean()),
                'std': float(radial[mask].std()),
                'p95': float(np.percentile(radial[mask], 95))
            }
    
    results['overall'] = {
        'n': len(X),
        'mean': float(radial.mean()),
        'std': float(radial.std()),
        'p95': float(np.percentile(radial, 95))
    }
    
    return results


def main():
    # Setup
    base_dir = Path(__file__).parent.parent.parent
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("WEIGHTED LOSS TRAINING EXPERIMENT")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating training data...")
    X_train, Y_train = generate_data(20000, seed=42)
    print(f"Training samples: {len(X_train)}")
    
    print("\nGenerating test data...")
    X_test, Y_test = generate_data(2000, seed=123)
    print(f"Test samples: {len(X_test)}")
    
    # Experiment configurations
    experiments = [
        {'name': 'uniform', 'weight_type': 'uniform', 'weight_power': 1.0},
        {'name': 'inverse_p_1.0', 'weight_type': 'inverse_p', 'weight_power': 1.0},
        {'name': 'inverse_p_0.5', 'weight_type': 'inverse_p', 'weight_power': 0.5},
        {'name': 'inverse_p_2.0', 'weight_type': 'inverse_p', 'weight_power': 2.0},
        {'name': 'log_p', 'weight_type': 'log_p', 'weight_power': 1.0},
    ]
    
    all_results = {}
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Training: {exp['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        model, history = train_weighted_mlp(
            X_train, Y_train,
            epochs=1000,
            weight_type=exp['weight_type'],
            weight_power=exp['weight_power']
        )
        train_time = time.time() - start_time
        
        # Evaluate
        eval_results = evaluate_by_momentum(model, X_test, Y_test)
        eval_results['training_time_s'] = train_time
        eval_results['final_train_loss'] = history['train'][-1]
        eval_results['final_val_loss'] = history['val'][-1]
        
        all_results[exp['name']] = eval_results
        
        # Save model
        model_path = output_dir / f"mlp_weighted_{exp['name']}.bin"
        model.save_binary(str(model_path))
        print(f"\nSaved model to {model_path}")
        
        # Print results
        print(f"\nResults for {exp['name']}:")
        print(f"  Training time: {train_time:.1f}s")
        for key, val in eval_results.items():
            if isinstance(val, dict):
                print(f"  {key}: mean={val['mean']:.3f}mm, p95={val['p95']:.3f}mm")
    
    # Save all results
    results_path = output_dir / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Weighting':<20} {'Overall':<12} {'1-5 GeV':<12} {'5-20 GeV':<12} {'20-100 GeV':<12}")
    print("-" * 80)
    for name, res in all_results.items():
        overall = res['overall']['mean']
        low = res.get('1-5 GeV', {}).get('mean', 'N/A')
        mid = res.get('5-20 GeV', {}).get('mean', 'N/A')
        high = res.get('20-100 GeV', {}).get('mean', 'N/A')
        if isinstance(low, float):
            print(f"{name:<20} {overall:<12.3f} {low:<12.3f} {mid:<12.3f} {high:<12.3f}")
        else:
            print(f"{name:<20} {overall:<12.3f} {low:<12} {mid:<12} {high:<12}")


if __name__ == '__main__':
    main()
