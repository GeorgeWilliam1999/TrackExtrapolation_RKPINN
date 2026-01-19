#!/usr/bin/env python3
"""
Train multiple model variants for analysis notebook.
Saves all models to models/analysis/ directory.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import struct
import time
from pathlib import Path
from datetime import datetime

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TrackMLP(nn.Module):
    """Standard MLP for track extrapolation."""
    def __init__(self, hidden_dims=[128, 128, 64], activation='tanh'):
        super().__init__()
        
        layers = []
        prev_dim = 6
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 4))
        self.network = nn.Sequential(*layers)
        
        self.register_buffer('input_mean', torch.zeros(6))
        self.register_buffer('input_std', torch.ones(6))
        self.register_buffer('output_mean', torch.zeros(4))
        self.register_buffer('output_std', torch.ones(4))
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TrackPINN(TrackMLP):
    """Physics-Informed Neural Network."""
    def __init__(self, hidden_dims=[128, 128, 64], activation='tanh'):
        super().__init__(hidden_dims, activation)
        self.z_start = 4000.0
        self.z_end = 12000.0
        self.dz = self.z_end - self.z_start
    
    def physics_loss(self, x, y_pred):
        """Compute physics-based loss terms."""
        x_in, y_in = x[:, 0], x[:, 1]
        tx_in, ty_in = x[:, 2], x[:, 3]
        qop = x[:, 4]
        
        x_out, y_out = y_pred[:, 0], y_pred[:, 1]
        tx_out, ty_out = y_pred[:, 2], y_pred[:, 3]
        
        # Position consistency
        x_approx = x_in + tx_in * self.dz
        y_approx = y_in + ty_in * self.dz
        pos_residual_x = (x_out - x_approx) / (1 + 1000 * torch.abs(qop))
        pos_residual_y = (y_out - y_approx) / (1 + 1000 * torch.abs(qop))
        
        # Slope changes
        dtx = tx_out - tx_in
        dty = ty_out - ty_in
        max_bending = 0.3 * self.dz * torch.abs(qop) / 1000
        bending_violation = torch.relu(torch.abs(dtx) - max_bending - 0.5)
        ty_change_loss = dty ** 2
        
        return {
            'position_x': (pos_residual_x ** 2).mean(),
            'position_y': (pos_residual_y ** 2).mean(),
            'bending': (bending_violation ** 2).mean(),
            'ty_change': ty_change_loss.mean()
        }


def train_model(model, X_train, Y_train, X_val, Y_val, 
                epochs=500, batch_size=256, lr=0.001,
                use_physics_loss=False, physics_weight=0.1):
    """Train model and return history."""
    model = model.to(device)
    
    # Set normalization
    input_std = X_train.std(axis=0)
    input_std[input_std == 0] = 1.0
    output_std = Y_train.std(axis=0)
    output_std[output_std == 0] = 1.0
    
    model.input_mean = torch.FloatTensor(X_train.mean(axis=0)).to(device)
    model.input_std = torch.FloatTensor(input_std).to(device)
    model.output_mean = torch.FloatTensor(Y_train.mean(axis=0)).to(device)
    model.output_std = torch.FloatTensor(output_std).to(device)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    history = {'train_loss': [], 'val_loss': [], 'physics_loss': [], 'lr': []}
    best_val_loss = float('inf')
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_phys = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            
            if use_physics_loss and hasattr(model, 'physics_loss'):
                phys_losses = model.physics_loss(batch_X, pred)
                phys_total = sum(phys_losses.values())
                loss = loss + physics_weight * phys_total
                total_phys += phys_total.item()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['physics_loss'].append(total_phys / len(train_loader) if use_physics_loss else 0)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            Y_val_t = torch.FloatTensor(Y_val).to(device)
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
            history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")
    
    model.load_state_dict(best_state)
    model = model.to(device)
    
    history['train_time'] = time.time() - start_time
    history['best_val_loss'] = best_val_loss
    
    return model, history


def evaluate_model(model, X, Y, P):
    """Evaluate model and return metrics."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        pred = model(X_t).cpu().numpy()
    
    pos_error = np.sqrt((pred[:, 0] - Y[:, 0])**2 + (pred[:, 1] - Y[:, 1])**2)
    slope_error = np.sqrt((pred[:, 2] - Y[:, 2])**2 + (pred[:, 3] - Y[:, 3])**2)
    
    metrics = {
        'pos_mean': float(pos_error.mean()),
        'pos_std': float(pos_error.std()),
        'pos_median': float(np.median(pos_error)),
        'pos_p95': float(np.percentile(pos_error, 95)),
        'pos_p99': float(np.percentile(pos_error, 99)),
        'pos_max': float(pos_error.max()),
        'slope_mean': float(slope_error.mean()),
        'slope_p95': float(np.percentile(slope_error, 95)),
    }
    
    # Per momentum bin
    for low, high in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100)]:
        mask = (P >= low) & (P < high)
        if mask.sum() > 0:
            metrics[f'pos_mean_{low}_{high}GeV'] = float(pos_error[mask].mean())
    
    return metrics, pos_error, slope_error


def save_model(model, history, metrics, name, output_dir):
    """Save model weights and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch state dict
    torch.save(model.state_dict(), output_dir / f"{name}.pt")
    
    # Save metadata
    metadata = {
        'name': name,
        'architecture': model.hidden_dims,
        'activation': model.activation_name,
        'parameters': model.count_parameters(),
        'date': datetime.now().isoformat(),
        'metrics': metrics,
        'history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'physics_loss': history['physics_loss'],
            'lr': history['lr'],
            'train_time': history['train_time'],
            'best_val_loss': history['best_val_loss'],
        }
    }
    
    with open(output_dir / f"{name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved: {output_dir / name}.pt")


def main():
    # Load data
    data_dir = Path("../data")
    X = np.load(data_dir / "X_analysis.npy")
    Y = np.load(data_dir / "Y_analysis.npy")
    P = np.load(data_dir / "P_analysis.npy")
    
    print(f"Loaded {len(X)} samples")
    
    # Split data
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    n_test = int(n * 0.05)
    n_val = int(n * 0.1)
    
    X_test, Y_test, P_test = X[indices[:n_test]], Y[indices[:n_test]], P[indices[:n_test]]
    X_val, Y_val = X[indices[n_test:n_test+n_val]], Y[indices[n_test:n_test+n_val]]
    X_train, Y_train = X[indices[n_test+n_val:]], Y[indices[n_test+n_val:]]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save test set for notebook
    np.save(data_dir / "X_analysis_test.npy", X_test)
    np.save(data_dir / "Y_analysis_test.npy", Y_test)
    np.save(data_dir / "P_analysis_test.npy", P_test)
    
    output_dir = Path("../models/analysis")
    
    # =========================================
    # Train different architectures
    # =========================================
    architectures = {
        'tiny':   [32, 32],
        'small':  [64, 64, 32],
        'medium': [128, 128, 64],
        'large':  [256, 256, 128, 64],
        'xlarge': [512, 512, 256, 128],
    }
    
    print("\n" + "="*70)
    print("TRAINING ARCHITECTURE VARIANTS")
    print("="*70)
    
    for name, hidden in architectures.items():
        print(f"\n🔄 Training MLP_{name}: {hidden}")
        model = TrackMLP(hidden).to(device)
        model, history = train_model(model, X_train, Y_train, X_val, Y_val, epochs=500)
        metrics, _, _ = evaluate_model(model, X_test, Y_test, P_test)
        save_model(model, history, metrics, f"mlp_{name}", output_dir)
        print(f"  → Mean error: {metrics['pos_mean']:.2f} mm, P95: {metrics['pos_p95']:.2f} mm")
    
    # =========================================
    # Train PINN variants
    # =========================================
    print("\n" + "="*70)
    print("TRAINING PINN VARIANTS")
    print("="*70)
    
    physics_weights = [0.01, 0.05, 0.1, 0.2]
    arch = [128, 128, 64]  # Medium architecture
    
    for pw in physics_weights:
        print(f"\n🔄 Training PINN (λ={pw}): {arch}")
        model = TrackPINN(arch).to(device)
        model, history = train_model(model, X_train, Y_train, X_val, Y_val, 
                                     epochs=500, use_physics_loss=True, physics_weight=pw)
        metrics, _, _ = evaluate_model(model, X_test, Y_test, P_test)
        save_model(model, history, metrics, f"pinn_lambda_{str(pw).replace('.', '_')}", output_dir)
        print(f"  → Mean error: {metrics['pos_mean']:.2f} mm, P95: {metrics['pos_p95']:.2f} mm")
    
    # =========================================
    # Train activation variants
    # =========================================
    print("\n" + "="*70)
    print("TRAINING ACTIVATION VARIANTS")
    print("="*70)
    
    for act in ['tanh', 'relu', 'silu']:
        print(f"\n🔄 Training MLP with {act} activation: {arch}")
        model = TrackMLP(arch, activation=act).to(device)
        model, history = train_model(model, X_train, Y_train, X_val, Y_val, epochs=500)
        metrics, _, _ = evaluate_model(model, X_test, Y_test, P_test)
        save_model(model, history, metrics, f"mlp_act_{act}", output_dir)
        print(f"  → Mean error: {metrics['pos_mean']:.2f} mm, P95: {metrics['pos_p95']:.2f} mm")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAll models saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
