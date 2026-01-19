#!/usr/bin/env python3
"""
Train large models on GPU using pre-generated data.

This script loads pre-generated training data and trains models on GPU.

Usage:
    python train_on_gpu.py --data data/ --model large --epochs 2000
    
Author: G. Scriven
Date: 2026-01-12
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import struct
import time
from pathlib import Path
from datetime import datetime


class TrackMLP(nn.Module):
    """Configurable MLP architecture."""
    def __init__(self, hidden_dims=[512, 512, 256, 128], activation='tanh'):
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


def train_model(model, X_train, Y_train, X_val, Y_val, 
                epochs=2000, batch_size=512, lr=0.001, device='cuda'):
    """Train model on GPU."""
    
    print(f"\nTraining on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = model.to(device)
    
    # Compute normalization (avoid division by zero for constant features)
    input_std = X_train.std(axis=0)
    input_std[input_std == 0] = 1.0  # Keep constant features unchanged
    output_std = Y_train.std(axis=0)
    output_std[output_std == 0] = 1.0
    
    model.input_mean = torch.FloatTensor(X_train.mean(axis=0)).to(device)
    model.input_std = torch.FloatTensor(input_std).to(device)
    model.output_mean = torch.FloatTensor(Y_train.mean(axis=0)).to(device)
    model.output_std = torch.FloatTensor(output_std).to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=100)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training: {epochs} epochs, batch size {batch_size}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            
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
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Train: {avg_train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses, best_val_loss


def benchmark_model(model, X_test, Y_test, device='cuda'):
    """Benchmark model performance."""
    model = model.to(device)
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_test_t)
    
    # Timing
    torch.cuda.synchronize() if device == 'cuda' else None
    n_trials = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_trials):
            pred = model(X_test_t)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = (time.time() - start) / n_trials
    
    # Metrics
    pred = pred.cpu().numpy()
    errors = pred - Y_test
    radial_errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
    
    results = {
        'mean_error': float(np.mean(radial_errors)),
        'std_error': float(np.std(radial_errors)),
        'max_error': float(np.max(radial_errors)),
        'p50_error': float(np.percentile(radial_errors, 50)),
        'p95_error': float(np.percentile(radial_errors, 95)),
        'p99_error': float(np.percentile(radial_errors, 99)),
        'time_per_track_us': (elapsed / len(X_test)) * 1e6,
        'total_params': sum(p.numel() for p in model.parameters())
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train track extrapolation models on GPU')
    parser.add_argument('--data', type=str, default='../data/', help='Data directory')
    parser.add_argument('--model', type=str, default='large', 
                       choices=['large', 'xlarge', 'deep', 'custom'],
                       help='Model architecture')
    parser.add_argument('--hidden', type=int, nargs='+', default=None,
                       help='Custom hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batch', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../models/', help='Output directory')
    parser.add_argument('--name', type=str, default=None, help='Model name')
    parser.add_argument('--dataset', type=str, default='train', help='Dataset name (e.g., train, test)')
    
    args = parser.parse_args()
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU available, using CPU")
    
    # Load data
    data_dir = Path(args.data)
    print(f"\nLoading data from {data_dir}...")
    
    X = np.load(data_dir / f'X_{args.dataset}.npy')
    Y = np.load(data_dir / f'Y_{args.dataset}.npy')
    P = np.load(data_dir / f'P_{args.dataset}.npy')
    
    print(f"Loaded {len(X)} samples")
    print(f"  Momentum range: {P.min():.2f} - {P.max():.2f} GeV")
    
    # Split data
    n_train = int(0.85 * len(X))
    n_val = int(0.10 * len(X))
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    Y_train, Y_val, Y_test = Y[:n_train], Y[n_train:n_train+n_val], Y[n_train+n_val:]
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Define architecture
    architectures = {
        'large': [512, 512, 256, 128],
        'xlarge': [1024, 512, 256, 128],
        'deep': [256, 256, 256, 256, 128]
    }
    
    if args.hidden is not None:
        hidden_dims = args.hidden
        arch_name = 'custom'
    else:
        hidden_dims = architectures[args.model]
        arch_name = args.model
    
    model_name = args.name or f'mlp_{arch_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"Architecture: {hidden_dims}")
    print(f"{'='*70}")
    
    # Create and train model
    model = TrackMLP(hidden_dims=hidden_dims, activation='tanh')
    
    model, train_losses, val_losses, best_val_loss = train_model(
        model, X_train, Y_train, X_val, Y_val,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device
    )
    
    # Benchmark
    print(f"\n{'='*70}")
    print("Benchmarking on test set...")
    print(f"{'='*70}")
    
    results = benchmark_model(model, X_test, Y_test, device=device)
    
    print(f"\nTest Results:")
    print(f"  Mean error:   {results['mean_error']:.4f} mm")
    print(f"  Median error: {results['p50_error']:.4f} mm")
    print(f"  95th pct:     {results['p95_error']:.4f} mm")
    print(f"  99th pct:     {results['p99_error']:.4f} mm")
    print(f"  Max error:    {results['max_error']:.4f} mm")
    print(f"  Time/track:   {results['time_per_track_us']:.2f} μs")
    print(f"  Parameters:   {results['total_params']:,}")
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.cpu()
    model_path = output_dir / f"{model_name}.bin"
    model.save_binary(str(model_path))
    print(f"\n✓ Saved model: {model_path}")
    
    # Save metadata
    metadata = {
        'name': model_name,
        'architecture': hidden_dims,
        'date': datetime.now().isoformat(),
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch,
            'learning_rate': args.lr,
            'best_val_loss': float(best_val_loss),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'performance': results,
        'history': {
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses]
        }
    }
    
    meta_path = output_dir / f"{model_name}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {meta_path}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
