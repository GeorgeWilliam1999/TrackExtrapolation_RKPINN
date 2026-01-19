#!/usr/bin/env python3
"""
Training Script for Track Extrapolation Neural Networks

This script handles:
1. Data loading from NPZ format
2. Train/validation/test splitting
3. Training loop with logging
4. Checkpoint management
5. Early stopping
6. Results export

Author: G. Scriven
Date: January 2026

Usage:
    python train.py --model mlp --preset medium --epochs 100
    python train.py --model pinn --preset medium --lambda_pde 1.0 --epochs 100
    python train.py --config configs/my_experiment.yaml
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))
from architectures import (
    create_model, get_preset_config, MODEL_REGISTRY, ARCHITECTURE_PRESETS
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    'data_path': '../data_generation/data/training_50M.npz',
    'train_fraction': 0.8,
    'val_fraction': 0.1,
    'test_fraction': 0.1,
    'max_samples': None,  # Use all data
    
    # Model
    'model_type': 'mlp',
    'hidden_dims': [128, 128, 64],
    'activation': 'silu',
    'dropout': 0.0,
    
    # Physics (PINN/RK_PINN only)
    'lambda_pde': 1.0,  # Weight for PDE residual loss
    'lambda_ic': 1.0,   # Weight for initial condition loss
    'n_collocation': 10,  # Number of collocation points
    
    # Training
    'batch_size': 2048,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',  # 'cosine', 'step', 'none'
    'warmup_epochs': 5,
    
    # Early stopping
    'patience': 20,
    'min_delta': 1e-6,
    
    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'save_every': 10,
    
    # Logging
    'log_every': 1,
    'use_tensorboard': False,
    'use_wandb': False,
    'experiment_name': None,
    
    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True,
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load track data from NPZ file.
    
    Returns:
        X: Input features [N, 6] - [x, y, tx, ty, q/p, dz]
        Y: Output targets [N, 4] - [x_out, y_out, tx_out, ty_out]
        P: Momentum [N] - for analysis
    """
    print(f"Loading data from {config['data_path']}...")
    
    data = np.load(config['data_path'])
    X = data['X']  # [N, 5] - [x, y, tx, ty, q/p]
    Y = data['Y']  # [N, 5] - [x, y, tx, ty, qop_out]
    P = data['P']  # [N] - momentum
    
    # Load metadata for dz
    if 'dz_mean' in data:
        dz_mean = float(data['dz_mean'])
    else:
        dz_mean = 2300.0  # Default: VELO to T station
    
    # Add dz to input (constant for now, could be varied)
    N = X.shape[0]
    dz = np.full((N, 1), dz_mean, dtype=np.float32)
    X = np.hstack([X, dz])  # [N, 6]
    
    # Extract output (x, y, tx, ty only)
    Y = Y[:, :4].astype(np.float32)
    
    # Limit samples if requested
    if config['max_samples'] is not None:
        max_n = min(config['max_samples'], N)
        indices = np.random.choice(N, max_n, replace=False)
        X = X[indices]
        Y = Y[indices]
        P = P[indices]
    
    print(f"  Loaded {X.shape[0]:,} samples")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  P shape: {P.shape}")
    print(f"  dz: {dz_mean:.1f} mm")
    
    return X.astype(np.float32), Y.astype(np.float32), P.astype(np.float32)


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    config: dict
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split data into train/val/test sets."""
    N = X.shape[0]
    
    train_frac = config['train_fraction']
    val_frac = config['val_fraction']
    
    # Shuffle indices
    indices = np.random.permutation(N)
    
    # Split
    train_end = int(N * train_frac)
    val_end = int(N * (train_frac + val_frac))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    splits = {
        'train': (X[train_idx], Y[train_idx], P[train_idx]),
        'val': (X[val_idx], Y[val_idx], P[val_idx]),
        'test': (X[test_idx], Y[test_idx], P[test_idx]),
    }
    
    print(f"\nData splits:")
    for name, (x, y, p) in splits.items():
        print(f"  {name}: {x.shape[0]:,} samples")
    
    return splits


def create_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: dict
) -> Dict[str, DataLoader]:
    """Create PyTorch dataloaders for each split."""
    loaders = {}
    
    for name, (X, Y, P) in splits.items():
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.from_numpy(P)
        )
        
        shuffle = (name == 'train')
        loaders[name] = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=(name == 'train'),  # Drop incomplete batches for training
        )
    
    return loaders


# =============================================================================
# Training Utilities
# =============================================================================

def create_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler."""
    scheduler_type = config['scheduler']
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warmup
        total_steps = config['epochs'] * steps_per_epoch
        warmup_steps = config['warmup_epochs'] * steps_per_epoch
        
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(0.01, step / warmup_steps)
            remaining = total_steps - warmup_steps
            if remaining <= 0:
                return 1.0
            progress = (step - warmup_steps) / remaining
            return max(0.01, 0.5 * (1 + np.cos(np.pi * min(progress, 1.0))))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['epochs'] // 3,
            gamma=0.1
        )
    
    else:
        return None


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_pred: Predicted values [N, 4]
        y_true: Ground truth values [N, 4]
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Position error (x, y)
        pos_error = torch.sqrt(
            (y_pred[:, 0] - y_true[:, 0])**2 +
            (y_pred[:, 1] - y_true[:, 1])**2
        )
        
        # Slope error (tx, ty)
        slope_error = torch.sqrt(
            (y_pred[:, 2] - y_true[:, 2])**2 +
            (y_pred[:, 3] - y_true[:, 3])**2
        )
        
        # Individual errors
        x_error = torch.abs(y_pred[:, 0] - y_true[:, 0])
        y_error = torch.abs(y_pred[:, 1] - y_true[:, 1])
        tx_error = torch.abs(y_pred[:, 2] - y_true[:, 2])
        ty_error = torch.abs(y_pred[:, 3] - y_true[:, 3])
        
        return {
            'pos_mean_mm': pos_error.mean().item(),
            'pos_std_mm': pos_error.std().item(),
            'pos_95_mm': torch.quantile(pos_error, 0.95).item(),
            'slope_mean': slope_error.mean().item(),
            'x_mean_mm': x_error.mean().item(),
            'y_mean_mm': y_error.mean().item(),
            'tx_mean': tx_error.mean().item(),
            'ty_mean': ty_error.mean().item(),
        }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    device: torch.device,
    config: dict
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    n_batches = 0
    
    criterion = nn.MSELoss()
    
    for x, y, p in loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x)
        
        # Data loss
        data_loss = criterion(y_pred, y)
        total_data_loss += data_loss.item()
        
        # Physics loss (PINN models)
        loss = data_loss
        if hasattr(model, 'compute_physics_loss'):
            physics_losses = model.compute_physics_loss(x, y_pred)
            physics_loss = sum(physics_losses.values())
            loss = loss + physics_loss
            total_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'data_loss': total_data_loss / n_batches,
        'physics_loss': total_physics_loss / n_batches,
        'lr': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict
) -> Dict[str, float]:
    """Validate model on a dataset."""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    n_batches = 0
    
    criterion = nn.MSELoss()
    
    for x, y, p in loader:
        x = x.to(device)
        y = y.to(device)
        
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        all_preds.append(y_pred.cpu())
        all_targets.append(y.cpu())
        total_loss += loss.item()
        n_batches += 1
    
    # Concatenate all predictions
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(y_pred, y_true)
    metrics['loss'] = total_loss / n_batches
    
    return metrics


def train(config: dict):
    """Full training loop."""
    
    # Setup
    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment_name'] or f"{config['model_type']}_{timestamp}"
    exp_dir = Path(config['checkpoint_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    X, Y, P = load_data(config)
    splits = split_data(X, Y, P, config)
    loaders = create_dataloaders(splits, config)
    
    # Create model
    model_kwargs = {
        'hidden_dims': config['hidden_dims'],
        'activation': config['activation'],
    }
    
    if config['model_type'] in ['pinn', 'rk_pinn']:
        model_kwargs['lambda_pde'] = config['lambda_pde']
        model_kwargs['lambda_ic'] = config['lambda_ic']
        if config['model_type'] == 'pinn':
            model_kwargs['n_collocation'] = config['n_collocation']
    
    if config['model_type'] != 'rk_pinn':
        model_kwargs['dropout'] = config.get('dropout', 0.0)
    
    model = create_model(config['model_type'], **model_kwargs)
    
    # Set normalization from training data
    X_train = torch.from_numpy(splits['train'][0])
    Y_train = torch.from_numpy(splits['train'][1])
    model.set_normalization(X_train, Y_train)
    
    model = model.to(device)
    
    print(f"\nModel: {config['model_type'].upper()}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Config: {model.get_config()}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    steps_per_epoch = len(loaders['train'])
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta']
    )
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'test_final': None,
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'training_time': 0,
    }
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # Train
        train_metrics = train_epoch(
            model, loaders['train'], optimizer, scheduler, device, config
        )
        
        # Validate
        val_metrics = validate(model, loaders['val'], device, config)
        
        # Record history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Log progress
        if epoch % config['log_every'] == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"train_loss={train_metrics['loss']:.6f}, "
                  f"val_loss={val_metrics['loss']:.6f}, "
                  f"val_pos={val_metrics['pos_mean_mm']:.4f}mm, "
                  f"lr={train_metrics['lr']:.2e}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = best_val_loss
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, exp_dir / 'best_model.pt')
            
            # Save normalization
            model.save_normalization(str(exp_dir / 'normalization.json'))
        
        # Periodic checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    history['training_time'] = training_time
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Training time: {training_time/60:.1f} minutes")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best val loss: {history['best_val_loss']:.6f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(exp_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = validate(model, loaders['test'], device, config)
    history['test_final'] = test_metrics
    
    print(f"\nTest Results:")
    print(f"  Position error: {test_metrics['pos_mean_mm']:.4f} ± {test_metrics['pos_std_mm']:.4f} mm")
    print(f"  Position 95%:   {test_metrics['pos_95_mm']:.4f} mm")
    print(f"  Slope error:    {test_metrics['slope_mean']:.6f}")
    
    # Save final history
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model config
    with open(exp_dir / 'model_config.json', 'w') as f:
        json.dump(model.get_config(), f, indent=2)
    
    print(f"\nResults saved to: {exp_dir}")
    
    return model, history, exp_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train track extrapolation neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default=DEFAULT_CONFIG['data_path'],
                       help='Path to training data NPZ file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use')
    
    # Model
    parser.add_argument('--model', type=str, default='mlp',
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model type')
    parser.add_argument('--preset', type=str, default=None,
                       choices=list(ARCHITECTURE_PRESETS.keys()),
                       help='Architecture preset (overrides hidden_dims)')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                       default=[128, 128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--activation', type=str, default='silu',
                       help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    
    # Physics (PINN/RK_PINN)
    parser.add_argument('--lambda_pde', type=float, default=1.0,
                       help='Weight for PDE residual loss (PINN/RK_PINN only)')
    parser.add_argument('--lambda_ic', type=float, default=1.0,
                       help='Weight for initial condition loss (PINN/RK_PINN only)')
    parser.add_argument('--n_collocation', type=int, default=10,
                       help='Number of collocation points (PINN only)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Maximum number of epochs (early stopping determines actual)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=1e-7,
                       help='Minimum improvement for early stopping')
    
    # Experiment
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config from args
    config = DEFAULT_CONFIG.copy()
    
    config['data_path'] = args.data_path
    config['max_samples'] = args.max_samples
    config['model_type'] = args.model
    config['activation'] = args.activation
    config['dropout'] = args.dropout
    config['lambda_pde'] = args.lambda_pde
    config['lambda_ic'] = args.lambda_ic
    config['n_collocation'] = args.n_collocation
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['weight_decay'] = args.weight_decay
    config['patience'] = args.patience
    config['min_delta'] = args.min_delta
    config['experiment_name'] = args.name
    config['checkpoint_dir'] = args.checkpoint_dir
    config['num_workers'] = args.num_workers
    
    # Apply preset first, then allow hidden_dims to override
    if args.preset:
        preset_config = get_preset_config(args.preset)
        config['hidden_dims'] = preset_config['hidden_dims']
    
    # If hidden_dims explicitly provided, use it (overrides preset)
    if args.hidden_dims and args.hidden_dims != [128, 128, 64]:  # Not default
        config['hidden_dims'] = args.hidden_dims
    elif not args.preset:
        config['hidden_dims'] = args.hidden_dims
    
    # Device
    if args.device:
        config['device'] = args.device
    
    # Train
    model, history, exp_dir = train(config)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
