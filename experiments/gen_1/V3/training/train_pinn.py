#!/usr/bin/env python3
"""
V3 PINN Training Script with Supervised Collocation

Trains PINN models using:
1. Endpoint loss (z_frac=1)
2. IC loss (z_frac=0, guaranteed by residual architecture)
3. Supervised collocation loss (ground truth at intermediate z_frac)

Key features:
- Residual architecture: Output = IC + z_frac × Correction
- Supervised collocation from trajectory data (no physics residual needed!)
- Efficient batched training over all z_frac values

Usage:
    python train_pinn.py --config configs/pinn_v3_res_256_col10.json

Author: G. Scriven
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
import sys

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from pinn_residual import PINNResidual, PINNWithZFracInput, create_pinn


class PINNDataset(Dataset):
    """
    Dataset for PINN training with supervised collocation.
    
    Data format:
        X: [n_samples, 6] = [x, y, tx, ty, qop, dz]
        Y: [n_samples, 4] = [x, y, tx, ty] at endpoint
        z_frac: [n_samples, n_col] = collocation z_frac values
        Y_col: [n_samples, n_col, 4] = TRUE states at collocation
    """
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        z_frac: np.ndarray,
        Y_col: np.ndarray,
        normalize: bool = True,
    ):
        # Convert to tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.z_frac = torch.from_numpy(z_frac).float()
        self.Y_col = torch.from_numpy(Y_col).float()
        
        # Compute normalization stats
        if normalize:
            self.input_mean = self.X.mean(dim=0)
            self.input_std = self.X.std(dim=0)
            self.input_std[self.input_std < 1e-6] = 1.0  # Avoid div by zero
            
            self.output_mean = self.Y.mean(dim=0)
            self.output_std = self.Y.std(dim=0)
            self.output_std[self.output_std < 1e-6] = 1.0
            
            # Normalize
            self.X = (self.X - self.input_mean) / self.input_std
            self.Y = (self.Y - self.output_mean) / self.output_std
            
            # Normalize collocation outputs (same stats as endpoint)
            self.Y_col = (self.Y_col - self.output_mean) / self.output_std
        else:
            self.input_mean = torch.zeros(6)
            self.input_std = torch.ones(6)
            self.output_mean = torch.zeros(4)
            self.output_std = torch.ones(4)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'Y': self.Y[idx],
            'z_frac': self.z_frac[idx],
            'Y_col': self.Y_col[idx],
        }
    
    def get_norm_stats(self) -> Dict[str, torch.Tensor]:
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
        }


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load PINN training data with collocation."""
    data = np.load(path)
    return data['X'], data['Y'], data['z_frac'], data['Y_col']


def pinn_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    lambda_ic: float = 10.0,
    lambda_endpoint: float = 1.0,
    lambda_collocation: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PINN loss with supervised collocation.
    
    Losses:
    1. IC loss: At z_frac=0, output should equal input (for residual, this is 0)
    2. Endpoint loss: At z_frac=1, output should match Y
    3. Collocation loss: At intermediate z_frac, output should match Y_col
    
    For residual architecture, IC is guaranteed, so IC loss should be ~0.
    """
    X = batch['X']           # [B, 6]
    Y = batch['Y']           # [B, 4]
    z_frac = batch['z_frac']  # [B, N_col]
    Y_col = batch['Y_col']   # [B, N_col, 4]
    
    B = X.size(0)
    device = X.device
    
    # 1. IC Loss (z_frac=0)
    # For residual architecture, this should be exactly 0
    z_frac_0 = torch.zeros(B, device=device)
    pred_ic = model(X, z_frac_0)
    
    # For normalized data, IC should be normalized input[:, :4]
    # But we need to be careful with normalization
    # The model outputs in normalized space, and X[:, :4] is also normalized
    target_ic = X[:, :4]  # Initial state (normalized)
    ic_loss = F.mse_loss(pred_ic, target_ic)
    
    # 2. Endpoint Loss (z_frac=1)
    z_frac_1 = torch.ones(B, device=device)
    pred_endpoint = model(X, z_frac_1)
    endpoint_loss = F.mse_loss(pred_endpoint, Y)
    
    # 3. Collocation Loss (intermediate z_frac)
    pred_col = model(X, z_frac)  # [B, N_col, 4]
    collocation_loss = F.mse_loss(pred_col, Y_col)
    
    # Total loss
    total_loss = (lambda_ic * ic_loss + 
                  lambda_endpoint * endpoint_loss +
                  lambda_collocation * collocation_loss)
    
    # Return losses for logging
    losses = {
        'total': total_loss.item(),
        'ic': ic_loss.item(),
        'endpoint': endpoint_loss.item(),
        'collocation': collocation_loss.item(),
    }
    
    return total_loss, losses


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_ic: float,
    lambda_endpoint: float,
    lambda_collocation: float,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_losses = {'total': 0, 'ic': 0, 'endpoint': 0, 'collocation': 0}
    n_batches = 0
    
    for batch in loader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward + loss
        optimizer.zero_grad()
        loss, losses = pinn_loss(
            model, batch,
            lambda_ic=lambda_ic,
            lambda_endpoint=lambda_endpoint,
            lambda_collocation=lambda_collocation,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] += v
        n_batches += 1
    
    # Average
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_ic: float,
    lambda_endpoint: float,
    lambda_collocation: float,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_losses = {'total': 0, 'ic': 0, 'endpoint': 0, 'collocation': 0}
    n_batches = 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, losses = pinn_loss(
            model, batch,
            lambda_ic=lambda_ic,
            lambda_endpoint=lambda_endpoint,
            lambda_collocation=lambda_collocation,
        )
        for k, v in losses.items():
            total_losses[k] += v
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}


def train(config: Dict):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {config['data']['train_path']}...")
    X, Y, z_frac, Y_col = load_data(config['data']['train_path'])
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  z_frac shape: {z_frac.shape}")
    print(f"  Y_col shape: {Y_col.shape}")
    
    # Split train/val
    n_samples = len(X)
    n_val = int(n_samples * config['data']['val_split'])
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Create datasets
    train_dataset = PINNDataset(
        X[train_idx], Y[train_idx], z_frac[train_idx], Y_col[train_idx],
        normalize=True
    )
    val_dataset = PINNDataset(
        X[val_idx], Y[val_idx], z_frac[val_idx], Y_col[val_idx],
        normalize=True
    )
    
    # Save normalization stats
    norm_stats = train_dataset.get_norm_stats()
    
    print(f"\n  Train samples: {n_train:,}")
    print(f"  Val samples: {n_val:,}")
    print(f"  Collocation points: {z_frac.shape[1]}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model = create_pinn(
        architecture=config['model'].get('architecture', config['model'].get('type', 'residual')),
        hidden_dims=config['model']['hidden_dims'],
        activation=config['model']['activation'],
        dropout=config['model'].get('dropout', 0.0),
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {config['model'].get('architecture', config['model'].get('type', 'residual'))}")
    print(f"  Hidden dims: {config['model']['hidden_dims']}")
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Scheduler
    epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss weights
    lambda_ic = config['loss']['lambda_ic']
    lambda_endpoint = config['loss']['lambda_endpoint']
    lambda_collocation = config['loss']['lambda_collocation']
    
    # Output directory
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("="*70)
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            lambda_ic, lambda_endpoint, lambda_collocation,
            grad_clip=config['training'].get('grad_clip', 1.0),
        )
        
        # Validate
        val_losses = validate(
            model, val_loader, device,
            lambda_ic, lambda_endpoint, lambda_collocation,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train'].append(train_losses)
        history['val'].append(val_losses)
        
        elapsed = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_losses['total']:.4f} (ic={train_losses['ic']:.4f}, "
              f"end={train_losses['endpoint']:.4f}, col={train_losses['collocation']:.4f}) | "
              f"Val: {val_losses['total']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_losses['total'] < best_val_loss and config['output']['save_best']:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'norm_stats': norm_stats,
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
        
        # Periodic save
        save_every = config['output'].get('save_every', 0)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'norm_stats': norm_stats,
            }, output_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'norm_stats': norm_stats,
        'history': history,
    }, output_dir / 'final_model.pt')
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*70)
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train V3 PINN")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    print("="*70)
    print(f"V3 PINN Training: {config['name']}")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Description: {config.get('description', 'N/A')}")
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
