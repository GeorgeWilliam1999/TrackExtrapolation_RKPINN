#!/usr/bin/env python3
"""
V4 Unified Training Script

Trains MLP, QuadraticResidual, and PINNZFracInput models from JSON config files.
Supports variable dz, supervised collocation, and all V4 architectures.

Usage:
    python train_v4.py --config configs/quad_1024.json
    python train_v4.py --config configs/mlp_512.json

Author: G. Scriven
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# =============================================================================
# V4 Architecture Definitions (self-contained, no V3 dependency)
# =============================================================================

class MLPV4(nn.Module):
    """Standard MLP baseline for track extrapolation."""
    
    def __init__(self, hidden_dims: List[int], activation: str = 'silu'):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())
        
        layers = []
        in_dim = 6  # [x, y, tx, ty, qop, dz]
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))  # [x, y, tx, ty]
        self.network = nn.Sequential(*layers)
        
        # Small init for output layer
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, state_dz, z_frac=None):
        """Forward pass. z_frac is ignored (MLP only predicts endpoint)."""
        return self.network(state_dz)


class QuadraticResidual(nn.Module):
    """
    Output = IC + z_frac × c₁ + z_frac² × c₂
    
    Guarantees IC at z_frac=0.
    The z_frac² term captures parabolic position trajectories
    arising from integration of linearly-varying slopes.
    """
    
    def __init__(self, hidden_dims: List[int], activation: str = 'silu'):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())
        
        layers = []
        in_dim = 6
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        
        # Two output heads: linear and quadratic corrections
        self.head_linear = nn.Linear(in_dim, 4)     # c₁
        self.head_quadratic = nn.Linear(in_dim, 4)  # c₂
        
        for head in [self.head_linear, self.head_quadratic]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
    
    def forward(self, state_dz, z_frac=None):
        initial = state_dz[:, :4]
        features = self.backbone(state_dz)
        c1 = self.head_linear(features)
        c2 = self.head_quadratic(features)
        
        if z_frac is None:
            return initial + c1 + c2
        
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)
        
        if z_frac.size(-1) == 1:
            return initial + z_frac * c1 + z_frac**2 * c2
        else:
            zf = z_frac.unsqueeze(-1)
            return (initial.unsqueeze(1) +
                    zf * c1.unsqueeze(1) +
                    zf**2 * c2.unsqueeze(1))


class PINNZFracInput(nn.Module):
    """
    z_frac as 7th input with residual IC guarantee.
    Output = IC + z_frac × network(state, dz, z_frac)
    
    The network can learn arbitrary nonlinear z_frac dependence.
    """
    
    def __init__(self, hidden_dims: List[int], activation: str = 'silu'):
        super().__init__()
        act_map = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh(), 'gelu': nn.GELU()}
        act = act_map.get(activation, nn.SiLU())
        
        layers = []
        in_dim = 7  # 6 + z_frac
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.core = nn.Sequential(*layers)
        
        nn.init.xavier_uniform_(self.core[-1].weight, gain=0.1)
        nn.init.zeros_(self.core[-1].bias)
    
    def forward(self, state_dz, z_frac=None):
        B = state_dz.size(0)
        initial = state_dz[:, :4]
        
        if z_frac is None:
            z_frac = torch.ones(B, 1, device=state_dz.device)
        if z_frac.dim() == 1:
            z_frac = z_frac.unsqueeze(-1)
        
        if z_frac.size(-1) == 1:
            inp = torch.cat([state_dz, z_frac], dim=-1)
            correction = self.core(inp)
            return initial + z_frac * correction
        else:
            N_col = z_frac.size(-1)
            state_exp = state_dz.unsqueeze(1).expand(-1, N_col, -1)
            zf_exp = z_frac.unsqueeze(-1)
            inp = torch.cat([state_exp, zf_exp], dim=-1)
            correction = self.core(inp)
            return initial.unsqueeze(1) + zf_exp * correction


MODEL_REGISTRY = {
    'mlp': MLPV4,
    'quadratic': QuadraticResidual,
    'zfrac': PINNZFracInput,
}


# =============================================================================
# Dataset
# =============================================================================

class V4Dataset(Dataset):
    """
    Unified dataset for MLP and PINN training.
    For MLP: only X, Y are used.
    For PINN: X, Y, z_frac, Y_col are all used.
    """
    
    def __init__(self, X, Y, z_frac=None, Y_col=None, normalize=True):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.has_collocation = z_frac is not None and Y_col is not None
        
        if self.has_collocation:
            self.z_frac = torch.from_numpy(z_frac).float()
            self.Y_col = torch.from_numpy(Y_col).float()
        
        # Normalization
        if normalize:
            self.input_mean = self.X.mean(dim=0)
            self.input_std = self.X.std(dim=0)
            self.input_std[self.input_std < 1e-6] = 1.0
            
            self.output_mean = self.Y.mean(dim=0)
            self.output_std = self.Y.std(dim=0)
            self.output_std[self.output_std < 1e-6] = 1.0
            
            self.X = (self.X - self.input_mean) / self.input_std
            self.Y = (self.Y - self.output_mean) / self.output_std
            if self.has_collocation:
                self.Y_col = (self.Y_col - self.output_mean) / self.output_std
        else:
            self.input_mean = torch.zeros(6)
            self.input_std = torch.ones(6)
            self.output_mean = torch.zeros(4)
            self.output_std = torch.ones(4)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.has_collocation:
            return self.X[idx], self.Y[idx], self.z_frac[idx], self.Y_col[idx]
        else:
            return self.X[idx], self.Y[idx]
    
    def get_norm_stats(self):
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
        }


# =============================================================================
# Loss Functions
# =============================================================================

def mlp_loss(model, X, Y):
    """Standard MSE loss for MLP (endpoint only)."""
    pred = model(X)
    return F.mse_loss(pred, Y), {'total': F.mse_loss(pred, Y).item()}


def pinn_loss(model, X, Y, z_frac, Y_col,
              lambda_ic=10.0, lambda_endpoint=1.0, lambda_collocation=1.0):
    """PINN loss with IC, endpoint, and supervised collocation."""
    B = X.size(0)
    
    z0 = torch.zeros(B, device=X.device)
    pred_ic = model(X, z0)
    target_ic = X[:, :4]
    loss_ic = F.mse_loss(pred_ic, target_ic)
    
    z1 = torch.ones(B, device=X.device)
    pred_end = model(X, z1)
    loss_end = F.mse_loss(pred_end, Y)
    
    pred_col = model(X, z_frac)
    loss_col = F.mse_loss(pred_col, Y_col)
    
    total = lambda_ic * loss_ic + lambda_endpoint * loss_end + lambda_collocation * loss_col
    
    return total, {
        'total': total.item(),
        'ic': loss_ic.item(),
        'endpoint': loss_end.item(),
        'collocation': loss_col.item(),
    }


# =============================================================================
# Training Loop
# =============================================================================

def train(config: Dict):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    model_type = config['model']['type']
    is_pinn = model_type in ('quadratic', 'zfrac')
    
    # ---- Load data ----
    data_path = config['data']['train_path']
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    X_all = data['X']
    Y_all = data['Y']
    
    if is_pinn:
        z_frac_all = data['z_frac']
        Y_col_all = data['Y_col']
        print(f"  X: {X_all.shape}, Y: {Y_all.shape}")
        print(f"  z_frac: {z_frac_all.shape}, Y_col: {Y_col_all.shape}")
    else:
        z_frac_all = None
        Y_col_all = None
        print(f"  X: {X_all.shape}, Y: {Y_all.shape}")
    
    # ---- Split ----
    n = len(X_all)
    val_frac = config['data'].get('val_split', 0.1)
    n_val = int(n * val_frac)
    n_train = n - n_val
    
    np.random.seed(config.get('seed', 42))
    idx = np.random.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    
    # Create datasets
    if is_pinn:
        train_ds = V4Dataset(X_all[train_idx], Y_all[train_idx],
                             z_frac_all[train_idx], Y_col_all[train_idx])
        val_ds = V4Dataset(X_all[val_idx], Y_all[val_idx],
                           z_frac_all[val_idx], Y_col_all[val_idx])
    else:
        train_ds = V4Dataset(X_all[train_idx], Y_all[train_idx])
        val_ds = V4Dataset(X_all[val_idx], Y_all[val_idx])
    
    norm_stats = train_ds.get_norm_stats()
    
    print(f"  Train: {n_train:,}, Val: {n_val:,}")
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # ---- Create model ----
    hidden_dims = config['model']['hidden_dims']
    activation = config['model'].get('activation', 'silu')
    
    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(hidden_dims=hidden_dims, activation=activation).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_type} {hidden_dims}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Activation: {activation}")
    
    # ---- Optimizer & Scheduler ----
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    wd = config['training'].get('weight_decay', 1e-4)
    warmup = config['training'].get('warmup_epochs', 5)
    grad_clip = config['training'].get('grad_clip', 1.0)
    patience = config['training'].get('patience', 20)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
    
    # ---- Loss config ----
    if is_pinn:
        lambda_ic = config['loss']['lambda_ic']
        lambda_end = config['loss']['lambda_endpoint']
        lambda_col = config['loss']['lambda_collocation']
        print(f"  Loss weights: λ_ic={lambda_ic}, λ_end={lambda_end}, λ_col={lambda_col}")
    
    # ---- Output ----
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = None
    if config['output'].get('tensorboard', False) and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / 'tb_logs')
    
    # ---- Training loop ----
    print(f"\nTraining for {epochs} epochs (patience={patience})...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': []}
    no_improve = 0
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        model.train()
        train_losses = {}
        n_batch = 0
        
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            if is_pinn:
                X, Y, zf, Yc = batch
                loss, losses = pinn_loss(model, X, Y, zf, Yc,
                                         lambda_ic, lambda_end, lambda_col)
            else:
                X, Y = batch
                loss, losses = mlp_loss(model, X, Y)
            
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            for k, v in losses.items():
                train_losses[k] = train_losses.get(k, 0) + v
            n_batch += 1
        
        train_losses = {k: v / n_batch for k, v in train_losses.items()}
        
        # Validate
        model.eval()
        val_losses = {}
        n_batch = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                if is_pinn:
                    X, Y, zf, Yc = batch
                    _, losses = pinn_loss(model, X, Y, zf, Yc,
                                          lambda_ic, lambda_end, lambda_col)
                else:
                    X, Y = batch
                    _, losses = mlp_loss(model, X, Y)
                
                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v
                n_batch += 1
        
        val_losses = {k: v / n_batch for k, v in val_losses.items()}
        
        scheduler.step()
        history['train'].append(train_losses)
        history['val'].append(val_losses)
        
        elapsed = time.time() - t0
        
        # TensorBoard
        if writer:
            for k, v in train_losses.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_losses.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        
        # Print
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            if is_pinn:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {train_losses['total']:.6f} "
                      f"(ic={train_losses.get('ic',0):.4f}, end={train_losses.get('endpoint',0):.4f}, "
                      f"col={train_losses.get('collocation',0):.4f}) | "
                      f"Val: {val_losses['total']:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
            else:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {train_losses['total']:.6f} | "
                      f"Val: {val_losses['total']:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
        
        # Best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
                'norm_stats': norm_stats,
                'model_type': model_type,
                'hidden_dims': hidden_dims,
            }, output_dir / 'best_model.pt')
        else:
            no_improve += 1
        
        # Periodic checkpoint
        save_every = config['output'].get('save_every', 10)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'norm_stats': norm_stats,
            }, output_dir / f'checkpoint_epoch{epoch+1}.pt')
        
        # Early stopping
        if patience > 0 and no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Save final
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'norm_stats': norm_stats,
        'history': history,
        'model_type': model_type,
        'hidden_dims': hidden_dims,
    }, output_dir / 'final_model.pt')
    
    # Save config copy
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        # Convert to serializable format
        hist_ser = {
            'train': [{k: float(v) for k, v in h.items()} for h in history['train']],
            'val': [{k: float(v) for k, v in h.items()} for h in history['val']],
        }
        json.dump(hist_ser, f, indent=2)
    
    if writer:
        writer.close()
    
    print("=" * 80)
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f} (epoch {best_epoch+1})")
    print(f"  Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='V4 Training Script')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    print("=" * 80)
    print(f"V4 Training: {config.get('description', args.config)}")
    print("=" * 80)
    
    train(config)


if __name__ == '__main__':
    main()
