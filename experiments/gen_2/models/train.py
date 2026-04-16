#!/usr/bin/env python3
"""
================================================================================
gen_2 Training Script — MLP, PINN, and RK_PINN
================================================================================

Enhanced from gen_1/V1 with:
  - Physics-informed loss for PINN and RK_PINN models
  - Per-component loss logging (data, PDE, IC)
  - JSON config file loading
  - Jacobian (transport matrix) evaluation
  - Comprehensive MLflow monitoring

Data format:
  Input  X [N, 6]:  [x, y, tx, ty, q/p, dz]   (mm, mm, -, -, 1/MeV, mm)
  Output Y [N, 4]:  [x_out, y_out, tx_out, ty_out]
  dz is VARIABLE per-sample: U[25, 10000] mm (covers VELO through full detector)

Usage:
  # MLP from config file
  python train.py --config ../configs/mlp/medium.json

  # PINN with physics loss from CLI
  python train.py --model pinn --preset medium --lambda_pde 1.0 --epochs 200

  # RK_PINN
  python train.py --config ../configs/rk_pinn/medium_lam1.0.json

Author: G. Scriven
Date: April 2026
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# MLflow support
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Import architectures (symlinked from gen_1)
sys.path.insert(0, str(Path(__file__).parent))
from architectures import (
    create_model, get_preset_config, MODEL_REGISTRY, ARCHITECTURE_PRESETS
)


# =============================================================================
# Configuration
# =============================================================================

# Default data path — gen_2 50M dataset with dz_min=25mm (VELO coverage)
_DEFAULT_DATA = str(
    Path(__file__).resolve().parent.parent
    / 'data' / 'train_50M_dz25.npz'
)

DEFAULT_CONFIG = {
    # Reproducibility
    'seed': 42,

    # Data
    'data_path': _DEFAULT_DATA,
    'train_fraction': 0.8,
    'val_fraction': 0.1,
    'test_fraction': 0.1,
    'max_samples': None,

    # Model
    'model_type': 'mlp',
    'hidden_dims': [256, 256, 128],
    'activation': 'silu',
    'dropout': 0.0,

    # Physics loss (PINN / RK_PINN only)
    'lambda_pde': 1.0,
    'lambda_ic': 0.1,
    'n_collocation': 10,
    'field_type': 'interpolated',   # 'gaussian' or 'interpolated'

    # Training
    'batch_size': 2048,
    'epochs': 300,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'grad_clip': 1.0,

    # Early stopping
    'patience': 30,
    'min_delta': 1e-7,

    # Checkpointing
    'checkpoint_dir': '../trained_models',
    'save_every': 25,

    # Logging
    'log_every': 1,
    'use_tensorboard': True,
    'use_mlflow': True,
    'mlflow_tracking_uri': None,
    'mlflow_experiment_name': 'gen_2_track_extrapolation',
    'experiment_name': None,

    # Jacobian
    'eval_jacobian': True,
    'jacobian_n_samples': 1000,

    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True,
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def load_config(config_path: str) -> dict:
    """Load experiment config from JSON file, merging with defaults."""
    with open(config_path, 'r') as f:
        file_config = json.load(f)
    config = DEFAULT_CONFIG.copy()
    config.update(file_config)
    return config


# =============================================================================
# Data Loading (shared with gen_1)
# =============================================================================

def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load track data from NPZ file.

    Returns X [N,6], Y [N,4], P [N].
    """
    print(f"Loading data from {config['data_path']}...")
    data = np.load(config['data_path'])
    X = data['X']
    Y = data['Y']
    P = data['P']

    if X.shape[1] == 5:
        dz_mean = float(data.get('dz_mean', 2300.0))
        X = np.hstack([X, np.full((X.shape[0], 1), dz_mean, dtype=np.float32)])
        print(f"  Added dz column (mean={dz_mean:.1f} mm)")
    elif X.shape[1] == 6:
        print(f"  dz in data: mean={X[:, 5].mean():.1f}, std={X[:, 5].std():.1f} mm")

    Y = Y[:, :4].astype(np.float32)

    if config['max_samples'] is not None:
        max_n = min(config['max_samples'], X.shape[0])
        indices = np.random.choice(X.shape[0], max_n, replace=False)
        X, Y, P = X[indices], Y[indices], P[indices]

    print(f"  Loaded {X.shape[0]:,} samples  (X: {X.shape}, Y: {Y.shape})")
    return X.astype(np.float32), Y.astype(np.float32), P.astype(np.float32)


def split_data(X, Y, P, config):
    """Split data into train/val/test."""
    N = X.shape[0]
    indices = np.random.permutation(N)
    t1 = int(N * config['train_fraction'])
    t2 = int(N * (config['train_fraction'] + config['val_fraction']))
    splits = {
        'train': (X[indices[:t1]], Y[indices[:t1]], P[indices[:t1]]),
        'val':   (X[indices[t1:t2]], Y[indices[t1:t2]], P[indices[t1:t2]]),
        'test':  (X[indices[t2:]], Y[indices[t2:]], P[indices[t2:]]),
    }
    for name, (x, y, p) in splits.items():
        print(f"  {name}: {x.shape[0]:,} samples")
    return splits


def create_dataloaders(splits, config):
    """Create PyTorch DataLoaders."""
    loaders = {}
    for name, (X, Y, P) in splits.items():
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(P))
        loaders[name] = DataLoader(
            ds,
            batch_size=config['batch_size'],
            shuffle=(name == 'train'),
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=(name == 'train'),
        )
    return loaders


# =============================================================================
# Scheduler / Early Stopping
# =============================================================================

def create_scheduler(optimizer, config, steps_per_epoch):
    stype = config['scheduler']
    if stype == 'cosine':
        total = config['epochs'] * steps_per_epoch
        warmup = config['warmup_epochs'] * steps_per_epoch
        def lr_fn(step):
            if warmup > 0 and step < warmup:
                return max(0.01, step / warmup)
            rem = total - warmup
            if rem <= 0:
                return 1.0
            prog = (step - warmup) / rem
            return max(0.01, 0.5 * (1 + np.cos(np.pi * min(prog, 1.0))))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    elif stype == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=config['epochs'] // 3, gamma=0.1)
    return None


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
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

def compute_metrics(y_pred, y_true):
    """Compute position and slope error metrics."""
    with torch.no_grad():
        pos_err = torch.sqrt((y_pred[:, 0] - y_true[:, 0])**2 + (y_pred[:, 1] - y_true[:, 1])**2)
        slope_err = torch.sqrt((y_pred[:, 2] - y_true[:, 2])**2 + (y_pred[:, 3] - y_true[:, 3])**2)
        return {
            'pos_mean_mm': pos_err.mean().item(),
            'pos_std_mm': pos_err.std().item(),
            'pos_95_mm': torch.quantile(pos_err, 0.95).item(),
            'pos_99_mm': torch.quantile(pos_err, 0.99).item(),
            'slope_mean': slope_err.mean().item(),
            'slope_95': torch.quantile(slope_err, 0.95).item(),
            'x_mean_mm': torch.abs(y_pred[:, 0] - y_true[:, 0]).mean().item(),
            'y_mean_mm': torch.abs(y_pred[:, 1] - y_true[:, 1]).mean().item(),
            'tx_mean': torch.abs(y_pred[:, 2] - y_true[:, 2]).mean().item(),
            'ty_mean': torch.abs(y_pred[:, 3] - y_true[:, 3]).mean().item(),
        }


# =============================================================================
# Training Loop (supports physics loss)
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, device, config):
    """Train one epoch. Adds physics loss for PINN/RK_PINN models."""
    model.train()

    total_loss = 0
    total_data_loss = 0
    total_pde_loss = 0
    total_ic_loss = 0
    n_batches = 0
    n_skipped = 0

    criterion = nn.MSELoss()
    grad_clip = config.get('grad_clip', 1.0)
    use_physics = config['model_type'] in ('pinn', 'rk_pinn') and config.get('lambda_pde', 0) > 0

    for x, y, _p in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        data_loss = criterion(y_pred, y)

        if use_physics and hasattr(model, 'compute_physics_loss'):
            physics = model.compute_physics_loss(x, y_pred)
            pde_loss = physics.get('pde', torch.tensor(0.0, device=device))
            ic_loss = physics.get('ic', torch.tensor(0.0, device=device))
            loss = data_loss + pde_loss + ic_loss
            total_pde_loss += pde_loss.item()
            total_ic_loss += ic_loss.item()
        else:
            loss = data_loss

        if torch.isnan(loss) or torch.isinf(loss):
            n_skipped += 1
            continue

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {'loss': float('inf'), 'data_loss': float('inf'),
                'pde_loss': 0, 'ic_loss': 0, 'lr': optimizer.param_groups[0]['lr'],
                'skipped': n_skipped}

    return {
        'loss': total_loss / n_batches,
        'data_loss': total_data_loss / n_batches,
        'pde_loss': total_pde_loss / n_batches if use_physics else 0,
        'ic_loss': total_ic_loss / n_batches if use_physics else 0,
        'lr': optimizer.param_groups[0]['lr'],
        'skipped': n_skipped,
    }


@torch.no_grad()
def validate(model, loader, device, config):
    """Validate model — MSE + metrics (no physics loss)."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    n_batches = 0
    criterion = nn.MSELoss()

    for x, y, _p in loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        total_loss += criterion(y_pred, y).item()
        all_preds.append(y_pred.cpu())
        all_targets.append(y.cpu())
        n_batches += 1

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    metrics = compute_metrics(y_pred, y_true)
    metrics['loss'] = total_loss / n_batches
    return metrics


# =============================================================================
# Jacobian Evaluation
# =============================================================================

def evaluate_jacobian(model, X_test, device, n_samples=1000):
    """
    Evaluate transport matrix (Jacobian) on test samples.

    Returns summary statistics of ||J||_F and per-element statistics.
    """
    model.eval()
    idx = np.random.choice(X_test.shape[0], min(n_samples, X_test.shape[0]), replace=False)
    x = torch.from_numpy(X_test[idx]).to(device)

    jac = model.compute_jacobian(x)  # [n, 4, 5]
    jac_np = jac.cpu().numpy()

    # Frobenius norm per sample
    frob = np.sqrt((jac_np ** 2).sum(axis=(1, 2)))

    # Per-element mean and std
    elem_mean = jac_np.mean(axis=0)  # [4, 5]
    elem_std = jac_np.std(axis=0)    # [4, 5]

    return {
        'frobenius_mean': float(frob.mean()),
        'frobenius_std': float(frob.std()),
        'element_mean': elem_mean.tolist(),
        'element_std': elem_std.tolist(),
    }


# =============================================================================
# Main Training
# =============================================================================

def train(config: dict):
    """Full training loop with physics loss support and comprehensive logging."""
    set_seed(config['seed'])
    device = torch.device(config['device'])
    print(f"\nDevice: {device}")

    # GPU warmup for HTCondor watchdog
    if device.type == 'cuda':
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        print("  GPU warmup complete")

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment_name'] or f"{config['model_type']}_{timestamp}"
    exp_dir = Path(config['checkpoint_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {exp_dir}")

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    X, Y, P = load_data(config)
    splits = split_data(X, Y, P, config)
    loaders = create_dataloaders(splits, config)

    # Model
    model_kwargs = {
        'hidden_dims': config['hidden_dims'],
        'activation': config['activation'],
    }
    if config['model_type'] == 'mlp':
        model_kwargs['dropout'] = config.get('dropout', 0.0)
    if config['model_type'] in ('pinn', 'rk_pinn'):
        model_kwargs['lambda_pde'] = config['lambda_pde']
        model_kwargs['lambda_ic'] = config['lambda_ic']
        if config['model_type'] == 'pinn':
            model_kwargs['n_collocation'] = config['n_collocation']
            model_kwargs['dropout'] = config.get('dropout', 0.0)

    model = create_model(config['model_type'], **model_kwargs)

    # Set normalization
    X_train = torch.from_numpy(splits['train'][0])
    Y_train = torch.from_numpy(splits['train'][1])
    model.set_normalization(X_train, Y_train)
    model = model.to(device)

    n_params = model.count_parameters()
    print(f"\nModel: {config['model_type'].upper()}")
    print(f"  Architecture: {config['hidden_dims']}")
    print(f"  Parameters: {n_params:,}")
    if config['model_type'] in ('pinn', 'rk_pinn'):
        print(f"  lambda_pde: {config['lambda_pde']}")
        print(f"  lambda_ic: {config['lambda_ic']}")

    # Optimizer / Scheduler / Early stopping
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = create_scheduler(optimizer, config, len(loaders['train']))
    early_stop = EarlyStopping(config['patience'], config['min_delta'])

    # TensorBoard
    writer = None
    if config.get('use_tensorboard') and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))
        print(f"TensorBoard: {exp_dir / 'tensorboard'}")

    # MLflow
    mlflow_run = None
    if config.get('use_mlflow') and MLFLOW_AVAILABLE:
        uri = config.get('mlflow_tracking_uri')
        if uri:
            mlflow.set_tracking_uri(uri)
        else:
            mlruns = Path(__file__).parent.parent / 'mlruns'
            mlflow.set_tracking_uri(f'file://{mlruns.resolve()}')

        mlflow.set_experiment(config['mlflow_experiment_name'])
        mlflow_run = mlflow.start_run(run_name=exp_name)

        # Log config
        flat = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                flat[k] = v
            elif isinstance(v, list):
                flat[k] = str(v)
        mlflow.log_params(flat)
        mlflow.log_param('n_parameters', n_params)
        mlflow.set_tags({
            'model_type': config['model_type'],
            'generation': 'gen_2',
            'architecture': str(config['hidden_dims']),
            'activation': config['activation'],
        })
        print(f"MLflow: {mlflow.get_tracking_uri()}")
        print(f"  Experiment: {config['mlflow_experiment_name']}")
        print(f"  Run: {mlflow_run.info.run_id[:8]}...")

    # ─── Training ─────────────────────────────────────────────
    history = {'train': [], 'val': [], 'test_final': None,
               'best_epoch': 0, 'best_val_loss': float('inf'), 'training_time': 0}

    print("\n" + "=" * 70)
    print(f"Training {config['model_type'].upper()} — {config['epochs']} epochs max")
    print("=" * 70)

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # Train
        t_metrics = train_epoch(model, loaders['train'], optimizer, scheduler, device, config)

        # Validate
        v_metrics = validate(model, loaders['val'], device, config)

        history['train'].append(t_metrics)
        history['val'].append(v_metrics)

        # ── Logging ──────────────────────────────────────────
        if epoch % config['log_every'] == 0:
            phys_str = ""
            if t_metrics['pde_loss'] > 0:
                phys_str = f"  pde={t_metrics['pde_loss']:.4e}  ic={t_metrics['ic_loss']:.4e}"
            print(f"[{epoch+1:3d}/{config['epochs']}] "
                  f"train={t_metrics['loss']:.6f}  val={v_metrics['loss']:.6f}  "
                  f"pos={v_metrics['pos_mean_mm']:.3f}mm  "
                  f"slope={v_metrics['slope_mean']:.5f}  "
                  f"lr={t_metrics['lr']:.2e}{phys_str}")

        # TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train_total', t_metrics['loss'], epoch)
            writer.add_scalar('Loss/train_data', t_metrics['data_loss'], epoch)
            writer.add_scalar('Loss/val', v_metrics['loss'], epoch)
            writer.add_scalar('Error/pos_mean_mm', v_metrics['pos_mean_mm'], epoch)
            writer.add_scalar('Error/pos_95_mm', v_metrics['pos_95_mm'], epoch)
            writer.add_scalar('Error/slope_mean', v_metrics['slope_mean'], epoch)
            writer.add_scalar('Training/lr', t_metrics['lr'], epoch)
            if t_metrics['pde_loss'] > 0:
                writer.add_scalar('Loss/pde', t_metrics['pde_loss'], epoch)
                writer.add_scalar('Loss/ic', t_metrics['ic_loss'], epoch)

        # MLflow
        if mlflow_run is not None:
            step_metrics = {
                'train_loss': t_metrics['loss'],
                'train_data_loss': t_metrics['data_loss'],
                'val_loss': v_metrics['loss'],
                'val_pos_mean_mm': v_metrics['pos_mean_mm'],
                'val_pos_95_mm': v_metrics['pos_95_mm'],
                'val_pos_99_mm': v_metrics['pos_99_mm'],
                'val_slope_mean': v_metrics['slope_mean'],
                'val_slope_95': v_metrics['slope_95'],
                'val_x_mean_mm': v_metrics['x_mean_mm'],
                'val_y_mean_mm': v_metrics['y_mean_mm'],
                'val_tx_mean': v_metrics['tx_mean'],
                'val_ty_mean': v_metrics['ty_mean'],
                'learning_rate': t_metrics['lr'],
            }
            if t_metrics['pde_loss'] > 0:
                step_metrics['train_pde_loss'] = t_metrics['pde_loss']
                step_metrics['train_ic_loss'] = t_metrics['ic_loss']
            mlflow.log_metrics(step_metrics, step=epoch)

        # ── Checkpointing ────────────────────────────────────
        if v_metrics['loss'] < best_val_loss:
            best_val_loss = v_metrics['loss']
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = best_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_metrics['loss'],
                'config': config,
            }, exp_dir / 'best_model.pt')
            model.save_normalization(str(exp_dir / 'normalization.json'))

        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # ── Early stopping ───────────────────────────────────
        if early_stop(v_metrics['loss']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    history['training_time'] = training_time

    print("\n" + "=" * 70)
    print(f"Training complete — {training_time/60:.1f} min")
    print(f"  Best epoch: {history['best_epoch']}  val_loss: {history['best_val_loss']:.6f}")
    print("=" * 70)

    # ─── Final Evaluation ─────────────────────────────────────
    ckpt = torch.load(exp_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\nTest set evaluation:")
    test_metrics = validate(model, loaders['test'], device, config)
    history['test_final'] = test_metrics
    print(f"  Position:  {test_metrics['pos_mean_mm']:.4f} ± {test_metrics['pos_std_mm']:.4f} mm")
    print(f"  Position 95%: {test_metrics['pos_95_mm']:.4f} mm")
    print(f"  Position 99%: {test_metrics['pos_99_mm']:.4f} mm")
    print(f"  Slope:     {test_metrics['slope_mean']:.6f}")

    # ─── Jacobian Evaluation ──────────────────────────────────
    jac_results = None
    if config.get('eval_jacobian', False):
        print("\nJacobian (transport matrix) evaluation:")
        jac_results = evaluate_jacobian(
            model, splits['test'][0], device,
            n_samples=config.get('jacobian_n_samples', 1000)
        )
        history['jacobian'] = jac_results
        print(f"  ||J||_F: {jac_results['frobenius_mean']:.4f} ± {jac_results['frobenius_std']:.4f}")
        with open(exp_dir / 'jacobian.json', 'w') as f:
            json.dump(jac_results, f, indent=2)

    # ─── Save History & Config ────────────────────────────────
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(exp_dir / 'model_config.json', 'w') as f:
        json.dump(model.get_config(), f, indent=2)

    # ─── MLflow Final ─────────────────────────────────────────
    if mlflow_run is not None:
        final = {
            'test_pos_mean_mm': test_metrics['pos_mean_mm'],
            'test_pos_std_mm': test_metrics['pos_std_mm'],
            'test_pos_95_mm': test_metrics['pos_95_mm'],
            'test_pos_99_mm': test_metrics['pos_99_mm'],
            'test_slope_mean': test_metrics['slope_mean'],
            'test_slope_95': test_metrics['slope_95'],
            'test_x_mean_mm': test_metrics['x_mean_mm'],
            'test_y_mean_mm': test_metrics['y_mean_mm'],
            'test_tx_mean': test_metrics['tx_mean'],
            'test_ty_mean': test_metrics['ty_mean'],
            'best_val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
            'training_time_min': training_time / 60.0,
        }
        if jac_results:
            final['jacobian_frob_mean'] = jac_results['frobenius_mean']
            final['jacobian_frob_std'] = jac_results['frobenius_std']
        mlflow.log_metrics(final)

        for artifact in ['best_model.pt', 'normalization.json', 'config.json',
                         'history.json', 'model_config.json']:
            path = exp_dir / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
        if (exp_dir / 'jacobian.json').exists():
            mlflow.log_artifact(str(exp_dir / 'jacobian.json'))

        mlflow.end_run()
        print(f"\nMLflow run: {mlflow_run.info.run_id[:8]}...")

    print(f"\nResults: {exp_dir}")
    return model, history, exp_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='gen_2: Train MLP / PINN / RK_PINN track extrapolation models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (highest priority)
    parser.add_argument('--config', type=str, default=None,
                        help='JSON config file (overrides all other args)')

    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)

    # Model
    parser.add_argument('--model', type=str, default='mlp',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--preset', type=str, default=None,
                        choices=list(ARCHITECTURE_PRESETS.keys()))
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # Physics loss
    parser.add_argument('--lambda_pde', type=float, default=None)
    parser.add_argument('--lambda_ic', type=float, default=None)
    parser.add_argument('--n_collocation', type=int, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)

    # Experiment
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--tensorboard', action='store_true', default=None)
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--mlflow', action='store_true', default=None)
    parser.add_argument('--no-mlflow', dest='mlflow', action='store_false')
    parser.add_argument('--mlflow-experiment', type=str, default=None)

    # Jacobian
    parser.add_argument('--eval-jacobian', action='store_true', default=None)
    parser.add_argument('--no-jacobian', dest='eval_jacobian', action='store_false')

    # Hardware
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Start from config file or defaults
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config: {args.config}")
    else:
        config = DEFAULT_CONFIG.copy()

    # CLI overrides (only non-None values)
    cli_map = {
        'data_path': args.data_path,
        'max_samples': args.max_samples,
        'model_type': args.model if not args.config else None,
        'activation': args.activation,
        'dropout': args.dropout,
        'lambda_pde': args.lambda_pde,
        'lambda_ic': args.lambda_ic,
        'n_collocation': args.n_collocation,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'seed': args.seed,
        'experiment_name': args.name,
        'checkpoint_dir': args.checkpoint_dir,
        'use_tensorboard': args.tensorboard,
        'use_mlflow': args.mlflow,
        'mlflow_experiment_name': args.mlflow_experiment,
        'eval_jacobian': args.eval_jacobian,
        'device': args.device,
        'num_workers': args.num_workers,
    }
    for k, v in cli_map.items():
        if v is not None:
            config[k] = v

    # Preset
    if args.preset:
        config['hidden_dims'] = get_preset_config(args.preset)['hidden_dims']
    if args.hidden_dims:
        config['hidden_dims'] = args.hidden_dims

    train(config)


if __name__ == '__main__':
    main()
