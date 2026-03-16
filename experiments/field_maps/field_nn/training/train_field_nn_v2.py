#!/usr/bin/env python3
"""
Field Map NN v2 -- Loss Function & Training Strategy Comparison

Fixed architecture: silu 2L 128H (17,411 params).
Compares 8 training strategies to fix the relative-error problem
where the NN collapses to ~0 in the strong-field region.

Strategies
----------
  1. baseline        : uniform MSE on normalised targets (gen-1 reproduction)
  2. weighted_mse    : importance-weighted MSE,  w = 1 + (|B|/B_ref)^alpha
  3. relative_mse    : MSE on (pred-true)/( |true| + eps )
  4. log_space       : MSE on sign(B)*log(|B|+1) transformed targets
  5. curriculum_lin  : 2-phase curriculum: phase-1 uniform, phase-2 linear ramp
                       of |B|-based weights from 1 -> w_max
  6. curriculum_exp  : like curriculum_lin but weights ramp exponentially
  7. mixed_loss      : 0.5*MSE_norm  +  0.5*relative_MSE
  8. weighted_mse_strong : heavier weighting,  w = 1 + (|B|/B_ref)^2

Each writes the same output artefacts as train_field_nn.py:
  model.pt, normalization.json, model_config.json, config.json,
  history.json, exports/field_nn.onnx, exports/field_nn_weights.h

Usage
-----
  python train_field_nn_v2.py --config configs_v2/loss_baseline.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trilinear import TrilinearGrid


# ── Model (identical to v1) ──────────────────────────────────────────────────

class FieldMLP(nn.Module):
    def __init__(self, hidden_dims, activation='relu'):
        super().__init__()
        act_map = {'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        act_fn = act_map[activation]
        layers = []
        prev = 3
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(hidden_dims, activation='relu'):
    layers_list = [3] + list(hidden_dims) + [3]
    flops = 0
    for i in range(len(layers_list) - 1):
        flops += layers_list[i] * layers_list[i + 1]
        flops += (layers_list[i] - 1) * layers_list[i + 1]
        flops += layers_list[i + 1]
    act_cost = {'relu': 1, 'silu': 4, 'tanh': 5, 'gelu': 5}
    for h in hidden_dims:
        flops += h * act_cost.get(activation, 1)
    return flops


# ── Loss helpers ─────────────────────────────────────────────────────────────

def compute_sample_weights(Y_raw, strategy_cfg):
    """Pre-compute per-sample importance weights from *physical* targets.

    Parameters
    ----------
    Y_raw : (N, 3) tensor in Gauss (un-normalised)
    strategy_cfg : dict with strategy-specific parameters

    Returns
    -------
    weights : (N,) tensor  (mean ≈ 1)
    """
    B_mag = Y_raw.norm(dim=1)  # |B| per sample in Gauss
    strategy = strategy_cfg['strategy']

    if strategy in ('baseline', 'relative_mse', 'log_space', 'mixed_loss'):
        return torch.ones(len(Y_raw))

    B_ref = strategy_cfg.get('B_ref', 10.0)   # reference scale in Gauss
    alpha = strategy_cfg.get('alpha', 1.0)

    if strategy in ('weighted_mse', 'weighted_mse_strong'):
        w = 1.0 + (B_mag / B_ref) ** alpha
    elif strategy.startswith('curriculum'):
        # Phase-1 weights are uniform; phase-2 weights computed per-epoch
        w = torch.ones(len(Y_raw))
    else:
        w = torch.ones(len(Y_raw))

    # Normalise so mean(w) = 1  →  comparable learning rates across strategies
    w = w / w.mean()
    return w


def make_loss_fn(strategy_cfg, Y_std, Y_mean):
    """Return a loss function  f(pred_norm, target_norm, weights, epoch, total_epochs).

    All inputs are in normalised space except `weights` which are in physical space.
    """
    strategy = strategy_cfg['strategy']
    eps = strategy_cfg.get('eps', 1.0)         # Gauss, for relative_mse denom

    if strategy == 'baseline':
        # Vanilla uniform MSE in normalised space (gen-1 equivalent)
        def loss_fn(pred, target, weights, epoch, total_epochs):
            return nn.functional.mse_loss(pred, target)
        return loss_fn

    elif strategy in ('weighted_mse', 'weighted_mse_strong'):
        # Weighted MSE: per-sample weight * MSE, in normalised space
        def loss_fn(pred, target, weights, epoch, total_epochs):
            se = ((pred - target) ** 2).sum(dim=1)  # (N,)
            return (se * weights).mean()
        return loss_fn

    elif strategy == 'relative_mse':
        # MSE on (pred - true) / (|true| + eps)  in *physical* space
        def loss_fn(pred_norm, target_norm, weights, epoch, total_epochs):
            pred_phys = pred_norm * Y_std + Y_mean
            true_phys = target_norm * Y_std + Y_mean
            denom = true_phys.norm(dim=1, keepdim=True).clamp(min=eps)
            rel_err = (pred_phys - true_phys) / denom
            return (rel_err ** 2).mean()
        return loss_fn

    elif strategy == 'log_space':
        # MSE on sign(B)*log(|B|+1) transformed targets, computed in physical space
        def log_transform(B_phys):
            return torch.sign(B_phys) * torch.log1p(B_phys.abs())

        def loss_fn(pred_norm, target_norm, weights, epoch, total_epochs):
            pred_phys = pred_norm * Y_std + Y_mean
            true_phys = target_norm * Y_std + Y_mean
            return nn.functional.mse_loss(log_transform(pred_phys),
                                          log_transform(true_phys))
        return loss_fn

    elif strategy == 'mixed_loss':
        # 0.5 * MSE_norm  +  0.5 * relative_MSE_phys
        mix_alpha = strategy_cfg.get('mix_alpha', 0.5)
        def loss_fn(pred_norm, target_norm, weights, epoch, total_epochs):
            mse_norm = nn.functional.mse_loss(pred_norm, target_norm)
            pred_phys = pred_norm * Y_std + Y_mean
            true_phys = target_norm * Y_std + Y_mean
            denom = true_phys.norm(dim=1, keepdim=True).clamp(min=eps)
            rel_err = (pred_phys - true_phys) / denom
            mse_rel = (rel_err ** 2).mean()
            return mix_alpha * mse_norm + (1.0 - mix_alpha) * mse_rel
        return loss_fn

    elif strategy == 'curriculum_lin':
        # Phase 1 (epochs 0..phase1_frac): uniform MSE
        # Phase 2 (remaining): weight linearly ramps from 1 to w_max
        phase1_frac = strategy_cfg.get('phase1_frac', 0.3)
        B_ref = strategy_cfg.get('B_ref', 10.0)
        w_max = strategy_cfg.get('w_max', 50.0)

        def loss_fn(pred_norm, target_norm, weights, epoch, total_epochs):
            se = ((pred_norm - target_norm) ** 2).sum(dim=1)
            phase1_end = int(phase1_frac * total_epochs)
            if epoch < phase1_end:
                return se.mean()
            else:
                # Linear ramp: progress goes from 0→1 over phase 2
                progress = (epoch - phase1_end) / max(total_epochs - phase1_end - 1, 1)
                # Current max weight factor
                cur_wmax = 1.0 + progress * (w_max - 1.0)
                w = 1.0 + (weights - 1.0) * (cur_wmax - 1.0) / max(w_max - 1.0, 1e-8)
                w = w / w.mean()
                return (se * w).mean()
        return loss_fn

    elif strategy == 'curriculum_exp':
        # Phase 1: uniform MSE
        # Phase 2: weight exponentially ramps
        phase1_frac = strategy_cfg.get('phase1_frac', 0.3)
        B_ref = strategy_cfg.get('B_ref', 10.0)
        w_max = strategy_cfg.get('w_max', 50.0)

        def loss_fn(pred_norm, target_norm, weights, epoch, total_epochs):
            se = ((pred_norm - target_norm) ** 2).sum(dim=1)
            phase1_end = int(phase1_frac * total_epochs)
            if epoch < phase1_end:
                return se.mean()
            else:
                progress = (epoch - phase1_end) / max(total_epochs - phase1_end - 1, 1)
                # Exponential ramp: w_factor = exp(progress * ln(w_max))
                cur_wmax = math.exp(progress * math.log(w_max))
                w = 1.0 + (weights - 1.0) * (cur_wmax - 1.0) / max(w_max - 1.0, 1e-8)
                w = w / w.mean()
                return (se * w).mean()
        return loss_fn

    else:
        raise ValueError(f'Unknown strategy: {strategy}')


# ── Validation metrics (always computed identically) ─────────────────────────

def compute_val_metrics(model, X_val_dev, Y_val, Y_std, Y_mean, Y_val_dev):
    """Return dict of validation metrics (in Gauss + normalised MSE)."""
    model.eval()
    with torch.no_grad():
        val_pred_norm = model(X_val_dev)
        val_mse = nn.functional.mse_loss(val_pred_norm, Y_val_dev).item()

        val_pred_phys = val_pred_norm.cpu() * Y_std + Y_mean
        val_true_phys = Y_val * Y_std + Y_mean
        errors_gauss = (val_pred_phys - val_true_phys).abs()

        mae_gauss = errors_gauss.mean().item()
        p99_gauss = float(torch.quantile(errors_gauss.flatten(), 0.99))
        max_gauss = errors_gauss.max().item()

        # Relative error on |B| (where |B| > 1 G)
        B_mag_true = val_true_phys.norm(dim=1)
        B_mag_pred = val_pred_phys.norm(dim=1)
        mask = B_mag_true > 1.0
        if mask.sum() > 0:
            rel_err_mag = ((B_mag_pred[mask] - B_mag_true[mask]) / B_mag_true[mask]).abs()
            rel_mae = rel_err_mag.mean().item()
            rel_p95 = float(torch.quantile(rel_err_mag, 0.95))
            rel_max = rel_err_mag.max().item()
        else:
            rel_mae = rel_p95 = rel_max = float('nan')

    return {
        'val_mse': val_mse,
        'mae_gauss': mae_gauss,
        'p99_gauss': p99_gauss,
        'max_gauss': max_gauss,
        'rel_mae': rel_mae,
        'rel_p95': rel_p95,
        'rel_max': rel_max,
    }


# ── Main training loop ──────────────────────────────────────────────────────

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:    {torch.cuda.get_device_name(0)}')

    strategy_cfg = config.get('strategy_cfg', {'strategy': 'baseline'})
    strategy = strategy_cfg['strategy']
    print(f'Strategy: {strategy}')
    print(f'  Config: {json.dumps(strategy_cfg, indent=4)}')

    # ── Load data ────────────────────────────────────────────────────────────
    field_map_path = config['field_map_path']
    print(f'Loading field map: {field_map_path}')
    t0 = time.time()
    grid = TrilinearGrid.from_file(field_map_path)
    print(f'  Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.nx*grid.ny*grid.nz:,} pts'
          f'  ({time.time()-t0:.1f}s)')

    data = np.loadtxt(field_map_path)
    coords, fields = data[:, :3], data[:, 3:]

    X_train_raw = torch.tensor(coords, dtype=torch.float32)
    Y_train_raw = torch.tensor(fields, dtype=torch.float32)

    X_mean, X_std = X_train_raw.mean(0), X_train_raw.std(0)
    Y_mean, Y_std = Y_train_raw.mean(0), Y_train_raw.std(0)

    X_train = (X_train_raw - X_mean) / X_std
    Y_train = (Y_train_raw - Y_mean) / Y_std

    # Pre-compute sample weights from *physical* field magnitudes
    sample_weights = compute_sample_weights(Y_train_raw, strategy_cfg)
    print(f'  Training samples: {len(X_train):,}')
    print(f'  Weight stats: min={sample_weights.min():.3f}  '
          f'median={sample_weights.median():.3f}  '
          f'max={sample_weights.max():.1f}  mean={sample_weights.mean():.3f}')

    # Validation data
    n_val = config.get('n_val', 100_000)
    rng = np.random.default_rng(config.get('val_seed', 42))
    val_pts = grid.random_off_grid_points(n_val, rng=rng)
    val_B = grid.query(val_pts)

    X_val_raw = torch.tensor(val_pts, dtype=torch.float32)
    Y_val_raw = torch.tensor(val_B, dtype=torch.float32)
    X_val = (X_val_raw - X_mean) / X_std
    Y_val = (Y_val_raw - Y_mean) / Y_std
    print(f'  Validation samples: {n_val:,}')

    # ── Model ────────────────────────────────────────────────────────────────
    hidden_dims = config['hidden_dims']
    activation = config.get('activation', 'silu')
    model = FieldMLP(hidden_dims, activation).to(device)
    n_params = count_params(model)
    n_flops = count_flops(hidden_dims, activation)
    print(f'  Model: {hidden_dims} {activation}  ({n_params:,} params, {n_flops:,} FLOPs)')

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    epochs = config.get('epochs', 300)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 4096)
    patience = config.get('patience', 30)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build loss function (closure over Y_std, Y_mean for physical-space losses)
    Y_std_dev = Y_std.to(device)
    Y_mean_dev = Y_mean.to(device)
    loss_fn = make_loss_fn(strategy_cfg, Y_std_dev, Y_mean_dev)

    # DataLoader with weights
    train_ds = TensorDataset(X_train, Y_train, sample_weights)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_mse = float('inf')
    best_state = None
    no_improve = 0

    history = {
        'train_loss': [], 'val_mse': [],
        'val_mae_gauss': [], 'val_p99_gauss': [], 'val_max_gauss': [],
        'val_rel_mae': [], 'val_rel_p95': [], 'val_rel_max': [],
        'lr': [],
    }

    print(f'\nTraining: {epochs} epochs, batch={batch_size}, lr={lr}, patience={patience}')
    print('=' * 100)

    t_start = time.time()
    X_val_dev = X_val.to(device)
    Y_val_dev = Y_val.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb, wb, epoch, epochs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n_samples += len(xb)

        train_loss = epoch_loss / n_samples
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Validate
        metrics = compute_val_metrics(model, X_val_dev, Y_val, Y_std, Y_mean, Y_val_dev)

        history['train_loss'].append(train_loss)
        history['val_mse'].append(metrics['val_mse'])
        history['val_mae_gauss'].append(metrics['mae_gauss'])
        history['val_p99_gauss'].append(metrics['p99_gauss'])
        history['val_max_gauss'].append(metrics['max_gauss'])
        history['val_rel_mae'].append(metrics['rel_mae'])
        history['val_rel_p95'].append(metrics['rel_p95'])
        history['val_rel_max'].append(metrics['rel_max'])
        history['lr'].append(current_lr)

        if metrics['val_mse'] < best_val_mse:
            best_val_mse = metrics['val_mse']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = ' *'
        else:
            no_improve += 1
            marker = ''

        if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
            print(f'  Ep {epoch+1:3d}/{epochs}  '
                  f'loss={train_loss:.6f}  valMSE={metrics["val_mse"]:.6f}  '
                  f'MAE={metrics["mae_gauss"]:.2f}G  '
                  f'max={metrics["max_gauss"]:.0f}G  '
                  f'relMAE={metrics["rel_mae"]:.3f}  '
                  f'lr={current_lr:.1e}{marker}')

        if no_improve >= patience:
            print(f'  Early stop at epoch {epoch+1} (no improvement for {patience})')
            break

    train_time = time.time() - t_start
    print(f'\nDone in {train_time:.1f}s  ({epoch+1} epochs)')

    # ── Final evaluation ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    final = compute_val_metrics(model, X_val_dev, Y_val, Y_std, Y_mean, Y_val_dev)

    # Per-component breakdown
    model.eval()
    with torch.no_grad():
        val_pred_phys = (model(X_val_dev).cpu() * Y_std + Y_mean)
        val_true_phys = Y_val * Y_std + Y_mean
        errors_gauss = (val_pred_phys - val_true_phys).abs()
        per_comp = {}
        for i, comp in enumerate(['Bx', 'By', 'Bz']):
            comp_err = errors_gauss[:, i]
            per_comp[comp] = {
                'mae_gauss': comp_err.mean().item(),
                'p99_gauss': float(torch.quantile(comp_err, 0.99)),
                'max_gauss': comp_err.max().item(),
            }

    print(f'\nFinal (best checkpoint):')
    print(f'  MAE:      {final["mae_gauss"]:.2f} Gauss')
    print(f'  p99:      {final["p99_gauss"]:.1f} Gauss')
    print(f'  Max:      {final["max_gauss"]:.0f} Gauss')
    print(f'  Rel MAE:  {final["rel_mae"]:.4f}  (|B|>1G points)')
    print(f'  Rel p95:  {final["rel_p95"]:.4f}')
    print(f'  Rel max:  {final["rel_max"]:.4f}')
    for comp, v in per_comp.items():
        print(f'  {comp}: MAE={v["mae_gauss"]:.2f}  p99={v["p99_gauss"]:.1f}  '
              f'max={v["max_gauss"]:.0f} G')

    # ── Save ─────────────────────────────────────────────────────────────────
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'exports'), exist_ok=True)

    torch.save(best_state, os.path.join(output_dir, 'model.pt'))

    norm = {
        'input_mean': X_mean.numpy().tolist(),
        'input_std': X_std.numpy().tolist(),
        'output_mean': Y_mean.numpy().tolist(),
        'output_std': Y_std.numpy().tolist(),
    }
    with open(os.path.join(output_dir, 'normalization.json'), 'w') as f:
        json.dump(norm, f, indent=2)

    model_cfg = {
        'model_type': 'FieldMLP',
        'hidden_dims': hidden_dims,
        'activation': activation,
        'parameters': n_params,
        'flops': n_flops,
        'weight_bytes_f32': n_params * 4,
        'fits_L1_32KB': (n_params * 4) <= 32 * 1024,
    }
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_cfg, f, indent=2)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    history_out = {
        'history': history,
        'train_time_s': train_time,
        'epochs_run': epoch + 1,
        'best_epoch': int(np.argmin(history['val_mse'])) + 1,
        'final_metrics': {
            'mae_gauss': final['mae_gauss'],
            'p99_gauss': final['p99_gauss'],
            'max_gauss': final['max_gauss'],
            'rel_mae': final['rel_mae'],
            'rel_p95': final['rel_p95'],
            'rel_max': final['rel_max'],
            'per_component': per_comp,
            'val_mse_normalised': best_val_mse,
        },
        'device': str(device),
        'strategy': strategy,
    }
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history_out, f, indent=2)

    # ONNX export
    try:
        model_cpu = FieldMLP(hidden_dims, activation)
        model_cpu.load_state_dict(best_state)
        model_cpu.eval()
        dummy = torch.randn(1, 3)
        onnx_path = os.path.join(output_dir, 'exports', 'field_nn.onnx')
        torch.onnx.export(model_cpu, dummy, onnx_path,
                          input_names=['coords'], output_names=['field'],
                          dynamic_axes={'coords': {0: 'batch'}, 'field': {0: 'batch'}},
                          opset_version=13)
        print(f'  ONNX exported: {onnx_path}')
    except Exception as e:
        print(f'  ONNX export failed: {e}')

    # C header export
    try:
        header_path = os.path.join(output_dir, 'exports', 'field_nn_weights.h')
        with open(header_path, 'w') as hf:
            hf.write(f'// Auto-generated: {hidden_dims} {activation}\n')
            hf.write(f'// Strategy: {strategy}\n')
            hf.write(f'// Params: {n_params}, FLOPs: {n_flops}\n')
            hf.write(f'// MAE: {final["mae_gauss"]:.2f} G, relMAE: {final["rel_mae"]:.4f}\n\n')
            hf.write('#pragma once\n\n')
            model_cpu = FieldMLP(hidden_dims, activation)
            model_cpu.load_state_dict(best_state)
            for name, param in model_cpu.named_parameters():
                arr = param.detach().numpy().flatten()
                cname = name.replace('.', '_')
                hf.write(f'static const float {cname}[{len(arr)}] = {{\n  ')
                hf.write(', '.join(f'{v:.8f}' for v in arr))
                hf.write('\n};\n\n')
            hf.write(f'static const float input_mean[3] = '
                     f'{{{", ".join(f"{v:.8f}" for v in norm["input_mean"])}}};\n')
            hf.write(f'static const float input_std[3]  = '
                     f'{{{", ".join(f"{v:.8f}" for v in norm["input_std"])}}};\n')
            hf.write(f'static const float output_mean[3] = '
                     f'{{{", ".join(f"{v:.8f}" for v in norm["output_mean"])}}};\n')
            hf.write(f'static const float output_std[3]  = '
                     f'{{{", ".join(f"{v:.8f}" for v in norm["output_std"])}}};\n')
        print(f'  C header exported: {header_path}')
    except Exception as e:
        print(f'  C header export failed: {e}')

    print(f'\nAll outputs saved to: {output_dir}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train field NN v2 (loss comparison)')
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    if args.epochs is not None:
        config['epochs'] = args.epochs

    print(f'Config: {args.config}')
    print(json.dumps(config, indent=2))
    sys.exit(train(config))
