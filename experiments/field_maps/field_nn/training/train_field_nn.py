#!/usr/bin/env python3
"""
Field Map NN Grid Search -- Training Script

Trains a single FieldMLP model to approximate (x,y,z) -> (Bx,By,Bz).
  - Trains on 100% of the grid vertices
  - Validates on 100k random off-grid points using trilinear interpolation
  - Saves model, normalization, config, history, and ONNX export

Usage:
    python train_field_nn.py --config configs/field_nn_relu_1L_64H.json
    python train_field_nn.py --config configs/field_nn_relu_1L_64H.json --epochs 2  # quick test
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add training dir to path for trilinear import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trilinear import TrilinearGrid


# -- Model --

class FieldMLP(nn.Module):
    def __init__(self, hidden_dims, activation='relu'):
        super().__init__()
        act_map = {'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        act_fn = act_map[activation]

        layers = []
        prev = 3  # input: x, y, z
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        layers.append(nn.Linear(prev, 3))  # output: Bx, By, Bz
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(hidden_dims, activation='relu'):
    """Estimate forward-pass FLOPs for a single sample."""
    layers_list = [3] + list(hidden_dims) + [3]
    flops = 0
    for i in range(len(layers_list) - 1):
        flops += layers_list[i] * layers_list[i+1]      # multiply
        flops += (layers_list[i] - 1) * layers_list[i+1] # add
        flops += layers_list[i+1]                         # bias
    act_cost = {'relu': 1, 'silu': 4, 'tanh': 5, 'gelu': 5}
    for h in hidden_dims:
        flops += h * act_cost.get(activation, 1)
    return flops


# -- Training --

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:    {torch.cuda.get_device_name(0)}')

    # Load field map grid
    field_map_path = config['field_map_path']
    print(f'Loading field map: {field_map_path}')
    t0 = time.time()
    grid = TrilinearGrid.from_file(field_map_path)
    print(f'  Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.nx*grid.ny*grid.nz:,} points'
          f'  ({time.time()-t0:.1f}s)')

    # Training data: ALL grid vertices
    data = np.loadtxt(field_map_path)
    coords = data[:, :3]
    fields = data[:, 3:]

    X_train_raw = torch.tensor(coords, dtype=torch.float32)
    Y_train_raw = torch.tensor(fields, dtype=torch.float32)

    X_mean, X_std = X_train_raw.mean(0), X_train_raw.std(0)
    Y_mean, Y_std = Y_train_raw.mean(0), Y_train_raw.std(0)

    X_train = (X_train_raw - X_mean) / X_std
    Y_train = (Y_train_raw - Y_mean) / Y_std

    print(f'  Training samples: {len(X_train):,} (all grid vertices)')

    # Validation data: off-grid points with trilinear ground truth
    n_val = config.get('n_val', 100_000)
    rng = np.random.default_rng(config.get('val_seed', 42))
    val_pts = grid.random_off_grid_points(n_val, rng=rng)
    val_B = grid.query(val_pts)

    X_val_raw = torch.tensor(val_pts, dtype=torch.float32)
    Y_val_raw = torch.tensor(val_B, dtype=torch.float32)
    X_val = (X_val_raw - X_mean) / X_std
    Y_val = (Y_val_raw - Y_mean) / Y_std

    print(f'  Validation samples: {n_val:,} (random off-grid, trilinear ground truth)')

    # Model
    hidden_dims = config['hidden_dims']
    activation = config.get('activation', 'relu')
    model = FieldMLP(hidden_dims, activation).to(device)
    n_params = count_params(model)
    n_flops = count_flops(hidden_dims, activation)
    print(f'  Model: {hidden_dims}, activation={activation}')
    print(f'  Parameters: {n_params:,}  FLOPs: {n_flops:,}')

    # Training setup
    epochs = config.get('epochs', 200)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 4096)
    patience = config.get('patience', 20)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    # Training loop
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {
        'train_mse': [], 'val_mse': [],
        'val_mae_gauss': [], 'val_p99_gauss': [], 'val_max_gauss': [],
        'lr': [],
    }

    print(f'\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}, patience={patience}')
    print('=' * 80)

    t_start = time.time()
    X_val_dev = X_val.to(device)
    Y_val_dev = Y_val.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n_samples += len(xb)
        train_mse = epoch_loss / n_samples
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(X_val_dev)
            val_mse = nn.functional.mse_loss(val_pred_norm, Y_val_dev).item()

            val_pred_phys = val_pred_norm.cpu() * Y_std + Y_mean
            val_true_phys = Y_val * Y_std + Y_mean
            # Field map data is already in Gauss (not Tesla)
            errors_gauss = (val_pred_phys - val_true_phys).abs()

            mae_gauss = errors_gauss.mean().item()
            p99_gauss = float(torch.quantile(errors_gauss.flatten(), 0.99))
            max_gauss = errors_gauss.max().item()

        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['val_mae_gauss'].append(mae_gauss)
        history['val_p99_gauss'].append(p99_gauss)
        history['val_max_gauss'].append(max_gauss)
        history['lr'].append(current_lr)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = ' *'
        else:
            no_improve += 1
            marker = ''

        if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
            print(f'  Epoch {epoch+1:3d}/{epochs}  '
                  f'train={train_mse:.6f}  val={val_mse:.6f}  '
                  f'MAE={mae_gauss:.2f}G  p99={p99_gauss:.1f}G  '
                  f'max={max_gauss:.1f}G  lr={current_lr:.1e}{marker}')

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break

    train_time = time.time() - t_start
    print(f'\nTraining complete in {train_time:.1f}s')

    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred_norm = model(X_val_dev)
        val_pred_phys = val_pred_norm.cpu() * Y_std + Y_mean
        val_true_phys = Y_val * Y_std + Y_mean
        # Field map data is already in Gauss (not Tesla)
        errors_gauss = (val_pred_phys - val_true_phys).abs()

        final_mae = errors_gauss.mean().item()
        final_p99 = float(torch.quantile(errors_gauss.flatten(), 0.99))
        final_max = errors_gauss.max().item()

        per_comp = {}
        for i, comp in enumerate(['Bx', 'By', 'Bz']):
            comp_err = errors_gauss[:, i]
            per_comp[comp] = {
                'mae_gauss': comp_err.mean().item(),
                'p99_gauss': float(torch.quantile(comp_err, 0.99)),
                'max_gauss': comp_err.max().item(),
            }

    print(f'\nFinal results (best model):')
    print(f'  MAE:  {final_mae:.2f} Gauss')
    print(f'  p99:  {final_p99:.1f} Gauss')
    print(f'  Max:  {final_max:.1f} Gauss')
    for comp, vals in per_comp.items():
        print(f'  {comp}: MAE={vals["mae_gauss"]:.2f}  p99={vals["p99_gauss"]:.1f}  max={vals["max_gauss"]:.1f} Gauss')

    # Save outputs
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
            'mae_gauss': final_mae,
            'p99_gauss': final_p99,
            'max_gauss': final_max,
            'per_component': per_comp,
            'val_mse_normalised': best_val_loss,
        },
        'device': str(device),
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
        with open(header_path, 'w') as f:
            f.write(f'// Auto-generated field NN weights\n')
            f.write(f'// Architecture: {hidden_dims}, activation={activation}\n')
            f.write(f'// Parameters: {n_params}, FLOPs: {n_flops}\n')
            f.write(f'// MAE: {final_mae:.2f} Gauss, p99: {final_p99:.1f} Gauss\n\n')
            f.write('#pragma once\n\n')

            model_cpu = FieldMLP(hidden_dims, activation)
            model_cpu.load_state_dict(best_state)
            for name, param in model_cpu.named_parameters():
                arr = param.detach().numpy().flatten()
                cname = name.replace('.', '_')
                f.write(f'static const float {cname}[{len(arr)}] = {{\n  ')
                f.write(', '.join(f'{v:.8f}' for v in arr))
                f.write('\n};\n\n')

            f.write(f'static const float input_mean[3] = {{{", ".join(f"{v:.8f}" for v in norm["input_mean"])}}};\n')
            f.write(f'static const float input_std[3]  = {{{", ".join(f"{v:.8f}" for v in norm["input_std"])}}};\n')
            f.write(f'static const float output_mean[3] = {{{", ".join(f"{v:.8f}" for v in norm["output_mean"])}}};\n')
            f.write(f'static const float output_std[3]  = {{{", ".join(f"{v:.8f}" for v in norm["output_std"])}}};\n')
        print(f'  C header exported: {header_path}')
    except Exception as e:
        print(f'  C header export failed: {e}')

    print(f'\nAll outputs saved to: {output_dir}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train field map NN')
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs (for testing)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.epochs is not None:
        config['epochs'] = args.epochs

    print(f'Config: {args.config}')
    print(f'  {json.dumps(config, indent=2)}')

    sys.exit(train(config))
