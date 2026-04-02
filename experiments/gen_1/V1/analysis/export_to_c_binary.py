#!/usr/bin/env python3
"""
Export PyTorch MLP checkpoints to C-compatible binary format.

This produces .bin files loadable by TrackMLPExtrapolator.cpp (Gaudi).
Binary format matches SimpleNN::load() in ml_models/src/TrackMLPExtrapolator.cpp.

Usage:
    python export_to_c_binary.py                           # Export all
    python export_to_c_binary.py --model mlp_2x256         # Export one
    python export_to_c_binary.py --output-dir /path/to/out # Custom output
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))
from architectures import create_model


CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / 'models' / 'checkpoints'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'exported_bins'


def load_checkpoint(ckpt_dir: Path):
    """Load model from checkpoint directory."""
    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {ckpt_dir}")

    config_path = ckpt_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = create_model(
        config['model_type'],
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        dropout=config.get('dropout', 0.0),
    )
    model.load_state_dict(ckpt['model_state_dict'])

    norm_path = ckpt_dir / 'normalization.json'
    if norm_path.exists():
        model.load_normalization(str(norm_path))

    model.eval()
    return model, config, ckpt


def export_model_binary(model, config, output_path: Path):
    """
    Export model to binary format matching SimpleNN::load() in C++.

    Binary layout:
      int model_type       (0=MLP, 1=PINN residual)
      int num_layers
      for each layer:
        int rows, int cols
        float64[rows*cols]  weights (column-major for Eigen)
        float64[rows]       biases
      int input_size
      float64[input_size]   input_mean
      float64[input_size]   input_std
      int output_size
      float64[output_size]  output_mean
      float64[output_size]  output_std
      int activation_len
      char[activation_len]  activation string
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_type = 0  # MLP
    if config.get('model_type', 'mlp') in ('pinn', 'rk_pinn'):
        model_type = 1

    # Extract linear layer weights/biases
    layers = []
    for name, param in model.named_parameters():
        if 'network' in name or 'encoder' in name:
            if 'weight' in name:
                layers.append({'weight': param.detach().cpu().numpy()})
            elif 'bias' in name:
                layers[-1]['bias'] = param.detach().cpu().numpy()

    if not layers:
        # Fallback: iterate all Linear modules
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                layers.append({
                    'weight': m.weight.detach().cpu().numpy(),
                    'bias': m.bias.detach().cpu().numpy() if m.bias is not None
                            else np.zeros(m.out_features),
                })

    num_layers = len(layers)
    activation = config.get('activation', 'silu')

    # Normalization
    input_mean = model.input_mean.cpu().numpy().astype(np.float64)
    input_std = model.input_std.cpu().numpy().astype(np.float64)
    output_mean = model.output_mean.cpu().numpy().astype(np.float64)
    output_std = model.output_std.cpu().numpy().astype(np.float64)

    with open(output_path, 'wb') as f:
        # Model type header
        f.write(struct.pack('i', model_type))
        # Number of layers
        f.write(struct.pack('i', num_layers))

        for layer in layers:
            w = layer['weight'].astype(np.float64)
            b = layer['bias'].astype(np.float64)
            rows, cols = w.shape
            f.write(struct.pack('i', rows))
            f.write(struct.pack('i', cols))
            # Eigen uses column-major by default, but SimpleNN reads
            # row-major data into Eigen (data() pointer is column-major).
            # The C++ code does: file.read(weights[i].data(), rows*cols*sizeof(double))
            # Eigen stores column-major, so we write column-major (Fortran order).
            f.write(w.T.astype(np.float64).tobytes())  # Column-major
            f.write(b.tobytes())

        # Input normalization
        f.write(struct.pack('i', len(input_mean)))
        f.write(input_mean.tobytes())
        f.write(input_std.tobytes())

        # Output normalization
        f.write(struct.pack('i', len(output_mean)))
        f.write(output_mean.tobytes())
        f.write(output_std.tobytes())

        # Activation string
        act_bytes = activation.encode('utf-8')
        f.write(struct.pack('i', len(act_bytes)))
        f.write(act_bytes)

    print(f"  Exported: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def validate_export(model, bin_path: Path, n_samples=100, tol=1e-2):
    """Validate binary export by comparing Python forward pass."""
    # Read binary back and run forward pass manually
    with open(bin_path, 'rb') as f:
        model_type = struct.unpack('i', f.read(4))[0]
        num_layers = struct.unpack('i', f.read(4))[0]

        weights = []
        biases = []
        for _ in range(num_layers):
            rows, cols = struct.unpack('ii', f.read(8))
            w_data = np.frombuffer(f.read(rows * cols * 8), dtype=np.float64)
            w = w_data.reshape(cols, rows).T  # Column-major back to row-major
            b = np.frombuffer(f.read(rows * 8), dtype=np.float64)
            weights.append(w)
            biases.append(b)

        in_size = struct.unpack('i', f.read(4))[0]
        in_mean = np.frombuffer(f.read(in_size * 8), dtype=np.float64)
        in_std = np.frombuffer(f.read(in_size * 8), dtype=np.float64)

        out_size = struct.unpack('i', f.read(4))[0]
        out_mean = np.frombuffer(f.read(out_size * 8), dtype=np.float64)
        out_std = np.frombuffer(f.read(out_size * 8), dtype=np.float64)

    # SiLU activation
    def silu(x):
        return x / (1.0 + np.exp(-x))

    # Generate test inputs
    test_input = np.random.randn(n_samples, 6).astype(np.float32)
    test_input[:, 4] *= 1e-4  # q/p scale
    test_input[:, 5] = np.random.uniform(100, 10000, n_samples)  # dz

    # C++ forward pass simulation
    c_outputs = []
    for i in range(n_samples):
        x = (test_input[i].astype(np.float64) - in_mean) / in_std
        for j in range(len(weights) - 1):
            x = weights[j] @ x + biases[j]
            x = silu(x)
        x = weights[-1] @ x + biases[-1]
        x = x * out_std + out_mean
        c_outputs.append(x)
    c_outputs = np.array(c_outputs)

    # PyTorch forward pass
    model.eval()
    with torch.no_grad():
        py_outputs = model(torch.tensor(test_input)).numpy()

    max_diff = np.abs(c_outputs - py_outputs.astype(np.float64)).max()
    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"  Validation {status}: max_diff = {max_diff:.2e} (tol={tol:.0e})")
    return passed


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to C binary')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model name (e.g. mlp_2x256)')
    parser.add_argument('--checkpoints-dir', type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_dirs = [ckpt_dir / args.model]
    else:
        model_dirs = sorted(d for d in ckpt_dir.iterdir()
                            if d.is_dir() and (d / 'best_model.pt').exists())

    print(f"Exporting {len(model_dirs)} models to {out_dir}")
    print("=" * 60)

    results = {}
    for mdir in model_dirs:
        name = mdir.name
        print(f"\n[{name}]")
        try:
            model, config, ckpt = load_checkpoint(mdir)
            bin_path = out_dir / f"{name}.bin"
            export_model_binary(model, config, bin_path)

            if args.validate:
                valid = validate_export(model, bin_path)
                results[name] = {'path': str(bin_path), 'valid': bool(valid)}
            else:
                results[name] = {'path': str(bin_path), 'valid': None}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {'path': None, 'error': str(e)}

    # Write manifest
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    n_ok = sum(1 for r in results.values() if r.get('valid'))
    n_fail = sum(1 for r in results.values() if r.get('error'))
    print(f"Done: {n_ok} exported, {n_fail} failed")


if __name__ == '__main__':
    main()
