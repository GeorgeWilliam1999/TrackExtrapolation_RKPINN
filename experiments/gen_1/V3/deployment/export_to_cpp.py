#!/usr/bin/env python3
"""
Export V3 PyTorch models to C++ binary format.

Supports both MLP and PINN Residual architectures.

The binary format for MLP:
    - int: num_layers
    - For each layer:
        - int, int: rows, cols
        - double[rows*cols]: weights (row-major)
        - double[rows]: biases
    - int: input_size
    - double[input_size]: input_mean
    - double[input_size]: input_std
    - int: output_size
    - double[output_size]: output_mean
    - double[output_size]: output_std
    - int: activation_len
    - char[activation_len]: activation name

For PINN Residual (endpoint inference):
    - Same format as MLP
    - The network outputs corrections, C++ applies: output = input[:4] + correction
    - Flag: int (1 = residual mode)

Usage:
    python export_to_cpp.py <model_dir> [output_path]
    
Example:
    python export_to_cpp.py ../trained_models/pinn_v3_res_256_col10 ../../ml_models/models/pinn_v3_res_256_col10.bin
"""

import sys
import struct
import json
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def load_pytorch_model(model_dir: Path):
    """Load a trained PyTorch model and its config."""
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model checkpoint
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        # Try final model
        checkpoint_path = model_dir / "final_model.pt"
    if not checkpoint_path.exists():
        # Try latest checkpoint
        checkpoints = list(model_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]
        else:
            raise FileNotFoundError(f"No model checkpoint found in {model_dir}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    return checkpoint, config


def detect_model_type(checkpoint: dict, config: dict) -> str:
    """Detect if model is MLP or PINN."""
    state_dict = checkpoint['model_state_dict']
    
    # Check for PINN architecture keys
    if any(k.startswith('core.') for k in state_dict.keys()):
        return 'pinn_residual'
    elif any(k.startswith('network.') for k in state_dict.keys()):
        return 'mlp'
    elif 'architecture' in config.get('model', {}):
        return config['model']['architecture']
    else:
        # Default to MLP
        return 'mlp'


def extract_layers(state_dict: dict, prefix: str) -> list:
    """Extract layer weights and biases from state dict.
    
    Finds all Linear layers in a nn.Sequential stored under `prefix`.
    For Sequential([Linear, Activation, Linear, Activation, ..., Linear]),
    the Linear layers are at even indices (0, 2, 4, ...).
    """
    layers = []
    
    # Collect all weight keys with this prefix, sorted by index
    weight_keys = sorted(
        [k for k in state_dict.keys() if k.startswith(f'{prefix}.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[1])
    )
    
    for weight_key in weight_keys:
        bias_key = weight_key.replace('.weight', '.bias')
        if bias_key in state_dict:
            W = state_dict[weight_key].numpy().astype(np.float64)
            b = state_dict[bias_key].numpy().astype(np.float64)
            layers.append((W, b))
    
    return layers


def get_normalization(checkpoint: dict, model_dir: Path = None) -> dict:
    """Extract normalization parameters from checkpoint or file."""
    norm_params = {}
    required_keys = ['input_mean', 'input_std', 'output_mean', 'output_std']
    
    # Check in checkpoint
    if 'norm_stats' in checkpoint:
        for key in required_keys:
            if key in checkpoint['norm_stats']:
                val = checkpoint['norm_stats'][key]
                if isinstance(val, torch.Tensor):
                    val = val.numpy()
                norm_params[key] = np.array(val, dtype=np.float64)
    
    # Check in state dict
    state_dict = checkpoint.get('model_state_dict', {})
    for key in required_keys:
        if key not in norm_params and key in state_dict:
            norm_params[key] = state_dict[key].numpy().astype(np.float64)
    
    # Check for normalization.json file
    if len(norm_params) < 4 and model_dir:
        norm_file = model_dir / "normalization.json"
        if norm_file.exists():
            with open(norm_file) as f:
                norm_data = json.load(f)
            for key in required_keys:
                if key not in norm_params and key in norm_data:
                    norm_params[key] = np.array(norm_data[key], dtype=np.float64)
    
    # Validate
    for key in required_keys:
        if key not in norm_params:
            raise KeyError(f"Normalization parameter '{key}' not found!")
    
    # Fix near-zero std values
    for key in ['input_std', 'output_std']:
        std = norm_params[key]
        near_zero = np.abs(std) < 1e-8
        if np.any(near_zero):
            print(f"  Warning: Found near-zero std values in {key}, replacing with 1.0")
            std[near_zero] = 1.0
            norm_params[key] = std
    
    return norm_params


def export_mlp_to_binary(
    checkpoint: dict, 
    config: dict, 
    output_path: Path, 
    model_dir: Path = None,
    is_residual: bool = False
):
    """Export MLP or PINN model to C++ binary format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = checkpoint['model_state_dict']
    
    # Get activation function
    model_config = config.get('model', config)
    activation = model_config.get('activation', 'silu')
    
    # Detect layer prefix and extract
    if any(k.startswith('core.') for k in state_dict.keys()):
        layers = extract_layers(state_dict, 'core')
        is_residual = True
    elif any(k.startswith('network.') for k in state_dict.keys()):
        layers = extract_layers(state_dict, 'network')
    else:
        raise ValueError("Could not detect layer structure in model!")
    
    if not layers:
        raise ValueError("No layers found in model checkpoint!")
    
    # Get normalization
    norm_params = get_normalization(checkpoint, model_dir)
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Header: model type (0 = MLP, 1 = PINN residual)
        f.write(struct.pack('i', 1 if is_residual else 0))
        
        # Number of layers
        f.write(struct.pack('i', len(layers)))
        
        # Each layer
        for W, b in layers:
            rows, cols = W.shape
            f.write(struct.pack('ii', rows, cols))
            f.write(W.tobytes())  # row-major
            f.write(b.tobytes())
        
        # Input normalization
        input_mean = norm_params['input_mean']
        input_std = norm_params['input_std']
        f.write(struct.pack('i', len(input_mean)))
        f.write(input_mean.tobytes())
        f.write(input_std.tobytes())
        
        # Output normalization
        output_mean = norm_params['output_mean']
        output_std = norm_params['output_std']
        f.write(struct.pack('i', len(output_mean)))
        f.write(output_mean.tobytes())
        f.write(output_std.tobytes())
        
        # Activation function
        act_bytes = activation.encode('ascii')
        f.write(struct.pack('i', len(act_bytes)))
        f.write(act_bytes)
    
    print(f"\n✓ Exported to {output_path}")
    print(f"  Model type: {'PINN Residual' if is_residual else 'MLP'}")
    print(f"  Activation: {activation}")
    print(f"  Layers: {len(layers)}")
    for i, (W, b) in enumerate(layers):
        print(f"    Layer {i}: {W.shape[1]} → {W.shape[0]}")
    print(f"  Input dim: {len(input_mean)}")
    print(f"  Output dim: {len(output_mean)}")
    
    # File size
    size_kb = output_path.stat().st_size / 1024
    print(f"  File size: {size_kb:.1f} KB")
    
    return output_path


def export_model(model_dir: str, output_path: str = None):
    """Export a single model."""
    model_dir = Path(model_dir)
    
    if output_path is None:
        # Default: put in ml_models/models/ directory
        output_path = Path(__file__).parent.parent.parent.parent / "ml_models" / "models" / f"{model_dir.name}.bin"
    
    checkpoint, config = load_pytorch_model(model_dir)
    model_type = detect_model_type(checkpoint, config)
    
    print(f"\nExporting {model_dir.name}")
    print(f"  Detected type: {model_type}")
    
    return export_mlp_to_binary(
        checkpoint, config, output_path, model_dir,
        is_residual=(model_type == 'pinn_residual')
    )


def export_all_v3_models():
    """Export all trained V3 models."""
    trained_dir = Path(__file__).parent.parent / "trained_models"
    output_dir = Path(__file__).parent.parent.parent.parent / "ml_models" / "models"
    
    print(f"Scanning: {trained_dir}")
    print(f"Output: {output_dir}")
    
    exported = []
    for model_dir in sorted(trained_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "best_model.pt").exists() and not (model_dir / "final_model.pt").exists():
            print(f"\n⚠ Skipping {model_dir.name} (no checkpoint)")
            continue
        
        try:
            output_path = output_dir / f"{model_dir.name}.bin"
            export_model(model_dir, output_path)
            exported.append(model_dir.name)
        except Exception as e:
            print(f"\n✗ Failed to export {model_dir.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Exported {len(exported)} models:")
    for name in exported:
        print(f"  - {name}")
    
    return exported


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable models:")
        trained_dir = Path(__file__).parent.parent / "trained_models"
        if trained_dir.exists():
            for d in sorted(trained_dir.iterdir()):
                if d.is_dir():
                    has_best = (d / "best_model.pt").exists()
                    has_final = (d / "final_model.pt").exists()
                    status = "✓" if (has_best or has_final) else "✗"
                    print(f"  {status} {d.name}")
        
        print("\nUsage:")
        print("  python export_to_cpp.py <model_dir> [output_path]")
        print("  python export_to_cpp.py --all")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        export_all_v3_models()
    else:
        model_dir = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        export_model(model_dir, output_path)


if __name__ == '__main__':
    main()
