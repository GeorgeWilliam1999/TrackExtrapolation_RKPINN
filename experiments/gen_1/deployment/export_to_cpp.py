#!/usr/bin/env python3
"""
Export PyTorch MLP models to C++ binary format for TrackMLPExtrapolator.

The binary format is:
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

Usage:
    python export_to_cpp.py <model_dir> [output_path]
    
Example:
    python export_to_cpp.py trained_models/mlp_v2_shallow_256 models/mlp_v2_shallow_256.bin
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
        # Try checkpoint
        checkpoints = list(model_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]  # Latest
        else:
            raise FileNotFoundError(f"No model checkpoint found in {model_dir}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    return checkpoint, config


def export_mlp_to_binary(checkpoint: dict, config: dict, output_path: Path, model_dir: Path = None):
    """Export MLP model to C++ binary format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = checkpoint['model_state_dict']
    
    # Get activation function from config
    activation = config.get('activation', 'tanh')
    
    # Extract layer info from state dict
    layers = []
    layer_idx = 0
    while True:
        weight_key = f'network.{layer_idx * 2}.weight'  # Linear layers at even indices
        bias_key = f'network.{layer_idx * 2}.bias'
        
        if weight_key not in state_dict:
            # Try encoder format (for PINN)
            weight_key = f'encoder.{layer_idx * 2}.weight'
            bias_key = f'encoder.{layer_idx * 2}.bias'
            if weight_key not in state_dict:
                break
        
        W = state_dict[weight_key].numpy().astype(np.float64)
        b = state_dict[bias_key].numpy().astype(np.float64)
        layers.append((W, b))
        layer_idx += 1
    
    # Check for output head (some architectures)
    if 'output_head.weight' in state_dict:
        W = state_dict['output_head.weight'].numpy().astype(np.float64)
        b = state_dict['output_head.bias'].numpy().astype(np.float64)
        layers.append((W, b))
    
    if not layers:
        raise ValueError("No layers found in model checkpoint!")
    
    # Get normalization parameters
    norm_keys = ['input_mean', 'input_std', 'output_mean', 'output_std']
    norm_params = {}
    for key in norm_keys:
        if key in state_dict:
            norm_params[key] = state_dict[key].numpy().astype(np.float64)
        elif key in checkpoint:
            norm_params[key] = np.array(checkpoint[key], dtype=np.float64)
        else:
            # Try loading from normalization.json in model directory
            norm_file = model_dir / "normalization.json" if model_dir else None
            if norm_file and norm_file.exists():
                with open(norm_file) as f:
                    norm_data = json.load(f)
                norm_params[key] = np.array(norm_data[key], dtype=np.float64)
            else:
                raise KeyError(f"Normalization parameter '{key}' not found!")
    
    # Fix near-zero std values (e.g., dz when trained with fixed value)
    # This prevents division by near-zero during inference
    for key in ['input_std', 'output_std']:
        std = norm_params[key]
        # Replace near-zero values with a sensible default
        near_zero = np.abs(std) < 1e-6
        if np.any(near_zero):
            print(f"  Warning: Found near-zero std values in {key}, replacing with corresponding mean")
            # Use the absolute value of mean as std (or 1.0 if mean is also ~0)
            mean_key = key.replace('std', 'mean')
            mean = np.abs(norm_params[mean_key])
            std[near_zero] = np.where(mean[near_zero] > 1e-6, mean[near_zero], 1.0)
            norm_params[key] = std
    
    # Write binary file
    with open(output_path, 'wb') as f:
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
        
        # Activation function (optional, for C++ loader)
        act_bytes = activation.encode('ascii')
        f.write(struct.pack('i', len(act_bytes)))
        f.write(act_bytes)
    
    print(f"✓ Exported to {output_path}")
    print(f"  Activation: {activation}")
    print(f"  Layers: {len(layers)}")
    for i, (W, b) in enumerate(layers):
        print(f"    Layer {i}: {W.shape[1]} → {W.shape[0]}")
    print(f"  Input dim: {len(input_mean)}")
    print(f"  Output dim: {len(output_mean)}")
    
    return output_path


def export_model(model_dir: str, output_path: str = None):
    """Export a single model."""
    model_dir = Path(model_dir)
    
    if output_path is None:
        output_path = model_dir / "exports" / f"{model_dir.name}.bin"
    
    checkpoint, config = load_pytorch_model(model_dir)
    return export_mlp_to_binary(checkpoint, config, output_path, model_dir)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable models:")
        models_dir = Path(__file__).parent.parent / "trained_models"
        if models_dir.exists():
            for d in sorted(models_dir.iterdir()):
                if d.is_dir() and (d / "best_model.pt").exists():
                    print(f"  {d.name}")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    export_model(model_dir, output_path)


if __name__ == "__main__":
    main()
