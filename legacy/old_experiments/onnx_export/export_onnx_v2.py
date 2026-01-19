"""
Export trained models to ONNX format for C++ inference.
"""

import numpy as np
import torch
import torch.nn as nn
import struct
from pathlib import Path

BASE_DIR = Path('/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators')
MODELS_DIR = BASE_DIR / 'ml_models' / 'models'
EXPORT_DIR = BASE_DIR / 'experiments' / 'onnx_export'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class TrackMLP(nn.Module):
    """Track extrapolation MLP."""
    def __init__(self, hidden_dims=[256, 256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = 6
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.Tanh()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        self.network = nn.Sequential(*layers)
        
        self.register_buffer('input_mean', torch.zeros(6))
        self.register_buffer('input_std', torch.ones(6))
        self.register_buffer('output_mean', torch.zeros(4))
        self.register_buffer('output_std', torch.ones(4))
        
    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        out = self.network(x_norm)
        return out * self.output_std + self.output_mean


def load_model_binary(filepath):
    """Load model from binary file."""
    model = TrackMLP()
    with open(filepath, 'rb') as f:
        n_layers = struct.unpack('i', f.read(4))[0]
        linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
        for layer in linear_layers:
            rows, cols = struct.unpack('ii', f.read(8))
            W = np.frombuffer(f.read(rows * cols * 8), dtype=np.float64).reshape(rows, cols)
            b = np.frombuffer(f.read(rows * 8), dtype=np.float64)
            layer.weight.data = torch.FloatTensor(W)
            layer.bias.data = torch.FloatTensor(b)
        
        for name in ['input_mean', 'input_std']:
            n = struct.unpack('i', f.read(4))[0]
            arr = np.frombuffer(f.read(n * 8), dtype=np.float64)
            setattr(model, name, torch.FloatTensor(arr))
        for name in ['output_mean', 'output_std']:
            n = struct.unpack('i', f.read(4))[0]
            arr = np.frombuffer(f.read(n * 8), dtype=np.float64)
            setattr(model, name, torch.FloatTensor(arr))
    return model


class TrackMLPExport(nn.Module):
    """Wrapper that includes normalization for clean ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.network = model.network
        self.input_mean = model.input_mean
        self.input_std = model.input_std
        self.output_mean = model.output_mean
        self.output_std = model.output_std
    
    def forward(self, x):
        # Input: [batch, 6] = [x, y, tx, ty, qop, dz]
        x_norm = (x - self.input_mean) / self.input_std
        out = self.network(x_norm)
        # Output: [batch, 4] = [x_out, y_out, tx_out, ty_out]
        return out * self.output_std + self.output_mean


def export_to_onnx(model, output_path, model_name="track_mlp"):
    """Export model to ONNX format."""
    export_model = TrackMLPExport(model)
    export_model.eval()
    
    # Dummy input: batch of 1 track
    dummy_input = torch.randn(1, 6)
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Exported {model_name} to {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model verified successfully!")
    except ImportError:
        print("  (onnx package not installed, skipping verification)")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")


def export_normalization_params(model, output_path):
    """Export normalization parameters separately for manual loading."""
    params = {
        'input_mean': model.input_mean.numpy().tolist(),
        'input_std': model.input_std.numpy().tolist(),
        'output_mean': model.output_mean.numpy().tolist(),
        'output_std': model.output_std.numpy().tolist(),
    }
    import json
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Exported normalization params to {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)
    
    # Export MLP full domain model
    mlp_path = MODELS_DIR / 'mlp_full_domain.bin'
    if mlp_path.exists():
        print(f"\nLoading {mlp_path}...")
        mlp_model = load_model_binary(str(mlp_path))
        
        export_to_onnx(mlp_model, str(EXPORT_DIR / 'mlp_full_domain.onnx'), "MLP Full Domain")
        export_normalization_params(mlp_model, str(EXPORT_DIR / 'mlp_full_domain_norm.json'))
    else:
        print(f"Warning: {mlp_path} not found")
    
    # Export PINN full domain model
    pinn_path = MODELS_DIR / 'pinn_full_domain.bin'
    if pinn_path.exists():
        print(f"\nLoading {pinn_path}...")
        pinn_model = load_model_binary(str(pinn_path))
        
        export_to_onnx(pinn_model, str(EXPORT_DIR / 'pinn_full_domain.onnx'), "PINN Full Domain")
        export_normalization_params(pinn_model, str(EXPORT_DIR / 'pinn_full_domain_norm.json'))
    else:
        print(f"Warning: {pinn_path} not found")
    
    print("\nDone!")
