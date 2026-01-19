# Trained Models

This directory contains trained neural network models for track extrapolation.

---

## 📂 Directory Structure

```
models/
├── README.md                       # This file
│
├── analysis/                       # Latest analysis models (PyTorch format)
│   ├── mlp_act_silu.pt             # ⭐ BEST: 0.21mm mean error
│   ├── mlp_act_tanh.pt             # Baseline: 0.63mm
│   ├── mlp_act_relu.pt             # ReLU: 0.77mm
│   ├── pinn_lambda_0_01.pt         # PINN λ=0.01: 18.8mm (failed)
│   ├── pinn_lambda_0_05.pt         # PINN λ=0.05: 106mm (failed)
│   ├── pinn_lambda_0_1.pt          # PINN λ=0.1: 197mm (failed)
│   ├── pinn_lambda_0_2.pt          # PINN λ=0.2: 329mm (failed)
│   └── *_metadata.json             # Training config for each model
│
├── production/                     # Production models (pending)
│   └── (empty - training in progress)
│
├── mlp_model_cpp_v2.bin            # Legacy C++ format (raw floats)
├── pinn_model_true.bin             # Legacy PINN (raw floats)
├── config.json                     # Model configuration
├── mlp_config.json
└── full_domain_results.json        # Benchmark results
```

---

## 📊 Available Models

### Analysis Models (Latest)

| File | Architecture | Activation | Mean Error | P95 | Use For |
|------|--------------|------------|------------|-----|---------|
| `mlp_act_silu.pt` ⭐ | 128-128-64 | SiLU | **0.21 mm** | 0.54 mm | **Production** |
| `mlp_act_tanh.pt` | 128-128-64 | Tanh | 0.63 mm | 1.68 mm | Baseline |
| `mlp_act_relu.pt` | 128-128-64 | ReLU | 0.77 mm | 1.78 mm | Testing |
| `pinn_lambda_*` | 128-128-64 | Tanh | >18 mm | >47 mm | ❌ Do not use |

### Missing Models

These should be trained but don't exist yet:
- `mlp_tiny.pt` - 64-32 architecture
- `mlp_small.pt` - 128-64 architecture
- `mlp_medium.pt` - 128-128-64 (same as act_silu)
- `mlp_large.pt` - 256-256-128-64
- `mlp_xlarge.pt` - 512-512-256-128

**To train:** Run `python ../python/train_analysis_models.py`

---

## 🔧 File Formats

### PyTorch Format (`.pt`)

Standard PyTorch state dict:

```python
import torch

# Load model
model = TrackMLP(hidden_sizes=[128, 128, 64], activation='silu')
model.load_state_dict(torch.load('analysis/mlp_act_silu.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(input_tensor)  # [N, 6] → [N, 4]
```

### Metadata (`.json`)

Training configuration and metrics:

```json
{
    "name": "mlp_act_silu",
    "architecture": [128, 128, 64],
    "activation": "silu",
    "input_dim": 6,
    "output_dim": 4,
    "epochs": 2000,
    "final_loss": 0.0001,
    "mean_error_mm": 0.207,
    "p95_error_mm": 0.543,
    "training_samples": 40000,
    "test_samples": 10000,
    "date": "2025-01-13"
}
```

### Legacy Binary (`.bin`)

Raw float32 arrays for C++ inference:

```
Header: [input_dim, output_dim, n_layers, hidden_1, hidden_2, ...]
Weights: [W0 flat, b0 flat, W1 flat, b1 flat, ...]
```

See `../src/TrackMLPExtrapolator.cpp` for C++ loading code.

---

## 🚀 Usage

### Python

```python
import torch
import numpy as np

# Define model class (same as training)
class TrackMLP(torch.nn.Module):
    def __init__(self, input_dim=6, output_dim=4, hidden_sizes=[128, 128, 64]):
        super().__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.SiLU())
            prev_size = size
        layers.append(torch.nn.Linear(prev_size, output_dim))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Load best model
model = TrackMLP()
model.load_state_dict(torch.load('analysis/mlp_act_silu.pt'))
model.eval()

# Extrapolate a track
# Input: [x, y, tx, ty, q/p, z] at z=4000mm
input_state = torch.tensor([[100.0, 50.0, 0.1, 0.05, 0.02, 4000.0]])

with torch.no_grad():
    output_state = model(input_state)
    # Output: [x', y', tx', ty'] at z=12000mm
    print(f"Extrapolated: x={output_state[0,0]:.1f}, y={output_state[0,1]:.1f}")
```

### C++ (with ONNX)

See `../experiments/onnx_export/` for exported ONNX models.

---

## 📈 Performance Summary

**Best Model:** `mlp_act_silu.pt`
- Mean error: 0.21 mm
- 95th percentile: 0.54 mm  
- Max error: 4.13 mm
- Parameters: 25,924
- Inference: ~6 μs per track

**Comparison to Traditional:**
- ~30,000× faster than RK8
- ~4,700× faster than RK4
- Accuracy: <1mm error (target achieved ✅)

---

**Last Updated:** January 2025
