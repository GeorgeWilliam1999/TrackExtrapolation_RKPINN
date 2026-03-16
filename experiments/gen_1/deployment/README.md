# Model Deployment to C++

This directory contains scripts for exporting trained PyTorch models to binary format
for use in the LHCb C++ framework (TrackMLPExtrapolator).

## Export Script

```bash
python export_to_cpp.py \
    --model-dir ../trained_models/mlp_v2_shallow_512 \
    --output-dir ../../ml_models/models/mlp_v2_shallow_512
```

## Binary Format

The exported `.bin` file contains:

```
Header:
  - n_layers (int32): Number of linear layers
  - layer_sizes (int32[]): Input/output sizes for each layer
  - activation (int32): 0=relu, 1=tanh, 2=silu

Normalization:
  - input_mean (float32[6]): Mean for each input feature
  - input_std (float32[6]): Std for each input feature
  - output_mean (float32[4]): Mean for each output feature
  - output_std (float32[4]): Std for each output feature

Weights (per layer):
  - weight (float32[out_size × in_size]): Row-major weight matrix
  - bias (float32[out_size]): Bias vector
```

## Input/Output Specification

**Inputs** (6 features):
| Index | Feature | Units | Typical Range |
|-------|---------|-------|---------------|
| 0 | x | mm | [-3000, 3000] |
| 1 | y | mm | [-3000, 3000] |
| 2 | tx | - | [-0.3, 0.3] |
| 3 | ty | - | [-0.3, 0.3] |
| 4 | q/p | c/GeV | [-1, 1] |
| 5 | dz | mm | [500, 12000] |

**Outputs** (4 features):
| Index | Feature | Units |
|-------|---------|-------|
| 0 | x_out | mm |
| 1 | y_out | mm |
| 2 | tx_out | - |
| 3 | ty_out | - |

## ⚠️ V1/V2 Normalization Issue

**Problem**: V1/V2 models trained with fixed dz=8000mm have:
```
input_std[5] ≈ 1e-9  (dz)
```

This causes division by zero during normalization for dz ≠ 8000.

**Workaround**: The export script patches small std values:
```python
# If std < 1e-6, use mean instead of normalization
# (x - mean) / std -> 0 when std is tiny
for i in range(6):
    if input_std[i] < 1e-6:
        input_std[i] = 1.0  # Effectively pass-through
```

**Solution**: Use V3 models trained with variable dz, which have proper `input_std[5] ≈ 3300`.

## C++ Integration

The binary is loaded by `TrackMLPExtrapolator` in `src/TrackMLPExtrapolator.cpp`:

```cpp
// Load model
std::ifstream file(model_path, std::ios::binary);
// Read header, normalization, weights...

// Forward pass
void MLPModel::forward(const float* input, float* output) {
    // Normalize inputs
    for (int i = 0; i < 6; i++) {
        normalized[i] = (input[i] - input_mean[i]) / input_std[i];
    }
    
    // Hidden layers
    for (int layer = 0; layer < n_layers - 1; layer++) {
        linear(input, output);
        activation(output);  // silu, tanh, or relu
    }
    
    // Output layer
    linear(input, output);
    
    // Denormalize outputs
    for (int i = 0; i < 4; i++) {
        output[i] = output[i] * output_std[i] + output_mean[i];
    }
}
```

## Model Registry

Available models for deployment:

| Model | Version | Parameters | Accuracy | dz Support |
|-------|---------|------------|----------|------------|
| `mlp_v2_shallow_512` | V2 | 135K | 0.03mm | Fixed 8000mm only |
| `mlp_v2_single_256` | V2 | 2.8K | 0.07mm | Fixed 8000mm only |
| `mlp_v3_*` | V3 | Various | TBD | ✅ Variable |

## See Also
- [V3/README.md](../V3/README.md) - V3 models with variable dz support
- [ml_models/README.md](../../ml_models/README.md) - C++ model infrastructure
- [src/TrackMLPExtrapolator.cpp](../../src/TrackMLPExtrapolator.cpp) - C++ implementation
