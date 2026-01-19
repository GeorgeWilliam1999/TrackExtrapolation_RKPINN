# ONNX Export Experiment

## Objective

Export trained PyTorch models to ONNX format for production deployment in the LHCb C++ framework.

## Why ONNX?

- **Portable**: Run models without PyTorch dependency
- **Fast**: ONNX Runtime is highly optimized for inference
- **Production-ready**: Can be integrated with C++ code
- **Hardware agnostic**: CPU, GPU, specialized accelerators

## Exported Models (2024-12-21)

| Model | ONNX File | Normalization | Status |
|-------|-----------|---------------|--------|
| MLP Full Domain | `mlp_full_domain.onnx` | `mlp_full_domain_norm.json` | ✅ Verified |
| PINN Full Domain | `pinn_full_domain.onnx` | `pinn_full_domain_norm.json` | ✅ Verified |

## Performance Results

| Method | Mean Error | 95% Error | Time/Track | Speedup vs RK8 |
|--------|------------|-----------|------------|----------------|
| RK8 (5mm) | 0.00 mm | 0.00 mm | 174,622 μs | 1.0x |
| MLP ONNX | 1.48 mm | 2.41 mm | ~6 μs | ~30,000x |
| PINN ONNX | 1.28 mm | 2.73 mm | ~6 μs | ~30,000x |

## Usage

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('mlp_full_domain.onnx')

# Input: [x, y, tx, ty, qop, dz]
input_data = np.array([[0, 0, 0.1, 0.05, 0.0002, 4000]], dtype=np.float32)

# Inference
output = session.run(None, {'input': input_data})[0]
# Output: [x_out, y_out, tx_out, ty_out]
```

### C++ (ONNX Runtime)

```cpp
#include <onnxruntime_cxx_api.h>

// Load model
Ort::Session session(env, "mlp_full_domain.onnx", session_options);

// Prepare input
std::vector<float> input = {x, y, tx, ty, qop, dz};
std::vector<int64_t> input_shape = {1, 6};

// Run inference
auto output = session.Run(...);
```

## Validation

The export script validates that:
1. ONNX outputs match PyTorch outputs (atol < 1e-5)
2. All operations are supported in the target opset
3. Dynamic batch sizes work correctly

## Files

- `export_onnx.py` - Export script
- `mlp_full_domain.onnx` - Exported MLP model
- `pinn_full_domain.onnx` - Exported PINN model
- `onnx_export_results.json` - Validation and benchmark results

## Requirements

```bash
pip install onnx onnxruntime
```

## Notes

- ONNX opset version 14 used for compatibility
- Normalization is baked into the ONNX graph (no separate preprocessing needed)
- Input/output shapes use dynamic batch dimension
