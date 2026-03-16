# ML Models — Deployed Neural Network Extrapolators

This directory contains the binary model files and C++ inference code for the `TrackMLPExtrapolator`, a neural network track extrapolator integrated into the LHCb framework.

---

## Directory Structure

```
ml_models/
├── README.md                      # This file
├── models/                        # Binary model files
│   ├── mlp_v2_shallow_256.bin     # MLP: [256] hidden, SiLU
│   ├── pinn_v3_res_256_col5.bin   # PINN v3 residual: 5 collocation points
│   ├── pinn_v3_res_256_col10.bin  # PINN v3 residual: 10 collocation points
│   └── pinn_v3_res_256_col20.bin  # PINN v3 residual: 20 collocation points
└── src/
    └── TrackMLPExtrapolator.cpp   # Eigen-based NN inference (571 lines)
```

---

## Deployed Models

| Model File | Architecture | Hidden Layers | Activation | Mode | Origin |
|------------|-------------|---------------|------------|------|--------|
| `mlp_v2_shallow_256.bin` | MLP | [256] | SiLU | Endpoint | V2 shallow-wide sweep |
| `pinn_v3_res_256_col5.bin` | PINN v3 Residual | [256, 256] | SiLU | IC + z_frac scaling | V3 PINN with skip connection |
| `pinn_v3_res_256_col10.bin` | PINN v3 Residual | [256, 256] | SiLU | IC + z_frac scaling | V3 PINN with skip connection |
| `pinn_v3_res_256_col20.bin` | PINN v3 Residual | [256, 256] | SiLU | IC + z_frac scaling | V3 PINN with skip connection |

---

## TrackMLPExtrapolator

The C++ inference engine (`src/TrackMLPExtrapolator.cpp`, 571 lines) provides:

- **Binary model loading** — reads layer weights, biases, normalization parameters, and architecture metadata from `.bin` files
- **Eigen-based forward pass** — matrix multiply + activation, no external ML framework dependency
- **Activation functions** — ReLU, Tanh, SiLU, Sigmoid
- **PINN residual mode** — optional skip connection: `output = initial_state + z_frac × network_correction`
- **Input/output normalization** — mean/std normalization baked into the model file

### Usage

The extrapolator is configured as a standard Gaudi tool and registered in `CMakeLists.txt`. It reads the model path from its properties:

```cpp
// Properties
declareProperty("ModelFile", m_modelFile = "mlp_v2_shallow_256.bin");
declareProperty("UsePINNResidual", m_usePINNResidual = false);
```

---

## Exporting New Models

To export a trained PyTorch model to the binary format:

```bash
cd experiments/gen_1
python deployment/export_to_cpp.py \
    trained_models/<model_name> \
    ../../ml_models/models/<model_name>.bin
```

See [experiments/gen_1/deployment/README.md](../experiments/gen_1/deployment/README.md) for the binary format specification and normalization handling.

---

*Last Updated: March 2026*
