# ML Models — Deployed Neural Network Extrapolators

This directory contains the binary model files and C++ inference code for the `TrackMLPExtrapolator`, a neural network track extrapolator integrated into the LHCb framework.

---

## Directory Structure

```
ml_models/
├── README.md                      # This file
├── models/                        # Binary model files (currently empty — to be regenerated)
└── src/
    └── TrackMLPExtrapolator.cpp   # Eigen-based NN inference (571 lines)
```

---

## Deployed Models

No models currently deployed. Models will be exported from V1 trained checkpoints using the deployment pipeline.

To export a trained model:

```bash
cd experiments/gen_1
python deployment/export_to_cpp.py \
    V1/trained_models/<model_name> \
    ../../ml_models/models/<model_name>.bin
```

See [experiments/gen_1/deployment/README.md](../experiments/gen_1/deployment/README.md) for the binary format specification.

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

*Last Updated: April 2026*
