# Track Extrapolation ML Experiments

This directory contains all machine learning experiments for neural network track extrapolation in LHCb.

---

## Directory Structure

```
experiments/
├── README.md                # This file
├── gen_1/                   # Active ML extrapolator experiments (V1–V5)
│   ├── README.md            # Master experiment overview
│   ├── MASTER_RESULTS.md    # Comprehensive results document
│   ├── MASTER_RESULTS.tex   # LaTeX version
│   ├── LHCB_STACK_MANUAL.md # LHCb stack setup guide
│   ├── DEPENDENCY_GRAPH.md  # Python module dependency map
│   ├── V1/                  # 53 models — architecture sweep (deprecated)
│   ├── V2/                  # 22 shallow-wide models (deprecated)
│   ├── V3/                  # Variable dz training (active)
│   ├── V4/                  # PINN diagnosis + width sweep (complete)
│   ├── V5/                  # PINN architecture fixes (active)
│   ├── deployment/          # C++ model export pipeline
│   ├── trained_models/      # All trained model checkpoints
│   └── archive/             # Historical analysis and early papers
│
├── gen_2/                   # Reference papers for future architectures
│   └── (4 PDFs — Neural ODE, ASR-PINN, RK-PINN, long-time integration)
│
└── field_maps/              # Magnetic field map experiments
    ├── nn_field_map_sizing.ipynb  # Field map NN sizing analysis
    ├── twodip.rtf                 # LHCb dipole field data
    └── field_nn/                  # NN field map training pipeline
```

---

## Experiment Generations

### gen_1 — Neural Network Track Extrapolators (Active)

Systematic comparison of neural network architectures for replacing C++ RK4 track propagation. 75+ models trained across 5 experiment versions.

**Best result:** `mlp_v2_shallow_512_256` — **0.028 mm** position error, **1.3× faster** than C++ RK4 (fixed dz = 8000 mm).

| Version | Models | Key Focus | Status |
|---------|--------|-----------|--------|
| V1 | 53 | MLP/PINN/RK_PINN architecture sweep | Deprecated |
| V2 | 22 | Shallow-wide speed optimization | Deprecated |
| V3 | — | Variable dz (500–12000 mm) | Active |
| V4 | — | PINN root-cause diagnosis + width sweep | Complete |
| V5 | 7 | PINN architecture fixes (Quadratic, ZFrac, PDE, Compositional) | Active |

See [gen_1/README.md](gen_1/README.md) for full details.

### gen_2 — Future Directions (Reference)

Collection of reference papers for potential next-generation approaches:
- Neural Ordinary Differential Equations
- Adaptive Step-size Runge-Kutta PINNs (ASR-PINN)
- Long-time integration of ODEs with PINNs
- RK-PINN parameter estimation for nonlinear systems

See [gen_2/README.md](gen_2/README.md).

### field_maps — Magnetic Field Approximation

Neural network approximation of the LHCb dipole magnetic field map. Separate from the track extrapolation task — this explores whether the field itself can be represented by a small NN for faster evaluation.

See [field_maps/README.md](field_maps/README.md) and [field_maps/field_nn/README.md](field_maps/field_nn/README.md).

---

## Key Results

| Generation | Best Model | Position Error | Speedup vs C++ | Dataset |
|-----------|-----------|---------------|----------------|---------|
| Legacy (pre-2025) | MLP SiLU 128-128-64 | 0.21 mm | ~30,000× vs Python RK8 | 50K tracks |
| gen_1 V1 | mlp_tiny_v1 | 0.024 mm | 2.3× vs C++ RK4 | 50M tracks, fixed dz |
| gen_1 V2 | mlp_v2_shallow_512_256 | **0.028 mm** | **1.3× vs C++ RK4** | 50M tracks, fixed dz |
| gen_1 V3 | mlp_shallow_256 | ~1.0 mm | 0.7× vs C++ RK4 | 100M tracks, variable dz |

**Note:** Legacy "30,000× speedup" was relative to a Python RK8 implementation, not C++ RK4. The meaningful comparison is against the C++ RK4 baseline (2.50 μs/track for fixed dz).

---

## Quick Start

```bash
# Navigate to active experiments
cd gen_1/

# See available versions and status
cat README.md

# Train a V5 model
python V5/training/train_v5.py --config V5/training/configs/mlp_v5_2L_1024_512.json

# View comprehensive results
cat MASTER_RESULTS.md
```

---

*Last Updated: March 2026*
