# Neural Network Models for Track Extrapolation

This directory contains the complete training infrastructure for ML-based track extrapolation in LHCb.

## ⚠️ Version Notes

| Version | Architecture Changes | Data |
|---------|---------------------|------|
| **V1** | Standard MLP/PINN/RK-PINN | Fixed dz=8000mm |
| **V2** | PINN residual: `Output = IC + z_frac × NetworkCorrection` | Fixed dz=8000mm |
| **V3** | Same as V2 | **Variable dz ∈ [500, 12000] mm** |

**Input Format (all versions):**
```
Input:  [x, y, tx, ty, q/p, dz] -> 6 features
Output: [x_out, y_out, tx_out, ty_out] -> 4 features
```

**Key Issue (V1/V2):** Models trained with fixed dz=8000mm have `input_std[dz] ≈ 0`, causing normalization explosion for other step sizes. V3 fixes this.

## Quick Start

```bash
# Train MLP baseline
python train.py --model mlp --preset medium --epochs 100

# Train PINN with physics loss
python train.py --model pinn --preset medium --lambda-pde 1.0 --epochs 100

# Train RK-PINN (multi-stage)
python train.py --model rk_pinn --preset medium --epochs 100

# Evaluate trained model
python evaluate.py --model-path checkpoints/mlp_medium_20260115/

# Compare all models
python evaluate.py --compare checkpoints/mlp_* checkpoints/pinn_*

# Run predefined experiment suite
python run_experiments.py --experiment architecture_comparison
```

---

## Table of Contents

1. [Physics Background](#physics-background)
2. [Model Architectures](#model-architectures)
3. [Training Pipeline](#training-pipeline)
4. [Evaluation](#evaluation)
5. [File Reference](#file-reference)
6. [Architecture Presets](#architecture-presets)

---

## Physics Background

### Track State Representation

A particle track in LHCb is parameterized by a **5-component state vector** at reference plane z:

| Component | Symbol | Description | Units |
|-----------|--------|-------------|-------|
| x | x | Horizontal position | mm |
| y | y | Vertical position | mm |
| tx | dx/dz | Horizontal slope | - |
| ty | dy/dz | Vertical slope | - |
| q/p | q/p | Charge over momentum | 1/MeV |

### The Extrapolation Problem

**Given:** Initial state `[x₀, y₀, tx₀, ty₀, q/p]` at `z_start`  
**Predict:** Final state `[x_f, y_f, tx_f, ty_f]` at `z_end = z_start + dz`

### Lorentz Force Equations

Charged particles follow the Lorentz force in the magnetic field:

```
dx/dz  = tx
dy/dz  = ty
dtx/dz = κ · √(1 + tx² + ty²) · [tx·ty·Bx - (1 + tx²)·By + ty·Bz]
dty/dz = κ · √(1 + tx² + ty²) · [(1 + ty²)·Bx - tx·ty·By - tx·Bz]
```

Where `κ = (q/p) × c_light` with `c_light = 2.99792458×10⁻⁴`.

---

## Model Architectures

### Overview

| Model | Physics | Inference | Training | Use Case |
|-------|---------|-----------|----------|----------|
| **MLP** | Implicit (data) | Fastest | Simple | Production baseline |
| **PINN** | Explicit (PDE loss) | Fast | Complex | Physics-constrained |
| **RK_PINN** | Explicit (staged) | Fast | Complex | Structured physics |

---

### 1. MLP (Multi-Layer Perceptron)

Standard feedforward network trained with data loss only.

```
┌─────────────────────────────────────────────────────────────┐
│                        MLP Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT [6]: [x₀, y₀, tx₀, ty₀, q/p, dz]                     │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │  Input Normalization   │  z-score: (x - μ) / σ    │
│         └────────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │  Linear → SiLU         │  ×N hidden layers        │
│         └────────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │  Output Linear         │                          │
│         └────────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │ Output Denormalization │  y × σ + μ               │
│         └────────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│  OUTPUT [4]: [x_f, y_f, tx_f, ty_f]                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Training Loss:**
```
L = MSE(prediction, ground_truth)
```

**Usage:**
```python
from architectures import MLP
model = MLP(hidden_dims=[256, 256, 128], activation='silu')
y_pred = model(x)  # x: [batch, 6] → y: [batch, 4]
```

---

### 2. PINN (Physics-Informed Neural Network)

Network trained with Lorentz force PDE residual via autodiff.

```
┌─────────────────────────────────────────────────────────────┐
│                       PINN Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT [5]: [x₀, y₀, tx₀, ty₀, q/p]                         │
│           + z_normalized ∈ [0, 1]                           │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │  Shared Network        │                          │
│         │  Linear → Tanh × N     │                          │
│         └────────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │  Trajectory y(z)       │  Continuous function     │
│         └────────────────────────┘                          │
│                      │                                       │
│           ┌─────────┴─────────┐                             │
│           ▼                   ▼                             │
│    ┌─────────────┐    ┌─────────────────────┐              │
│    │ y(z=1)      │    │ ∂y/∂z (autodiff)    │              │
│    │ = Endpoint  │    │ vs Lorentz Force    │              │
│    └─────────────┘    └─────────────────────┘              │
│           │                   │                             │
│           ▼                   ▼                             │
│      L_data              L_pde + L_ic                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Training Loss:**
```
L = L_data + λ_ic · L_ic + λ_pde · L_pde

L_data : MSE at endpoint vs ground truth
L_ic   : MSE at z=0 vs initial condition  
L_pde  : Σ ||∂y/∂z - F(y, B)||² at collocation points
```

**Usage:**
```python
from architectures import PINN
model = PINN(hidden_dims=[256, 256, 128], lambda_pde=1.0, n_collocation=10)
y_pred = model(x)
physics_loss = model.compute_physics_loss(x, y_pred)
```

---

### 3. RK_PINN (Runge-Kutta Physics-Informed Neural Network)

Multi-stage architecture inspired by RK4 numerical integrator.

```
┌─────────────────────────────────────────────────────────────┐
│                     RK_PINN Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT [6]: [x₀, y₀, tx₀, ty₀, q/p, dz]                     │
│                      │                                       │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │   Shared Backbone      │  Feature extraction      │
│         │   Linear → Tanh × N    │                          │
│         └────────────────────────┘                          │
│                      │                                       │
│    ┌─────────┬───────┴───────┬─────────┐                    │
│    ▼         ▼               ▼         ▼                    │
│ ┌──────┐ ┌──────┐       ┌──────┐ ┌──────┐                  │
│ │Head 1│ │Head 2│       │Head 3│ │Head 4│                  │
│ │z=0.25│ │z=0.50│       │z=0.75│ │z=1.00│                  │
│ └──────┘ └──────┘       └──────┘ └──────┘                  │
│    │         │               │         │                    │
│    ▼         ▼               ▼         ▼                    │
│   k₁        k₂              k₃        k₄                    │
│    │         │               │         │                    │
│    └─────────┴───────┬───────┴─────────┘                    │
│                      ▼                                       │
│         ┌────────────────────────┐                          │
│         │  Weighted Combination  │  w₁k₁ + w₂k₂ + w₃k₃ + w₄k₄│
│         │  (learnable weights)   │  init: [1,2,2,1]/6 (RK4) │
│         └────────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│  OUTPUT [4]: [x_f, y_f, tx_f, ty_f]                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Training Loss:**
```
L = L_data + λ_ic · L_ic + λ_pde · Σᵢ L_pde(stage_i)
```

**Usage:**
```python
from architectures import RK_PINN
model = RK_PINN(hidden_dims=[256, 256, 128], n_stages=4)
y_pred = model(x)
stage_preds = model.get_stage_predictions(x)  # List of 4 predictions
```

---

## Training Pipeline

### Basic Training

```bash
# MLP with default settings
python train.py --model mlp --preset medium

# PINN with physics loss tuning
python train.py --model pinn --preset medium --lambda-pde 0.1 --lambda-ic 1.0

# Custom architecture
python train.py --model mlp --hidden_dims 512 512 256 --activation gelu
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | - | Model type: mlp, pinn, rk_pinn |
| `--preset` | medium | Architecture preset |
| `--epochs` | 100 | Number of epochs |
| `--batch_size` | 2048 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 20 | Early stopping patience |
| `--lambda_pde` | 1.0 | PDE loss weight (PINN/RK_PINN) |
| `--lambda_ic` | 1.0 | IC loss weight (PINN/RK_PINN) |

### Running Experiments

```bash
# List available experiments
python run_experiments.py --list-experiments

# Architecture comparison (MLP vs PINN vs RK_PINN)
python run_experiments.py --experiment architecture_comparison

# Quick test (debugging)
python run_experiments.py --experiment quick_test

# Production training
python run_experiments.py --experiment production_training

# Train all models with same config
python run_experiments.py --all-models --preset large --epochs 200
```

### Output Structure

Training creates a directory under `checkpoints/`:

```
checkpoints/
└── mlp_medium_20260119_143052/
    ├── config.json           # Training configuration
    ├── model_config.json     # Model architecture
    ├── normalization.json    # Input/output normalization
    ├── best_model.pt         # Best checkpoint
    ├── checkpoint_epoch_50.pt
    ├── history.json          # Training curves
    └── plots/                # Training plots (if enabled)
```

---

## Evaluation

### Basic Evaluation

```bash
# Evaluate single model
python evaluate.py --model-path checkpoints/mlp_medium_20260119/

# With detailed plots
python evaluate.py --model-path checkpoints/pinn_medium/ --plots

# Momentum-binned analysis
python evaluate.py --model-path checkpoints/rk_pinn/ --momentum-bins

# Export for CI/CD
python evaluate.py --model-path checkpoints/best/ --export-json results.json
```

### Comparing Models

```bash
# Compare multiple models
python evaluate.py --compare \
    checkpoints/mlp_medium \
    checkpoints/pinn_medium \
    checkpoints/rk_pinn_medium
```

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `pos_mean_mm` | < 0.5 | Mean position error |
| `pos_95_mm` | < 2.0 | 95th percentile position error |
| `slope_mean` | < 1e-4 | Mean slope error |

---

## File Reference

| File | Description |
|------|-------------|
| `architectures.py` | Model definitions (MLP, PINN, RK_PINN) |
| `train.py` | Training script with full pipeline |
| `evaluate.py` | Comprehensive model evaluation |
| `run_experiments.py` | Experiment orchestration |
| `export_onnx.py` | ONNX export for deployment |

### architectures.py

Contains:
- `C_LIGHT = 2.99792458e-4` - Speed of light constant
- `MagneticField` - Gaussian field approximation
- `BaseTrackExtrapolator` - Base class with normalization
- `MLP` - Multi-layer perceptron
- `PINN` - Physics-informed neural network
- `RK_PINN` - Runge-Kutta physics-informed network
- `create_model()` - Factory function
- `ARCHITECTURE_PRESETS` - Predefined configurations

### train.py

Features:
- Data loading from NPZ format
- Train/val/test splitting
- AdamW optimizer with cosine scheduler + warmup
- Early stopping with patience
- Checkpoint management
- Comprehensive logging

### evaluate.py

Features:
- Detailed metrics (mean, std, percentiles)
- Momentum-binned analysis
- Publication-quality plots
- Model comparison tables
- JSON export for CI/CD

---

## Architecture Presets

| Preset | Hidden Dims | ~Parameters | Use Case |
|--------|-------------|-------------|----------|
| `tiny` | [64, 64] | 5k | Quick debugging |
| `small` | [128, 128] | 20k | Fast baseline |
| `medium` | [256, 256, 128] | 100k | Balanced |
| `large` | [512, 512, 256] | 400k | High accuracy |
| `wide` | [512, 512, 256, 128] | 500k | Maximum accuracy |

**Recommendation:** Start with `medium` for development, use `large` for production.

---

## Example Workflow

```bash
# 1. Generate training data (see ../data_generation/)
cd ../data_generation
python generate_data.py --n-tracks 50000000 --n-workers 32 --output data/training_50M.npz

# 2. Train models
cd ../models
python run_experiments.py --experiment architecture_comparison

# 3. Evaluate and compare
python evaluate.py --compare checkpoints/architecture_comparison_*

# 4. Export best model for deployment
python export_onnx.py --model-path checkpoints/best_model/ --output production.onnx
```

---

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 1024 or 512)
- Use smaller preset (`--preset small`)
- Limit samples (`--max_samples 1000000`)

### Training Unstable
- Reduce learning rate (`--lr 1e-4`)
- For PINN: reduce physics weight (`--lambda_pde 0.1`)
- Check data normalization

### Poor Generalization
- Increase training data
- Add dropout (`--dropout 0.1`)
- Use physics constraints (PINN/RK_PINN)

---

**Author:** G. Scriven  
**Date:** January 2026  
**Project:** LHCb Track Extrapolation with Neural Networks
