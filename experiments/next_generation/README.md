# Neural Network Track Extrapolators - Next Generation

**Project Goal:** Systematically train and compare neural network architectures for LHCb track extrapolation, storing all training metrics for comprehensive analysis.

**Status:** ✅ Training in progress (30 HTCondor GPU jobs, cluster 3880818, 10 epochs each)

---

## Table of Contents

1. [Overview](#overview)
2. [Experiment Design](#experiment-design)
3. [Model Architectures](#model-architectures)
4. [Training Configurations](#training-configurations)
5. [Running Experiments](#running-experiments)
6. [Loss Tracking & Convergence Analysis](#loss-tracking--convergence-analysis)
7. [Performance Benchmarking](#performance-benchmarking)
8. [Directory Structure](#directory-structure)
9. [Quick Reference](#quick-reference)

---

## Overview

We implement a systematic comparison of **three model architectures** with **varying complexity** and **physics constraints**:

| Architecture | Physics | Training Type | Key Characteristic |
|--------------|---------|---------------|-------------------|
| **MLP** | Implicit (from data) | Data loss only | Fast, simple baseline |
| **PINN** | Explicit (autodiff) | Data + PDE residual | Physics-constrained |
| **RK_PINN** | Explicit (multi-stage) | Data + staged PDE | RK4-inspired structure |

For each architecture, we test **four size presets**:

| Preset | Hidden Layers | Parameters | Use Case |
|--------|---------------|------------|----------|
| `tiny` | [64, 64] | ~5k | Debugging, quick tests |
| `small` | [128, 128] | ~20k | Fast training, baseline |
| `medium` | [256, 256, 128] | ~100k | Balanced performance |
| `wide` | [512, 512, 256, 128] | ~500k | Maximum accuracy |

This gives us **12 base experiments** (3 architectures × 4 sizes), plus physics loss ablations.

---

## Experiment Design

### Core Experiments Matrix

We will train and compare all combinations:

| Model | tiny | small | medium | wide |
|-------|------|-------|--------|------|
| **MLP** | mlp_tiny | mlp_small | mlp_medium | mlp_wide |
| **PINN** | pinn_tiny | pinn_small | pinn_medium | pinn_wide |
| **RK_PINN** | rkpinn_tiny | rkpinn_small | rkpinn_medium | rkpinn_wide |

### Physics Loss Ablations (PINN/RK_PINN)

For PINN and RK_PINN models, we study the effect of physics loss weights:

| Experiment | λ_data | λ_pde | λ_ic | Purpose |
|------------|--------|-------|------|---------|
| `data_only` | 1.0 | 0.0 | 0.0 | Baseline (equivalent to MLP) |
| `pde_weak` | 1.0 | 0.01 | 0.01 | Weak physics regularization |
| `pde_balanced` | 1.0 | 1.0 | 1.0 | Equal weighting (default) |
| `pde_strong` | 1.0 | 10.0 | 10.0 | Strong physics enforcement |
| `pde_dominant` | 0.1 | 1.0 | 1.0 | Physics-dominated training |

### Momentum Range Studies

Physics behavior varies with momentum (low-p tracks bend more):

| Study | Momentum Range | Expected Behavior |
|-------|----------------|-------------------|
| `low_p` | 0.5 - 5 GeV | High curvature, physics crucial |
| `mid_p` | 5 - 20 GeV | Moderate curvature |
| `high_p` | 20 - 100 GeV | Low curvature, nearly linear |
| `full_range` | 0.5 - 100 GeV | Full domain (production) |

---

## Model Architectures

### 1. MLP (Multi-Layer Perceptron)

**Philosophy:** Learn the input→output mapping directly from data.

```
┌─────────────────────────────────────────────────────────────────┐
│                         MLP Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input [6]              Hidden Layers              Output [4]  │
│  ┌─────────┐           ┌─────────────┐            ┌─────────┐  │
│  │ x₀      │           │             │            │ x_f     │  │
│  │ y₀      │    ───►   │  FC + ReLU  │   ───►     │ y_f     │  │
│  │ tx₀     │           │  FC + ReLU  │            │ tx_f    │  │
│  │ ty₀     │           │  ...        │            │ ty_f    │  │
│  │ q/p     │           │             │            │         │  │
│  │ dz      │           └─────────────┘            └─────────┘  │
│  └─────────┘                                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Loss:** `data_loss = MSE(prediction, ground_truth)`

**Advantages:** Fastest inference, simplest to train  
**Disadvantages:** No physics constraints, may extrapolate poorly

---

### 2. PINN (Physics-Informed Neural Network)

**Philosophy:** Constrain learning with Lorentz force equations via autodiff.

```
┌─────────────────────────────────────────────────────────────────┐
│                        PINN Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input [6]              Hidden Layers              Output [4]  │
│  ┌─────────┐           ┌─────────────┐            ┌─────────┐  │
│  │ x₀      │           │             │            │ x(z)    │  │
│  │ y₀      │    ───►   │  FC + ReLU  │   ───►     │ y(z)    │  │
│  │ tx₀     │           │  FC + ReLU  │            │ tx(z)   │  │
│  │ ty₀     │           │  ...        │            │ ty(z)   │  │
│  │ q/p     │           │             │            │         │  │
│  │ z_norm  │◄──────────└─────────────┘            └────┬────┘  │
│  └─────────┘                                           │       │
│       ▲                                                │       │
│       │              ┌─────────────────┐               │       │
│       │              │  PHYSICS LOSS   │◄──────────────┘       │
│       │              │  ∂y/∂z = F(y,B) │   (autodiff)          │
│       │              └─────────────────┘                       │
│       │                      │                                 │
│       └──────────────────────┘                                 │
│            (collocation points: z ∈ [0,1])                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Losses:**
- `data_loss = MSE(y(z=1), ground_truth)` — endpoint match
- `ic_loss = MSE(y(z=0), initial_state)` — initial condition
- `pde_loss = Σ ||dy/dz - Lorentz(y, B)||²` — PDE residual

**Advantages:** Physics-constrained, better generalization  
**Disadvantages:** Slower training (autodiff overhead), requires field model

---

### 3. RK_PINN (Runge-Kutta PINN)

**Philosophy:** Multi-stage architecture inspired by RK4 numerical integrator.

```
┌─────────────────────────────────────────────────────────────────┐
│                      RK_PINN Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input [6]           Backbone           Stage Heads            │
│  ┌─────────┐        ┌─────────┐        ┌─────────────┐         │
│  │ x₀      │        │         │   ┌───►│ Head 1      │──► k₁   │
│  │ y₀      │  ───►  │ Shared  │   │    │ (z = 0.25)  │         │
│  │ tx₀     │        │ FC      │   │    └─────────────┘         │
│  │ ty₀     │        │ Layers  │───┼───►┌─────────────┐         │
│  │ q/p     │        │         │   │    │ Head 2      │──► k₂   │
│  │ dz      │        │         │   │    │ (z = 0.5)   │         │
│  └─────────┘        └─────────┘   │    └─────────────┘         │
│                                   ├───►┌─────────────┐         │
│                                   │    │ Head 3      │──► k₃   │
│                                   │    │ (z = 0.75)  │         │
│                                   │    └─────────────┘         │
│                                   └───►┌─────────────┐         │
│                                        │ Head 4      │──► k₄   │
│                                        │ (z = 1.0)   │         │
│                                        └─────────────┘         │
│                                                                 │
│   Output = w₁·k₁ + w₂·k₂ + w₃·k₃ + w₄·k₄                       │
│                                                                 │
│   Weights: learnable, initialized to [1,2,2,1]/6 (RK4)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Losses:**
- `data_loss = MSE(weighted_sum, ground_truth)`
- `stage_pde_loss = Σᵢ ||∂yᵢ/∂z - Lorentz(yᵢ, B(zᵢ))||²` (at each stage)

**Advantages:** Interpretable stages, natural multi-scale learning  
**Disadvantages:** More complex architecture, 4× head computations

---

## Training Configurations

### Default Hyperparameters

```yaml
# Data
data_path: data_generation/data/full_dataset_50M.npz
train_fraction: 0.8    # 40M training samples
val_fraction: 0.1      # 5M validation samples
test_fraction: 0.1     # 5M test samples

# Training
batch_size: 2048       # Optimized for GPU memory
epochs: 100            # Maximum epochs
learning_rate: 1e-3    # Initial LR
weight_decay: 1e-4     # L2 regularization
scheduler: cosine      # LR schedule with warmup

# Early Stopping
patience: 20           # Stop if no improvement for 20 epochs
min_delta: 1e-6        # Minimum improvement threshold

# Physics (PINN/RK_PINN)
lambda_pde: 1.0        # PDE residual weight
lambda_ic: 1.0         # Initial condition weight
n_collocation: 10      # Collocation points (PINN only)
```

### Experiment-Specific Configs

Each experiment will have a YAML config file in `configs/`:

```yaml
# configs/mlp_medium.yaml
model_type: mlp
preset: medium
experiment_name: mlp_medium_v1

# configs/pinn_medium_strong_physics.yaml
model_type: pinn
preset: medium
lambda_pde: 10.0
lambda_ic: 10.0
experiment_name: pinn_medium_strong_v1
```

---

## Running Experiments

### Step 1: Generate Training Data (✅ Complete)

Training data has been generated and merged:
- `data_generation/data/training_50M.npz` - 50M tracks (3.7GB)
- `data_generation/data/training_low_p.npz` - 10M tracks (p < 5 GeV)
- `data_generation/data/training_mid_p.npz` - 10M tracks (5 ≤ p < 20 GeV)
- `data_generation/data/training_high_p.npz` - 10M tracks (p ≥ 20 GeV)

```bash
# Verify data:
python -c "import numpy as np; d=np.load('data_generation/data/training_50M.npz'); print(f'Loaded {d[\"X\"].shape[0]:,} tracks')"
```

### Step 2: Run Single Experiment

```bash
cd models

# Train MLP (data loss only)
python train.py --model mlp --preset medium --epochs 100 --name mlp_medium_v1

# Train PINN (data + physics)
python train.py --model pinn --preset medium --lambda_pde 1.0 --epochs 100 --name pinn_medium_v1

# Train RK_PINN (data + staged physics)
python train.py --model rk_pinn --preset medium --lambda_pde 1.0 --epochs 100 --name rkpinn_medium_v1
```

### Step 3: Run Full Experiment Suite

```bash
# Run all experiments via unified runner
python run_all_experiments.py --list        # List available experiments
python run_all_experiments.py --all         # Submit all to HTCondor
python run_all_experiments.py --local       # Run locally (interactive)

# Or submit individual jobs
cd training/jobs
condor_submit mlp_medium.sub
```

### Step 4: Evaluate and Compare

```bash
cd models

# Evaluate single model
python evaluate.py --model_path ../trained_models/mlp_medium/best_model.pt

# Run comprehensive analysis
cd ../analysis
jupyter notebook experiment_analysis.ipynb
```

---

## Loss Tracking & Convergence Analysis

### What We Store

Every training run saves comprehensive loss information:

```
checkpoints/<experiment_name>/
├── config.json              # Full training configuration
├── best_model.pt            # Model with best validation loss
├── normalization.json       # Input/output normalization parameters
├── history.json             # Complete training history (see below)
├── model_config.json        # Architecture configuration
└── checkpoint_epoch_N.pt    # Periodic checkpoints
```

### History Format (history.json)

```json
{
  "train": [
    {
      "epoch": 1,
      "loss": 0.0234,
      "data_loss": 0.0230,
      "physics_loss": 0.0004,
      "ic_loss": 0.0002,
      "pde_loss": 0.0002,
      "lr": 0.001
    }
  ],
  "val": [
    {
      "epoch": 1,
      "loss": 0.0198,
      "pos_mean_mm": 0.342,
      "pos_std_mm": 0.156,
      "pos_95_mm": 0.621,
      "slope_mean": 1.2e-5,
      "x_mean_mm": 0.241,
      "y_mean_mm": 0.243,
      "tx_mean": 8.5e-6,
      "ty_mean": 8.7e-6
    }
  ],
  "test_final": { },
  "best_epoch": 87,
  "best_val_loss": 0.00156,
  "training_time": 3247.5
}
```

### Stored Loss Components

| Loss Component | Models | Description |
|----------------|--------|-------------|
| `loss` | All | Total combined loss |
| `data_loss` | All | MSE(prediction, ground_truth) |
| `physics_loss` | PINN, RK_PINN | ic_loss + pde_loss |
| `ic_loss` | PINN, RK_PINN | Initial condition constraint |
| `pde_loss` | PINN, RK_PINN | Lorentz equation residual |
| `stage_losses` | RK_PINN | Per-stage PDE losses [4 values] |

### Convergence Analysis

After training, analyze convergence behavior using the analysis notebook or scripts:

```bash
cd analysis

# Main analysis notebook (recommended)
jupyter notebook experiment_analysis.ipynb

# Or use command-line tools:
python analyze_models.py --checkpoint_dir ../trained_models
python timing_benchmark.py --models_dir ../trained_models
```

#### Key Convergence Questions

1. **Does physics loss help?** Compare final accuracy: MLP vs PINN vs RK_PINN
2. **How fast do losses converge?** Plot epochs to reach 90% of final accuracy
3. **Does physics loss converge?** Track ic_loss and pde_loss separately
4. **Model size vs convergence speed?** Compare tiny/small/medium/wide
5. **Overfitting detection?** Plot train vs validation loss divergence

---

## Performance Benchmarking

### Metrics We Compare

#### Accuracy Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `pos_mean_mm` | < 0.1 mm | Mean position error (√(Δx² + Δy²)) |
| `pos_95_mm` | < 0.3 mm | 95th percentile position error |
| `slope_mean` | < 1e-5 | Mean slope error (√(Δtx² + Δty²)) |
| `x_mean_mm` | < 0.1 mm | Mean x position error |
| `y_mean_mm` | < 0.1 mm | Mean y position error |

#### Speed Metrics

| Metric | Measurement Method |
|--------|-------------------|
| `training_time` | Total wall-clock time for training |
| `inference_time_cpu` | Time per batch on CPU |
| `inference_time_gpu` | Time per batch on GPU |
| `throughput_gpu` | Tracks/second on GPU |

### Benchmark Protocol

```bash
cd analysis

# Run timing benchmarks on all trained models
python timing_benchmark.py \
    --models_dir ../trained_models \
    --data_path ../data_generation/data/training_50M.npz \
    --output results/timing_results.json

# Generate paper-quality plots
python generate_paper_quality_plots.py
```

### Expected Results Table

After all experiments complete, we generate a summary table:

| Model | Params | pos_mean (mm) | pos_95% (mm) | GPU tput (tracks/s) | Train time (min) |
|-------|--------|---------------|--------------|---------------------|------------------|
| mlp_tiny | 5,252 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| mlp_small | 20,228 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| mlp_medium | 99,588 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| mlp_wide | 467,972 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| pinn_tiny | 5,256 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| pinn_small | 20,232 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| pinn_medium | 99,592 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| pinn_wide | 467,976 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| rkpinn_tiny | 6,820 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| rkpinn_small | 25,384 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| rkpinn_medium | 117,132 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| rkpinn_wide | 538,124 | 0.xxx | 0.xxx | xxx,xxx | xx.x |
| **C++ RK4** | N/A | 0.000 (ref) | 0.000 (ref) | xx,xxx (baseline) | N/A |

---

## Directory Structure

```
next_generation/
├── README.md                    # This file - experiment plan
├── run_all_experiments.py       # ⭐ Unified experiment runner script
│
├── data_generation/             # Training data generation
│   ├── README.md               # Field map and data format docs
│   ├── generate_data.py        # Main data generator
│   ├── generate_cpp_data.py    # C++ extrapolator data wrapper
│   ├── merge_batches.py        # Combine HTCondor batch outputs
│   ├── create_momentum_splits.py # Split by momentum range
│   └── data/                   # Generated datasets
│       ├── training_50M.npz    # Full dataset (50M tracks)
│       ├── training_low_p.npz  # Low momentum (0.5-5 GeV, 10M)
│       ├── training_mid_p.npz  # Mid momentum (5-20 GeV, 10M)
│       └── training_high_p.npz # High momentum (20-100 GeV, 10M)
│
├── models/                      # Model definitions and training
│   ├── README.md               # Detailed architecture documentation
│   ├── architectures.py        # MLP, PINN, RK_PINN classes
│   ├── train.py                # ⭐ Main training script
│   ├── evaluate.py             # Model evaluation
│   ├── export_onnx.py          # ONNX export for C++ deployment
│   ├── run_experiments.py      # Batch experiment runner
│   └── submit_training.py      # HTCondor job generator
│
├── training/                    # HTCondor training jobs
│   ├── README.md               # Job documentation
│   ├── train_wrapper.sh        # Worker node script
│   ├── jobs/                   # 29 .sub files for all experiments
│   └── logs/                   # Job output logs
│
├── trained_models/              # Output: trained model checkpoints
│   └── <experiment_name>/
│       ├── best_model.pt
│       ├── config.json
│       ├── history.json        # ← All losses stored here
│       └── normalization.json
│
├── analysis/                    # Results analysis and visualization
│   ├── README.md               # Analysis tools documentation
│   ├── experiment_analysis.ipynb # ⭐ Main analysis notebook
│   ├── model_analysis.ipynb    # Interactive analysis
│   ├── analyze_models.py       # Analysis functions
│   ├── physics_analysis.py     # Physics-specific tests
│   ├── timing_benchmark.py     # ⭐ Timing benchmark tool
│   ├── timing_comparison_plots.py # Timing visualizations
│   ├── generate_paper_quality_plots.py # Publication-ready figures
│   ├── trajectory_visualizer.py # Track visualization
│   ├── run_analysis.py         # Batch analysis runner
│   ├── results/                # Analysis outputs
│   └── plots/                  # Generated figures
│
├── benchmarking/                # C++ baseline benchmarks
│   ├── benchmark_cpp.py        # Run C++ extrapolators
│   └── parse_benchmark_results.py # Parse benchmark logs
│
├── utils/                       # Utility modules
│   ├── README.md               # Utils documentation
│   ├── magnetic_field.py       # ⭐ Unified field map (InterpolatedFieldTorch)
│   └── rk4_propagator.py       # Python RK4 integrator
│
├── notes/                       # Documentation
│   ├── experimental_protocol.tex  # Full experiment methodology
│   └── experimental_protocol.pdf
│
└── cluster/                     # HTCondor utilities
    ├── README.md
    └── monitor_training.sh     # Job monitoring script
```

---

## Quick Reference

### Training Commands

```bash
# MLP EXPERIMENTS (data loss only)
python train.py --model mlp --preset tiny --name mlp_tiny_v1
python train.py --model mlp --preset small --name mlp_small_v1
python train.py --model mlp --preset medium --name mlp_medium_v1
python train.py --model mlp --preset wide --name mlp_wide_v1

# PINN EXPERIMENTS (data + physics loss)
# Default physics weights (λ_pde=1.0, λ_ic=1.0)
python train.py --model pinn --preset tiny --name pinn_tiny_v1
python train.py --model pinn --preset small --name pinn_small_v1
python train.py --model pinn --preset medium --name pinn_medium_v1
python train.py --model pinn --preset wide --name pinn_wide_v1

# Strong physics enforcement
python train.py --model pinn --preset medium --lambda_pde 10.0 --lambda_ic 10.0 --name pinn_medium_strong_v1

# Weak physics regularization
python train.py --model pinn --preset medium --lambda_pde 0.01 --lambda_ic 0.01 --name pinn_medium_weak_v1

# RK_PINN EXPERIMENTS (multi-stage physics)
python train.py --model rk_pinn --preset tiny --name rkpinn_tiny_v1
python train.py --model rk_pinn --preset small --name rkpinn_small_v1
python train.py --model rk_pinn --preset medium --name rkpinn_medium_v1
python train.py --model rk_pinn --preset wide --name rkpinn_wide_v1
```

### Analysis Commands

```bash
# After training, analyze results:
cd analysis

# Main analysis notebook (recommended)
jupyter notebook experiment_analysis.ipynb

# Command-line analysis
python analyze_models.py --checkpoint_dir ../trained_models
python timing_benchmark.py --models_dir ../trained_models

# Generate paper-quality plots
python generate_paper_quality_plots.py
```

### Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Generation | ✅ Complete | 50M tracks in `training_50M.npz` |
| Momentum Splits | ✅ Complete | Low/Mid/High-p datasets (10M each) |
| Training Script | ✅ Ready | `models/train.py` functional |
| Architectures | ✅ Ready | MLP, PINN, RK_PINN (using InterpolatedFieldTorch) |
| Loss Tracking | ✅ Ready | All losses stored in history.json |
| Evaluation | ✅ Ready | `models/evaluate.py` functional |
| HTCondor Jobs | ✅ Submitted | 29 experiments (clusters 3880473-3880501) |
| Unified Runner | ✅ Ready | `run_all_experiments.py` |
| Analysis Notebook | ✅ Ready | `analysis/experiment_analysis.ipynb` |
| Experiment Protocol | ✅ Ready | `notes/experimental_protocol.pdf` |
| ONNX Export | ✅ Ready | `models/export_onnx.py` functional |

### Recent Updates (January 22, 2026)

1. **Fixed PINN/RK_PINN field model**: Now uses `InterpolatedFieldTorch` (real field map) instead of `GaussianFieldTorch`
2. **Created unified experiment runner**: `run_all_experiments.py` for all 29+ experiments
3. **Created analysis notebook**: `analysis/experiment_analysis.ipynb` with sections for each experiment type
4. **Documented experiment protocol**: `notes/experimental_protocol.tex/.pdf` with full methodology
5. **Submitted all training jobs**: 29 HTCondor jobs for core, ablation, and momentum experiments

---

## References

- **Physics:** LHCb Tracking TDR, Lorentz force equations
- **PINN:** Raissi et al. "Physics-informed neural networks" (2019)
- **RK Methods:** Butcher "Numerical Methods for ODEs" (2008)
- **Field Map:** `twodip.rtf` - LHCb dipole field measurements

---

*Last updated: January 22, 2026*
