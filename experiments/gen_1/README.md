# Neural Network Track Extrapolators - Next Generation

**Project Goal:** Systematically train and compare neural network architectures for LHCb track extrapolation, storing all training metrics for comprehensive analysis.

**Status:** ✅ V1 Complete (53 models) | ✅ V2 Complete (22 models) | ✅ V3/V4 Complete | 🔧 V5 Architecture Comparison Active

**Reference Baseline:** C++ RK4 (CashKarp): **2.50 μs/track** (measured via TrackExtrapolatorTesterSOA)

---

## Directory Structure

```
gen_1/
├── README.md                        # This file
├── DEPENDENCY_GRAPH.md              # Project dependencies
├── LHCB_STACK_MANUAL.md             # LHCb stack build instructions
├── MASTER_RESULTS.md/.tex/.pdf      # Master results summary
├── cross_version_analysis.ipynb      # Cross-version comparison notebook
├── explore_field_map.ipynb           # Field map visualization
├── nn_field_map_sizing.ipynb         # NN field map sizing analysis
├── physics_exploration.ipynb         # Physics exploration notebook
├── *.png                            # Generated analysis plots (13 files)
├── V1/                              # V1 experiments (deprecated, 53 models)
│   ├── analysis/          # Analysis scripts and notebooks (4 notebooks)
│   ├── benchmarking/      # C++ benchmarks
│   ├── cluster/           # HTCondor jobs
│   ├── data_generation/   # Data generation scripts
│   ├── models/            # Neural network code
│   ├── notes/             # Documentation
│   ├── paper/             # Paper drafts
│   ├── results/           # CSV results
│   ├── trained_models/    # Symlinks to models
│   ├── training/          # Training job scripts
│   └── utils/             # Utility modules
├── V2/                              # V2 experiments (shallow-wide, 22 models)
│   ├── analysis/          # V2-specific analysis (1 notebook)
│   ├── cluster/           # V2 HTCondor jobs
│   ├── data_generation/   # Same data as V1
│   ├── models/            # Includes residual PINN
│   ├── results/           # V2 results
│   ├── trained_models/    # Symlinks to V2 models
│   ├── training/          # V2 training configs
│   └── utils/             # Utility modules
├── V3/                              # V3 experiments (variable dz, 8 models)
│   ├── analysis/          # Benchmarks and notebooks (2 notebooks)
│   ├── cluster/           # V3 HTCondor jobs (~35 scripts)
│   ├── data_generation/   # Variable dz data gen
│   ├── deployment/        # Export to C++ binary
│   ├── models/            # MLP + PINN architectures
│   ├── trained_models/    # 8 trained models
│   ├── training/          # Training configs (12 configs)
│   └── utils/             # Utility modules
├── V4/                              # V4 experiments (PINN diagnosis, 23 models)
│   ├── analysis/          # Results collection
│   ├── cluster/           # V4 HTCondor jobs
│   ├── trained_models/    # 23 trained models (mlp, quad, zfrac)
│   └── training/          # V4 training scripts + configs
├── V5/                              # V5 experiments (PINN architecture fixes)
│   └── ...                # 5 architectures, 7 configs
├── deployment/                      # Export models to C++ (export_to_cpp.py)
├── trained_models/                  # All trained model checkpoints (75+ models)
├── models/                          # Placeholder (README.md only)
├── data_generation/                 # Placeholder (README.md only)
├── training/                        # Placeholder (README.md only)
└── archive/                         # Historical experiments
```

---

## ⚠️ VERSION HISTORY

| Version | Status | Key Change | Limitation |
|---------|--------|------------|------------|
| **V1** | Deprecated | Initial experiments (53 models) | PINN IC failure, fixed dz=8000mm |
| **V2** | Deprecated | PINN residual fix, shallow-wide (22 models) | Still fixed dz=8000mm |
| **V3** | Active | Variable dz training (500-12000mm) | MLP ~1mm, PINN ~50mm position |
| **V4** | Complete | PINN root-cause diagnosis + width sweep (23 models) | MLP/Quad/ZFrac trained, PINN fixes identified |
| **V5** | **Active** | 5 PINN architecture fixes (7 configs) | Under evaluation |

See version-specific documentation:
- [V1/README.md](V1/README.md) - Original experiments (deprecated)
- [V2/README.md](V2/README.md) - PINN residual architecture (deprecated)
- [V3/README.md](V3/README.md) - Variable dz support
- [V4/README.md](V4/README.md) - PINN architecture diagnosis ([full analysis](V4/PINN_ARCHITECTURE_DIAGNOSIS.md))
- [V5/README.md](V5/README.md) - PINN architecture fixes (QuadraticResidual, ZFrac, PDE-Residual, Compositional)

---

## 🏆 Current State of the Art (January 2026)

### Best Models

| Model | Position Error | Timing | Speedup vs C++ | Recommendation |
|-------|---------------|--------|----------------|----------------|
| `mlp_v2_single_256` | 0.065 mm | 0.83 μs | **3.0×** | ⭐ Best for speed |
| `mlp_v2_shallow_256` | 0.044 mm | 1.50 μs | 1.7× | Balanced |
| `mlp_v2_shallow_512_256` | **0.028 mm** | 1.93 μs | 1.3× | ⭐ Best accuracy |

### Key Findings

1. **MLP outperforms PINN/RK_PINN** - Simple MLPs achieve 0.03-0.07 mm accuracy, 10-100× better than physics-informed models in V1/V2.

2. **Shallow-wide beats deep-narrow** - 1-2 layer networks with 256-1024 neurons outperform deeper architectures for both accuracy AND speed.

3. **10 models faster than C++ RK4** - The fastest (mlp_v2_single_256) achieves 3× speedup with acceptable accuracy.

### ⚠️ PINN/RK_PINN Failure Analysis

**Problem Discovered:** The original PINN and RK_PINN architectures have a fundamental flaw - they fail to satisfy the Initial Condition (IC) constraint:

| z_frac | PINN Output | Expected |
|--------|-------------|----------|
| 0.0 (IC) | x=2768 mm | x=207 mm |
| 1.0 | x=2752 mm | x=1039 mm |

The physics loss stays constant (~1.7) throughout training while data loss decreases. This means the network learned to **ignore z_frac entirely** and outputs nearly constant values.

**Root Cause:** The PINN forward pass sets `x[:, 5] = 1.0` for all inputs during training. The network can minimize data_loss at z=1 without learning the trajectory - it just learns a direct mapping from initial state to final state, ignoring the z_frac input.

**Solution: PINN_v3 Architecture** - Uses explicit skip connections that make z_frac impossible to ignore:
```
Output = InitialState + z_frac × NetworkCorrection
```
At z=0, output = initial state (IC automatically satisfied!)

See [PINN_v3 Training](#pinn_v3-training) for details.

---

## Table of Contents

1. [Overview](#overview)
2. [Current State of the Art](#-current-state-of-the-art-january-2026)
3. [Experiment Design](#experiment-design)
4. [Model Architectures](#model-architectures)
5. [PINN_v3 Training](#pinn_v3-training)
6. [Training Configurations](#training-configurations)
7. [Running Experiments](#running-experiments)
8. [Loss Tracking & Convergence Analysis](#loss-tracking--convergence-analysis)
9. [Performance Benchmarking](#performance-benchmarking)
10. [Directory Structure](#directory-structure)
11. [Quick Reference](#quick-reference)

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

### 4. PINN_v3 (Physics-Informed with Skip Connections) ⭐ NEW

**Philosophy:** Fix the IC constraint problem with residual formulation.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PINN_v3 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input [5]                                  Output [4]         │
│  ┌─────────┐    Encoder     Correction                          │
│  │ x₀      │   ┌───────┐    ┌───────┐                          │
│  │ y₀      │──►│ FC    │───►│ Head  │──► [Δtx, Δty, Δx, Δy]    │
│  │ tx₀     │   │ FC    │    └───┬───┘                          │
│  │ ty₀     │   └───────┘        │                              │
│  │ q/p     │                    │ Correction                   │
│  └─────────┘                    ▼                              │
│       │                   ╔═══════════════════════════════╗    │
│       │  z_frac ──────────║  Output = Initial +           ║    │
│       │    │              ║          z_frac × Correction  ║    │
│       │    │              ╚═══════════════════════════════╝    │
│       │    │                    │                              │
│       │    └────────────────────┤                              │
│       │                         │                              │
│       │   ┌─────────────────────┴─────────────────────┐        │
│       │   │  x_out = x₀ + tx₀×z×dz + z×Δx            │        │
│       │   │  y_out = y₀ + ty₀×z×dz + z×Δy            │        │
│       │   │  tx_out = tx₀ + z×Δtx                     │        │
│       │   │  ty_out = ty₀ + z×Δty                     │        │
│       └───►│                                          │        │
│           └───────────────────────────────────────────┘        │
│                                                                 │
│   KEY: At z=0, Output = Initial State (IC GUARANTEED!)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Innovation:**
- **Residual formulation:** Output = InitialState + z_frac × NetworkCorrection
- At z=0: Output exactly equals initial state (IC is **automatically satisfied**)
- At z=1: Network learns the full displacement
- **z_frac modulation:** Network CANNOT ignore z_frac because correction is multiplied by it

**Losses:**
- `data_loss = MSE(y(z=1), ground_truth)` — endpoint match
- `ic_loss = ~0` — automatically satisfied by construction!
- `pde_loss = Σ ||dy/dz - Lorentz(y, B)||²` — PDE residual

**Advantages:**
- IC constraint satisfied by design (no optimization needed)
- Physics loss actually contributes to gradient
- Network learns corrections, not raw outputs (easier task)

**Disadvantages:** Slightly more complex architecture, assumes straight-line baseline

---

## PINN_v3 Training

### Why PINN_v3?

The original PINN/RK_PINN architectures failed because:
1. Network could minimize data_loss without learning physics
2. IC constraint not enforced in network structure
3. z_frac input was effectively ignored

PINN_v3 fixes this with a residual formulation that **guarantees** IC satisfaction.

### Training PINN_v3 Models

```bash
# Train single model
cd experiments/gen_1
python training/train_pinn_v3.py --preset pinn_v3_shallow_256

# Submit all V3 configurations to cluster
condor_submit cluster/submit_pinn_v3.sub

# Submit specific configuration
condor_submit cluster/submit_pinn_v3.sub -append "Queue experiment_name from (pinn_v3_shallow_512)"
```

### V3 Configurations

| Model | Hidden Dims | Expected Timing | Purpose |
|-------|-------------|-----------------|---------|
| `pinn_v3_single_256` | [256] | ~1 μs | Fastest physics-informed |
| `pinn_v3_single_512` | [512] | ~1.5 μs | Balance |
| `pinn_v3_shallow_256` | [256, 256] | ~2 μs | Baseline |
| `pinn_v3_shallow_512` | [512, 512] | ~3 μs | High capacity |
| `pinn_v3_shallow_512_256` | [512, 256] | ~2.5 μs | Tapered |
| `pinn_v3_shallow_1024_256` | [1024, 256] | ~4 μs | Maximum capacity |

### Expected Outcomes

With PINN_v3:
- IC error should be **exactly 0** (by construction)
- PDE loss should decrease during training (unlike original PINN)
- Final accuracy may be similar to MLP but with physics guarantees
- Useful for extrapolation beyond training domain

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

| Model | Params | pos_mean (mm) | pos_95% (mm) | Time (μs/track) | vs C++ (2.50 μs) |
|-------|--------|---------------|--------------|-----------------|------------------|
| mlp_tiny | 5,252 | 0.024 | 0.052 | 1.10 | 2.27× faster |
| mlp_small | 20,228 | 0.023 | 0.051 | 1.15 | 2.17× faster |
| mlp_medium | 99,588 | 0.022 | 0.049 | 1.35 | 1.85× faster |
| mlp_wide | 467,972 | 0.021 | 0.047 | 1.75 | 1.43× faster |
| pinn_weak | ~20k | 0.030 | 0.065 | 1.55 | 1.61× faster |
| pinn_moderate | ~100k | 0.028 | 0.061 | 2.10 | 1.19× faster |
| rkpinn_coll5 | ~25k | 0.025 | 0.055 | 3.10 | 0.81× slower |
| rkpinn_coll10 | ~25k | 0.024 | 0.053 | 3.60 | 0.69× slower |
| **C++ RK4** | N/A | 0.000 (ref) | 0.000 (ref) | **2.50** (baseline) | 1.00× |

**Note:** All ML models achieve sub-0.1 mm position accuracy with sub-1e-5 slope error, meeting physics requirements.

---

## Directory Structure

```
gen_1/
├── README.md                        # This file
├── DEPENDENCY_GRAPH.md              # Project dependencies
├── LHCB_STACK_MANUAL.md             # LHCb stack build instructions
├── MASTER_RESULTS.md/.tex/.pdf      # Master results summary
├── cross_version_analysis.ipynb      # Cross-version comparison
├── explore_field_map.ipynb           # Field map visualization
├── nn_field_map_sizing.ipynb         # NN field map sizing
├── physics_exploration.ipynb         # Physics exploration
├── *.png                            # Analysis plots (13 files)
├── V1/                              # 53 models (deprecated)
├── V2/                              # 22 models (deprecated)
├── V3/                              # 8 models (variable dz)
├── V4/                              # 23 models (PINN diagnosis + width sweep)
├── V5/                              # PINN architecture fixes (active)
├── deployment/                      # export_to_cpp.py
├── trained_models/                  # 75+ model checkpoints
├── models/                          # Placeholder (README.md only)
├── data_generation/                 # Placeholder (README.md only)
├── training/                        # Placeholder (README.md only)
└── archive/                         # Historical experiments
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
| **V1 Training** | ✅ Complete | 53 models (cluster 3880818), 10 epochs |
| **V2 Training** | ✅ Complete | 22 shallow-wide models (cluster 3891076), 20 epochs |
| Unified Runner | ✅ Ready | `run_all_experiments.py` |
| Analysis Notebook | ✅ Ready | `analysis/experiment_analysis.ipynb` |
| Experiment Protocol | ✅ Ready | `notes/experimental_protocol.pdf` |
| ONNX Export | ✅ Ready | `models/export_onnx.py` functional |

### Key Results Summary

**C++ Reference Baseline:** 2.50 μs/track (CashKarp RK4, measured via TrackExtrapolatorTesterSOA)

**Best V1 Results:**
| Model Type | Best Model | Position Error (mm) | Time (μs/track) | Speedup vs C++ |
|------------|-----------|---------------------|-----------------|----------------|
| MLP | mlp_tiny_v1 | 0.024 | 1.10 | 2.27× faster |
| PINN | pinn_weak_v1 | 0.030 | 1.55 | 1.61× faster |
| RK_PINN | rkpinn_coll5_v1 | 0.025 | 3.10 | 0.81× (slower) |

**V2 Design Rationale:** Based on timing analysis:
- Depth ↔ Time: weak correlation (r=0.37)
- Width ↔ Time: moderate correlation (r=0.60)
- Parameters ↔ Time: strong correlation (r=0.83)

V2 uses shallow (1-2 layers) + wide (256-1024 neurons) architectures to maximize speed.

### Recent Updates (January 2026)

1. **V1 Training Complete**: 53 models trained (MLP, PINN, RK_PINN variants)
2. **V2 Training Complete**: 22 shallow-wide models (optimized for speed)
3. **Correct Baseline Established**: C++ RK4 = 2.50 μs/track (not 75 μs)
4. **Analysis Updated**: All notebooks use correct reference timing
5. **Fixed PINN/RK_PINN field model**: Uses `InterpolatedFieldTorch` (real field map)

---

## References

- **Physics:** LHCb Tracking TDR, Lorentz force equations
- **PINN:** Raissi et al. "Physics-informed neural networks" (2019)
- **RK Methods:** Butcher "Numerical Methods for ODEs" (2008)
- **Field Map:** `twodip.rtf` - LHCb dipole field measurements

---

*Last updated: March 2026*
