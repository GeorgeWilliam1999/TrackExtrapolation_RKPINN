# Track Extrapolation ML Experiments

This directory contains organized experimental results for ML-based track extrapolation in LHCb.

---

## 📊 Summary of Results (January 2025)

### Best Model: **MLP with SiLU Activation**

| Metric | Value |
|--------|-------|
| **Mean Error** | 0.21 mm |
| **P95 Error** | 0.54 mm |
| **Speedup** | ~30,000× vs RK8 |
| **Parameters** | 25,924 |

### Key Findings

1. ✅ **SiLU activation wins** (0.21mm vs 0.63mm tanh vs 0.77mm ReLU)
2. ❌ **PINN models failed** (physics loss formulation incorrect)
3. ❌ **Weighted loss made things worse** (too aggressive momentum weighting)
4. ✅ **Data-driven learning works** (implicit physics from examples)
5. ✅ **ONNX export ready** for C++ deployment

### Model Performance Timeline

| Date | Model | Mean Error | Notes |
|------|-------|------------|-------|
| 2024-11 | MLP v1 (tanh) | 1.48 mm | First working model |
| 2024-12 | PINN v1 | 1.28 mm | Physics loss helped (slightly) |
| 2024-12 | Weighted | 2.5+ mm | Made things worse |
| 2025-01 | MLP SiLU | **0.21 mm** | **Current best** |
| 2025-01 | PINN λ=0.01 | 18.8 mm | Physics loss failed |
| 2025-01 | PINN λ=0.2 | 328.9 mm | Catastrophic failure |

---

## 📂 Directory Structure

```
experiments/
│
├── README.md                       # This file
├── experiment_log.csv              # Master log of all experiments
│
├── baseline/                       # Initial experiments
│   ├── v1_positive_qop/            # Single charge training
│   │   └── [models, plots, logs]
│   └── v2_both_charges/            # Full charge spectrum (production baseline)
│       └── [models, plots, logs]
│
├── architecture/                   # Network architecture studies
│   ├── deeper_networks/            # 4-6 layer experiments
│   ├── wider_networks/             # 256-512 neuron layers
│   └── skip_connections/           # ResNet-style architectures
│
├── momentum_studies/               # Momentum range experiments
│   ├── low_p_05_2gev/              # Low momentum (hardest region)
│   ├── mid_p_2_10gev/              # Medium momentum
│   └── high_p_10_100gev/           # High momentum (easiest)
│
├── physics_informed/               # PINN experiments
│   ├── lorentz_loss/               # Lorentz force constraint (FAILED)
│   └── energy_conservation/        # Energy loss constraint
│
├── data_augmentation/              # Data sampling strategies
│   ├── dense_grid/                 # Uniform grid sampling
│   └── random_sampling/            # Random phase space sampling
│
├── field_maps/                     # Magnetic field studies
│   ├── simplified/                 # Simplified dipole model
│   └── simcond/                    # Full SimCond field map
│
├── weighted_loss/                  # Loss weighting experiments (FAILED)
│   ├── README.md                   # Results documentation
│   ├── train_weighted.py
│   ├── train_weighted_v2.py
│   └── training_log.txt
│
├── onnx_export/                    # Model export for deployment
│   ├── README.md
│   ├── export_onnx.py
│   ├── export_onnx_v2.py
│   ├── mlp_full_domain.onnx        # Exported MLP model
│   ├── mlp_full_domain_norm.json   # Normalization parameters
│   ├── pinn_full_domain.onnx       # Exported PINN model
│   └── pinn_full_domain_norm.json
│
└── production/                     # Production-ready models
    └── best_model/                 # Final model for deployment
```

---

## 🔬 Experiment Details

### 1. Baseline Experiments (`baseline/`)

**Goal:** Establish baseline MLP performance

| Version | Charge | Samples | Error | Notes |
|---------|--------|---------|-------|-------|
| v1 | + only | 50K | ~2 mm | Initial test |
| v2 | ±both | 50K | 1.48 mm | Production baseline |

**Conclusion:** Training on both charges is essential for correct physics.

### 2. Architecture Studies (`architecture/`)

**Goal:** Find optimal network size

| Architecture | Layers | Params | Error | Status |
|--------------|--------|--------|-------|--------|
| tiny | 64-32 | 2,692 | - | Not trained |
| small | 128-64 | 9,796 | - | Not trained |
| **medium** | 128-128-64 | 25,924 | **0.21 mm** | ✅ Default |
| large | 256-256-128-64 | 140,228 | - | Not trained |
| xlarge | 512-512-256-128 | 536,836 | - | Not trained |

**Status:** Only medium architecture tested with SiLU. Others need training.

**Next Step:** Train all architectures to verify medium is optimal.

### 3. Momentum Studies (`momentum_studies/`)

**Goal:** Understand momentum-dependent performance

| Range | Samples | Typical Error | Notes |
|-------|---------|---------------|-------|
| 0.5-2 GeV | 10K | ~3-5 mm | Hardest (most bending) |
| 2-10 GeV | 20K | ~1-2 mm | Medium difficulty |
| 10-100 GeV | 20K | ~0.5 mm | Easiest (straight) |

**Key Finding:** Low momentum tracks are hardest due to strong magnetic bending.

### 4. Physics-Informed (`physics_informed/`)

**Goal:** Improve generalization with physics constraints

#### Lorentz Loss (FAILED ❌)

Physics loss components:
```python
# Position consistency (wrong assumption!)
L_pos_x = |x_out - (x_in + dz × tx_in)|²
L_pos_y = |y_out - (y_in + dz × ty_in)|²

# Bending constraint (oversimplified!)
L_bend = |Δtx_out - κ × q/p|²

# Vertical penalty
L_ty = |Δty_out|²
```

**Results:**

| λ (physics weight) | Mean Error | Notes |
|--------------------|------------|-------|
| 0.00 (MLP) | 0.21 mm | Baseline |
| 0.01 | 18.8 mm | 90× worse |
| 0.05 | 106.7 mm | 500× worse |
| 0.10 | 197.2 mm | 940× worse |
| 0.20 | 328.9 mm | 1570× worse |

**Root Cause:** Physics loss assumes straight-line propagation between start/end points, which is physically wrong in a non-uniform magnetic field. Higher λ enforces wrong constraints harder → worse performance.

**Recommendation:** Abandon current PINN approach OR redesign with Neural ODEs.

### 5. Weighted Loss (`weighted_loss/`)

**Goal:** Improve low-momentum performance via loss weighting

```python
weights = (P_max / P)^power  # Higher weight for low momentum
loss = (weights * MSE).mean()
```

| Power | Low-p Error | High-p Error | Total | Notes |
|-------|-------------|--------------|-------|-------|
| 0 (uniform) | 2.5 mm | 1.0 mm | 1.48 mm | Baseline |
| 1 | 2.2 mm | 1.5 mm | 1.8 mm | Worse overall |
| 2 | 2.0 mm | 2.2 mm | 2.1 mm | Much worse |

**Conclusion:** Aggressive momentum weighting hurts high-p performance more than it helps low-p. Not worth the trade-off.

### 6. ONNX Export (`onnx_export/`)

**Goal:** Export models for C++ inference

**Status:** ✅ Working

Files:
- `mlp_full_domain.onnx` - Best MLP model
- `mlp_full_domain_norm.json` - Input/output normalization
- `pinn_full_domain.onnx` - PINN model (deprecated)

Usage in C++:
```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
// See ml_models/src/TrackMLPExtrapolator.cpp
```

---

## 📋 Experiment Log Format

The `experiment_log.csv` file tracks all experiments:

```csv
date,experiment,model,architecture,activation,epochs,samples,mean_error,p95_error,notes
2025-01-13,activation_study,mlp_act_silu,128-128-64,silu,2000,50000,0.207,0.543,Best model
2025-01-13,pinn_study,pinn_lambda_0_01,128-128-64,tanh,2000,50000,18.85,47.23,Physics loss failed
```

Add new experiments here for tracking.

---

## 🎯 Lessons Learned

### What Worked ✅
1. **SiLU activation** - 3× better than tanh with no extra cost
2. **Data-driven learning** - Let the network learn physics from data
3. **Medium architecture** - 128-128-64 is good balance
4. **Both charges** - Essential for correct physics

### What Failed ❌
1. **Physics-informed loss** - Oversimplified constraints hurt more than help
2. **Momentum weighting** - Trade-off not worthwhile
3. **Very deep networks** - No clear benefit (needs more testing)

### What Needs Testing ⏳
1. **Architecture sweep** - tiny to xlarge
2. **Learning rate schedules** - Cosine annealing, warmup
3. **Neural ODEs** - Proper physics integration
4. **Bayesian networks** - Uncertainty quantification

---

## 📚 References

- [Model Investigation Notebook](../model_investigation.ipynb) - Detailed analysis
- [ML Models README](../ml_models/README.md) - Training documentation
- [Main README](../README.md) - Project overview

---

**Last Updated:** January 2025
