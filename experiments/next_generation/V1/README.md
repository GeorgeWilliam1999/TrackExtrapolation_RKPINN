# V1 - Initial MLP/PINN Experiments
## Status: Deprecated - Superseded by V2/V3

## Directory Structure

```
V1/
├── README.md              # This file
├── data_generation/       # Data generation scripts
│   ├── README.md
│   └── generate_data.py
├── models/                # Neural network architectures
│   ├── README.md
│   ├── architectures.py
│   └── train.py
├── training/              # Training job scripts
│   ├── README.md
│   ├── train_wrapper.sh
│   └── jobs/              # HTCondor job files
├── cluster/               # Cluster submission
│   ├── README.md
│   └── submit_full_suite_gpu.sub
├── trained_models/        # Symlinks to trained checkpoints
│   └── README.md
├── analysis/              # Analysis results
│   └── README.md
└── utils/                 # Utility modules
    ├── README.md
    └── *.py
```

## Overview

V1 was the first generation of neural network track extrapolators exploring MLP, PINN, 
and RK-PINN architectures for learning track propagation in the LHCb magnetic field.

**Key Result**: V1 models had several issues that led to their deprecation:
1. PINN Initial Condition (IC) failure - networks ignored z_frac input
2. All models trained with fixed dz=8000mm - cannot generalize to other step sizes
3. Various architecture depths explored but none optimal

## Training Configuration

### Data Characteristics
- **Source**: Runge-Kutta 4th order integration with real LHCb field map (twodip.rtf)
- **Samples**: ~50M tracks
- **dz**: **FIXED at 8000mm** (major limitation discovered later)
- **z_start**: Varied VELO-to-TStation region
- **Momentum**: Full range 1-200 GeV/c
- **Polarity**: MagUp and MagDown

### Input/Output Format
```
Input:  [x, y, tx, ty, q/p, dz] -> 6 features
        dz = 8000 for ALL samples (discovered to be problematic)
Output: [x_out, y_out, tx_out, ty_out] -> 4 features
```

### Models Trained (53 total)
See `trained_models/` directory for full list. Key categories:

**MLP Variants** (15 models):
- `mlp_tiny_v1`: [64, 64, 32] - 6.6K params
- `mlp_small_v1`: [128, 128, 64] - 26K params  
- `mlp_medium_v1`: [256, 256, 128] - 100K params
- `mlp_large_v1`: [512, 512, 256, 128] - 530K params
- `mlp_wide_v1`: [512, 512, 256, 128] - 530K params
- `mlp_balanced_v1`: [256, 256, 128] - 100K params
- `mlp_narrow_deep_v1`: [64, 64, 64, 64, 32] - 17K params
- `mlp_wide_shallow_v1`: [512, 256] - 135K params

**PINN Variants** (16 models):
- `pinn_standard_v1`, `pinn_large_v1`, `pinn_moderate_v1`, etc.
- `pinn_light_v1`, `pinn_strong_v1`, `pinn_weak_v1`
- Various λ_pde / λ_ic combinations

**RK-PINN Variants** (11 models):
- `rkpinn_balanced_v1`, `rkpinn_large_v1`, `rkpinn_strong_v1`
- `rkpinn_coll5_v1`, `rkpinn_coll10_v1`, `rkpinn_coll20_v1`
- `rkpinn_wide_v1`

**Momentum-binned Studies** (9 models):
- `*_low_p`: p < 10 GeV/c
- `*_mid_p`: 10 < p < 50 GeV/c
- `*_high_p`: p > 50 GeV/c

## Known Issues

### Issue 1: PINN IC Architecture Failure
**Symptom**: PINN and RK-PINN models output constant values regardless of `z_frac` input.
**Root Cause**: Standard PINN architecture failed to learn z-dependence. The network 
effectively ignored the z_frac coordinate and learned average track propagation.
**Impact**: All PINN models produced incorrect intermediate states.
**Resolution**: Fixed in V2 with residual architecture: `Output = IC + z_frac × NetworkCorrection`

### Issue 2: Fixed dz=8000mm Training
**Symptom**: When deployed with dz≠8000, models produce NaN/Inf or incorrect results.
**Root Cause**: All training data had dz=8000mm exactly, so:
  - `input_std[dz] ≈ 1e-9` (essentially zero)
  - Normalization: `(dz - mean) / std` causes explosion for dz≠8000
  - Model never learned dz-dependence
**Impact**: Models cannot be used for variable step sizes in deployment.
**Resolution**: Fixed in V3 with variable dz training data.

### Issue 3: Normalization Constants
All V1 models have these normalization issues:
```
Feature    Mean        Std
-------------------------------
x          ~0.0        ~300 mm
y          ~0.0        ~200 mm  
tx         ~0.0        ~0.3
ty         ~0.0        ~0.2
q/p        ~0.0        ~0.1 c/GeV
dz         8000.0      ~1e-9     <- PROBLEM!
```

## Model Files
Each trained model contains:
- `model.pt` - PyTorch checkpoint
- `config.json` - Training configuration
- `metadata.json` - Input/output normalization stats
- `results.json` - Training/validation metrics

## Lessons Learned

1. **Test with variable inputs**: Training data should cover the full operational range
2. **Verify PINN z-dependence**: Check that physics-informed models actually use z coordinate
3. **Check normalization stats**: Ensure no near-zero standard deviations

## Directory Structure
```
V1/
├── README.md (this file)
└── (V1 models are in ../trained_models/*_v1/)
```

## See Also
- [V2/README.md](../V2/README.md) - Fixed PINN architecture
- [V3/README.md](../V3/README.md) - Variable dz support (current development)
