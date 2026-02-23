# V2 - Shallow-Wide Architecture & PINN Residual Fix
## Status: ⚠️ PINN Architecture Issue Identified (Feb 2026) - See PINN_ARCHITECTURE_ISSUE.md

> **Critical Finding:** V2 PINN models underperform MLPs by ~2× due to a fundamental architectural limitation.
> The residual formulation uses position-independent corrections that are linearly scaled by z_frac,
> which cannot capture the position-dependent curvature in the variable LHCb magnetic field.
> 
> **Read:** [PINN_ARCHITECTURE_ISSUE.md](./PINN_ARCHITECTURE_ISSUE.md) for full analysis and recommended fixes.

## Directory Structure

```
V2/
├── README.md              # This file
├── data_generation/       # Data generation (same data as V1)
│   ├── README.md
│   └── generate_data.py
├── models/                # Neural network architectures
│   ├── README.md
│   ├── architectures.py   # Includes residual PINN
│   └── train.py
├── training/              # Training configurations
│   ├── README.md
│   └── configs/           # JSON config files
├── cluster/               # Cluster submission
│   ├── README.md
│   ├── submit_v2_shallow_wide.sub
│   ├── submit_v2_pinn_retrain.sub
│   └── v2_jobs.txt
├── trained_models/        # Symlinks to trained checkpoints
│   └── README.md
├── analysis/              # Analysis results
│   └── README.md
└── utils/                 # Utility modules
    └── README.md
```

## Overview

V2 addressed the PINN initial condition (IC) failure from V1 by implementing a residual
architecture. V2 also explored "shallow-wide" MLP architectures (1-3 layers with 256-1024 
neurons) that showed better performance than deep-narrow networks.

**Key Achievements**:
1. ✅ Fixed PINN IC issue with residual formulation
2. ✅ Identified optimal shallow-wide architecture for MLPs
3. ❌ Still trained with fixed dz=8000mm (discovered later)

## Architecture Changes from V1

### PINN Residual Architecture (New in V2)
The V1 PINN architecture failed to learn z-dependence. V2 implements:

```
Standard (V1):     Output = Network(x, y, tx, ty, q/p, z_frac)
Residual (V2):     Output = IC + z_frac × Network(x, y, tx, ty, q/p, z_frac)
```

Where:
- `IC` = initial state [x, y, tx, ty] (passed through as-is)
- `z_frac` = normalized position along track (0 at start, 1 at end)
- Network learns the **correction** to initial state, not absolute position

This ensures:
- At z_frac=0: Output = IC (exact initial condition)
- At z_frac=1: Output = IC + Network correction (final state)
- Smooth interpolation for intermediate z values

### Shallow-Wide MLP Architecture
Analysis of V1 models showed shallow-wide networks outperformed deep-narrow:

| Architecture | Layers | Width | Params | Validation Loss |
|--------------|--------|-------|--------|-----------------|
| Deep-narrow  | 5      | 64    | 17K    | 0.0045          |
| Medium       | 4      | 128   | 100K   | 0.0018          |
| Shallow-wide | 2-3    | 256-512 | 135K | **0.0008**     |

V2 focuses on shallow-wide: 1-3 layers with 256-1024 neurons per layer.

## Training Configuration

### Data Characteristics
- **Source**: Same as V1 - RK4 integration with real LHCb field map
- **Samples**: ~50M tracks
- **dz**: **FIXED at 8000mm** (still problematic!)
- **z_start**: VELO-to-TStation region (varied)
- **Momentum**: Full range 1-200 GeV/c
- **Polarity**: MagUp and MagDown

### Input/Output Format
```
Input:  [x, y, tx, ty, q/p, dz] -> 6 features
Output: [x_out, y_out, tx_out, ty_out] -> 4 features

For PINN/RK-PINN: z_frac added as 7th input for physics loss computation
```

## Models Trained (22 total)

### MLP V2 Models (9)
| Model | Architecture | Parameters | Description |
|-------|--------------|------------|-------------|
| `mlp_v2_single_256` | [256] | 2.8K | Single layer, 256 wide |
| `mlp_v2_single_512` | [512] | 5.6K | Single layer, 512 wide |
| `mlp_v2_single_1024` | [1024] | 11K | Single layer, 1024 wide |
| `mlp_v2_shallow_256` | [256, 128] | 36K | 2 layers |
| `mlp_v2_shallow_512` | [512, 256] | 135K | 2 layers |
| `mlp_v2_shallow_512_256` | [512, 256] | 135K | 2 layers |
| `mlp_v2_shallow_1024_256` | [1024, 256] | 270K | 2 layers |
| `mlp_v2_shallow_1024_512` | [1024, 512] | 540K | 2 layers, widest |

### PINN V2 Models (7)
All use residual architecture:
| Model | Architecture | λ_pde | λ_ic | Description |
|-------|--------------|-------|------|-------------|
| `pinn_v2_single_256` | [256] | 1.0 | 1.0 | Single layer |
| `pinn_v2_single_512` | [512] | 1.0 | 1.0 | Single layer |
| `pinn_v2_shallow_256` | [256, 128] | 1.0 | 1.0 | 2 layers |
| `pinn_v2_shallow_512` | [512, 256] | 1.0 | 1.0 | 2 layers |
| `pinn_v2_shallow_512_256` | [512, 256] | 1.0 | 1.0 | 2 layers |
| `pinn_v2_shallow_1024_256` | [1024, 256] | 1.0 | 1.0 | 2 layers |
| `pinn_v2_shallow_1024_512` | [1024, 512] | 1.0 | 1.0 | 2 layers, widest |

### RK-PINN V2 Models (6)
All use residual architecture:
| Model | Architecture | n_coll | Description |
|-------|--------------|--------|-------------|
| `rkpinn_v2_single_256` | [256] | 10 | Single layer |
| `rkpinn_v2_single_512` | [512] | 10 | Single layer |
| `rkpinn_v2_shallow_256` | [256, 128] | 10 | 2 layers |
| `rkpinn_v2_shallow_512` | [512, 256] | 10 | 2 layers |
| `rkpinn_v2_shallow_512_256` | [512, 256] | 10 | 2 layers |
| `rkpinn_v2_shallow_1024_256` | [1024, 256] | 10 | 2 layers |

## Known Issues

### Issue: Fixed dz=8000mm (Same as V1)
Despite the PINN fix, V2 models still suffer from fixed dz training:
- `input_std[dz] ≈ 1e-9` causes normalization explosion for dz≠8000
- Models produce NaN/Inf or incorrect results when dz varies
- **Resolution**: V3 trains with variable dz ∈ [500, 12000] mm

### C++ Integration Note
When exporting V2 models to C++ binary format, the near-zero `input_std[dz]` must be 
handled specially. The export script patches small std values to prevent division by zero:
```cpp
// In TrackMLPExtrapolator.cpp
// If std < 1e-6, use mean value instead of normalization
```

## Normalization Statistics (V2)
```
Feature    Mean        Std          Notes
----------------------------------------------
x          ~0.0        ~300 mm      OK
y          ~0.0        ~200 mm      OK
tx         ~0.0        ~0.3         OK
ty         ~0.0        ~0.2         OK
q/p        ~0.0        ~0.1 c/GeV   OK
dz         8000.0      ~1e-9        PROBLEM - see V3
```

## Cluster Jobs

V2 training was submitted via:
- PINN retraining: Job 3933584 (completed)
- MLP shallow-wide: Jobs in `training/jobs/mlp_*_v2.sub`

## Results Summary

### Best MLP Performance (V2)
`mlp_v2_shallow_256` achieved:
- Position error: **~0.08 mm** (mean)
- Best accuracy among all V2 models
- Why it works: MLP takes full 6D input including dz/z_frac

### PINN Performance (V2) ⚠️
`pinn_v2_single_256` with residual architecture:
- Position error: **~0.15 mm** (mean) - **2× worse than MLP**
- IC issue fixed ✓ (correctly satisfies z=0 boundary condition)
- Physics loss decreases during training ✓
- **BUT:** Architecture cannot capture position-dependent curvature ❌

**Root Cause (Feb 2026 Analysis):**
- PINN encoder only sees `[x0, y0, tx0, ty0, qop]` (no position information!)
- Predicts single correction vector per track
- Correction is linearly scaled: `state(z) = baseline + z·correction`
- Cannot represent varying curvature in variable magnetic field
- See [PINN_ARCHITECTURE_ISSUE.md](./PINN_ARCHITECTURE_ISSUE.md) for detailed analysis

**Key Lesson:** Physics-informed architectures are only effective when the architecture has sufficient expressivity to represent the physics. The V2 residual formulation, while elegant for enforcing initial conditions, inadvertently constrains the solution space too severely.

## Directory Structure
```
V2/
├── README.md (this file)
└── (V2 models are in ../trained_models/*_v2_*/)
```

## See Also
- [V1/README.md](../V1/README.md) - Original experiments (deprecated)
- [V3/README.md](../V3/README.md) - Variable dz support (current development)
