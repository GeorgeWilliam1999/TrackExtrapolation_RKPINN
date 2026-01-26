# Dependency Graph: next_generation Project

This document maps the import dependencies and data flow across the project.

## Overview

```
                              ┌─────────────────────────────────────────────────────────────┐
                              │                    UNIFIED PHYSICS                           │
                              │                                                              │
                              │   utils/magnetic_field.py                                   │
                              │   ├── C_LIGHT = 2.99792458e-4                               │
                              │   ├── GaussianFieldNumpy / GaussianFieldTorch               │
                              │   └── InterpolatedFieldNumpy / InterpolatedFieldTorch       │
                              │                        ↑                                     │
                              │                    (field_maps/twodip.rtf)                  │
                              └───────────────┬─────────────────────────────────────────────┘
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                │                             │                             │
                ▼                             ▼                             ▼
    ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
    │ utils/rk4_propagator  │   │ data_generation/      │   │ models/architectures  │
    │                       │   │ generate_cpp_data.py  │   │                       │
    │ - RK4Integrator       │   │                       │   │ - MLP                 │
    │ - Uses real field     │◄──│ - HTCondor batch gen  │   │ - PINN                │
    │   interpolation       │   │ - Uses real field     │   │ - RK_PINN             │
    └───────────┬───────────┘   └───────────────────────┘   │ - Uses unified field  │
                │                                            └───────────┬───────────┘
                ▼                                                        │
    ┌───────────────────────┐                                           │
    │ data_generation/      │                                           │
    │ generate_data.py      │                                           │
    │                       │                                           │
    │ - Local data gen      │                                           │
    │ - Parallel processing │                                           │
    └───────────────────────┘                                           │
                                                                        │
                                      ┌─────────────────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────────┐
                         │     models/train.py          │
                         │                              │
                         │  Training Pipeline:          │
                         │  - Data loading              │
                         │  - Training loop             │
                         │  - Checkpoint management     │
                         │  - Early stopping            │
                         └───────────────┬──────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                │                        │                        │
                ▼                        ▼                        ▼
    ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
    │ models/evaluate.py│   │ models/export_    │   │ analysis/         │
    │                   │   │ onnx.py           │   │ analyze_models.py │
    │ Model evaluation  │   │                   │   │                   │
    │ metrics export    │   │ ONNX export for   │   │ Comprehensive     │
    │                   │   │ C++ integration   │   │ analysis/plots    │
    └───────────────────┘   └───────────────────┘   └───────────────────┘
```

## Module Descriptions

### Core Physics (SINGLE SOURCE OF TRUTH)

#### `utils/magnetic_field.py`
**Status**: ✅ Unified field module - ALL other code imports from here

| Export | Description |
|--------|-------------|
| `C_LIGHT` | Physical constant: 2.99792458e-4 |
| `GaussianFieldNumpy` | Analytical Gaussian approximation (NumPy) |
| `GaussianFieldTorch` | Analytical Gaussian approximation (PyTorch, differentiable) |
| `InterpolatedFieldNumpy` | Trilinear interpolation of twodip.rtf (NumPy) |
| `InterpolatedFieldTorch` | Trilinear interpolation of twodip.rtf (PyTorch, differentiable) |
| `get_field_numpy()` | Factory function for NumPy field models |
| `get_field_torch()` | Factory function for PyTorch field models |

**Field Map**: `field_maps/twodip.rtf` (81×81×146 grid, 957906 points)

### Data Generation

#### `data_generation/generate_cpp_data.py`
**Purpose**: HTCondor batch data generation
**Imports**: `magnetic_field.get_field_numpy, C_LIGHT, InterpolatedFieldNumpy`
**Uses**: Real field interpolation ✅

#### `data_generation/generate_data.py`
**Purpose**: Local parallel data generation
**Imports**: `rk4_propagator.RK4Integrator`, `magnetic_field.get_field_numpy`
**Uses**: Real field interpolation ✅

#### `data_generation/merge_batches.py`
**Purpose**: Combine HTCondor batch outputs
**Imports**: None (standalone utility)

### Propagation

#### `utils/rk4_propagator.py`
**Purpose**: RK4 numerical integrator for track propagation
**Imports**: `magnetic_field.get_field_numpy, C_LIGHT`
**Uses**: Real field interpolation ✅

#### `utils/archived/rk4_propagator.py` ⚠️
**Status**: ARCHIVED - DO NOT USE
**Contains**: Old Gaussian approximation (incorrect)
**Note**: Kept for reference only

### Neural Network Models

#### `models/architectures.py`
**Purpose**: Neural network architectures (MLP, PINN, RK_PINN)
**Imports**: `magnetic_field.GaussianFieldTorch, InterpolatedFieldTorch, get_field_torch, C_LIGHT`
**Uses**: Unified field module ✅

**Provides**:
- `MLP` - Standard feedforward network
- `PINN` - Physics-informed with Lorentz force constraint
- `RK_PINN` - Multi-stage RK4-inspired architecture
- `create_model()` - Factory function
- `GaussianMagneticField`, `InterpolatedMagneticField` - Re-exported from unified module

### Training & Evaluation

#### `models/train.py`
**Purpose**: Training pipeline
**Imports**: `architectures.create_model`
**Uses**: Unified field via architectures.py ✅

#### `models/evaluate.py`
**Purpose**: Model evaluation and metrics
**Imports**: `architectures.MODEL_REGISTRY, create_model`
**Uses**: Unified field via architectures.py ✅

#### `models/export_onnx.py`
**Purpose**: Export trained models to ONNX format
**Imports**: `architectures.create_model`
**Uses**: Unified field via architectures.py ✅

### Analysis

All analysis scripts import from `models/architectures.py`:

| Script | Purpose |
|--------|---------|
| `analysis/analyze_models.py` | Comprehensive model analysis |
| `analysis/timing_benchmark.py` ⭐ | Detailed timing analysis |
| `analysis/physics_analysis.py` | Physics constraint analysis |
| `analysis/trajectory_visualizer.py` | 3D trajectory visualization |
| `analysis/generate_paper_quality_plots.py` ⭐ | Publication-quality figures |
| `analysis/timing_comparison_plots.py` | Timing comparisons |
| `analysis/run_analysis.py` | Run analysis pipeline |

## Data Flow

```
Field Map (twodip.rtf)
         │
         ▼
┌─────────────────────────┐
│  magnetic_field.py      │
│  Load & interpolate     │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌─────────┐   ┌─────────────────┐
│ NumPy   │   │ PyTorch         │
│ Models  │   │ Models          │
└────┬────┘   └────────┬────────┘
     │                 │
     ▼                 ▼
┌─────────────┐  ┌───────────────┐
│ RK4         │  │ PINN Training │
│ Data Gen    │  │ Physics Loss  │
└─────┬───────┘  └───────┬───────┘
      │                  │
      ▼                  ▼
┌───────────────────────────────┐
│      Training Data (NPZ)      │
│  50M tracks × [x,y,tx,ty,qop] │
└───────────────────────────────┘
              │
              ▼
┌───────────────────────────────┐
│      Trained Models (.pth)    │
│  MLP, PINN, RK_PINN variants  │
└───────────────────────────────┘
              │
              ▼
┌───────────────────────────────┐
│      ONNX Export (.onnx)      │
│  For C++ integration          │
└───────────────────────────────┘
```

## Consistency Verification

### Field Model Consistency

All active code now uses the unified `magnetic_field.py` module:

| File | Field Source | Status |
|------|--------------|--------|
| `utils/rk4_propagator.py` | `magnetic_field.get_field_numpy()` | ✅ |
| `data_generation/generate_cpp_data.py` | `magnetic_field.get_field_numpy()` | ✅ |
| `data_generation/generate_data.py` | `magnetic_field.get_field_numpy()` | ✅ |
| `models/architectures.py` | `magnetic_field.get_field_torch()` | ✅ |

### C_LIGHT Constant Consistency

| Location | Value | Status |
|----------|-------|--------|
| `magnetic_field.py` | 2.99792458e-4 | ✅ Primary definition |
| `architectures.py` | Imported from magnetic_field | ✅ |
| `rk4_propagator.py` | Imported from magnetic_field | ✅ |
| `generate_cpp_data.py` | Imported from magnetic_field | ✅ |

## Notes

### Archived Code

The `utils/archived/` directory contains OLD implementations that should NOT be used:

- `rk4_propagator.py` - Uses Gaussian approximation, NOT real field interpolation

### Field Map Details

- **File**: `field_maps/twodip.rtf`
- **Grid**: 81 × 81 × 146 points
- **Range**: x,y: [-4000, 4000] mm, z: [-500, 14000] mm
- **Spacing**: 100 mm
- **Peak By**: -1.0320 T at (0, 0, 5000)
- **Interpolation Error**: O(h²) ≈ 10⁻⁵ T

### HTCondor Data Generation

Current batch: Cluster 3853287 (5000 jobs × 10k tracks = 50M tracks)

Old approximate data preserved in: `data_generation/data_approximate_gaussian/`

---
*Last Updated: January 2026*
*Author: G. Scriven*
