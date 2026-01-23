# Track Extrapolator Model Analysis Suite

This directory contains comprehensive analysis tools for evaluating trained track extrapolator neural networks.

## Overview

The analysis suite provides:

1. **Standard ML Metrics** - Position and slope errors, statistical analysis
2. **Physics-Aware Analysis** - Testing whether models learn correct physics
3. **Trajectory Visualization** - Visual comparison of predicted vs true tracks
4. **Benchmarking** - Timing comparisons with C++ extrapolators
5. **Experiment-Specific Analysis** - Dedicated sections for each experiment type

## Quick Start

### Main Analysis Notebook (Recommended)
```bash
# Open the comprehensive analysis notebook
jupyter notebook experiment_analysis.ipynb
```

The **experiment_analysis.ipynb** notebook includes sections for:
- Architecture Comparison (MLP/PINN/RK_PINN)
- Physics Loss Ablation (λ_PDE studies)
- Momentum Range Studies (Low/Mid/High-p)
- Learning Dynamics (loss curves, convergence)
- Timing Benchmarks
- Summary and Export

### Quick Analysis
```bash
python run_analysis.py --quick
```

### Full Analysis
```bash
python run_analysis.py --full
```

### Analyze Single Model
```bash
python run_analysis.py --model mlp_wide_v1
```

### Interactive Notebook
Open `model_analysis.ipynb` in Jupyter for interactive exploration.

## Analysis Components

### 1. `analyze_models.py` - Core Analysis
Main analyzer class providing:
- Model loading and inference
- Error computation (position, slope, percentiles)
- Trajectory comparison plots
- Residual distributions
- Momentum-dependent analysis
- Statistical summary tables

### 2. `physics_analysis.py` - Physics-Focused Analysis
Specialized physics tests:
- **Lorentz Force Analysis**: Does dtx/dz ∝ q/p? 
- **Charge Consistency**: Do opposite charges bend oppositely?
- **Systematic vs Random Errors**: Bias detection
- **Phase Space Coverage**: Error distribution across input space

### 3. `trajectory_visualizer.py` - Visualization
Advanced plotting tools:
- 3D trajectory visualization
- Trajectory gallery by momentum bin
- Error heatmaps
- Charge-separated trajectory comparison
- Prediction vs truth scatter plots

### 4. `model_analysis.ipynb` - Interactive Notebook
Jupyter notebook combining all analyses with explanations.

## Key Physics Tests

### Lorentz Force Test
In a magnetic field B, the Lorentz force causes:
- `d(tx)/dz ∝ q/p` (bending proportional to charge/momentum)
- `d(ty)/dz ≈ 0` (y-slope conserved for vertical B field)

A model that correctly learns physics should show:
- **Slope Ratio ≈ 1.0**: Linear relationship between dtx and q/p matches truth
- **High R²**: Strong correlation between dtx and q/p
- **Low σ(Δty)**: Y-slope is conserved

### Charge Consistency
Opposite charges should bend in opposite directions:
- Positive charges bend one way
- Negative charges bend the opposite way
- The magnitude of bending should be the same

## Model Architectures

The analysis suite evaluates three model types:

| Model | Description | Physics |
|-------|-------------|---------|
| **MLP** | Standard feedforward network | Data-driven only |
| **PINN** | Physics-informed with autodiff PDE loss | Lorentz force enforced |
| **RK-PINN** | Multi-stage with collocation points | Physics at intermediate z |

## Files

| File | Purpose |
|------|---------|
| `experiment_analysis.ipynb` | ⭐ Main analysis notebook (recommended) |
| `model_analysis.ipynb` | Interactive notebook |
| `run_analysis.py` | Main entry point |
| `analyze_models.py` | Core analysis functions |
| `physics_analysis.py` | Physics-specific tests |
| `trajectory_visualizer.py` | Visualization tools |
| `timing_benchmark.py` | ⭐ Comprehensive timing benchmarks |
| `timing_comparison_plots.py` | Timing visualizations |
| `generate_paper_quality_plots.py` | Publication-ready figures |
| `plots/` | Generated figures |
| `results/` | JSON results files |

## Metrics Computed

### Position Errors
- Mean absolute error (MAE) in x, y
- Root mean square error (RMSE)
- Percentiles (50th, 90th, 99th)

### Slope Errors  
- MAE in tx, ty
- RMSE
- Percentiles

### Physics Metrics
- Lorentz force slope ratio
- Charge asymmetry
- ty conservation

### Performance
- Inference time (ms per track)
- Throughput (tracks per second)
- Comparison with C++ (Herab, Runge-Kutta)

## Target Performance

- **Position Error:** < 10 μm
- **Slope Error:** < 10 μrad
- **Inference Speed:** > 100k tracks/second on GPU
