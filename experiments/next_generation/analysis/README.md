# Neural Network Track Extrapolator - Analysis Results

**Author:** G. Scriven  
**Date:** January 2026  
**Project:** LHCb Track Extrapolation with Physics-Informed Neural Networks

---

## Executive Summary

This analysis compares three neural network architectures (MLP, PINN, RK_PINN) for replacing the traditional C++ Runge-Kutta track extrapolator in the LHCb experiment. **All timings are measured against real C++ extrapolator benchmarks.**

| Metric | Best Model | Value |
|--------|-----------|-------|
| **Lowest validation loss** | `mlp_tiny` | 0.000781 |
| **Fastest inference** | `mlp_tiny` | **2.27× faster than C++ RK4** |
| **Best speed/accuracy** | `mlp_tiny` | 1.10 μs/track |

**Key Finding:** The fastest ML models (mlp_tiny, pinn_tiny) are **2.27× faster** than the accurate C++ reference extrapolator (CashKarp RK4), but only **0.73× as fast** as the inaccurate C++ Kisel extrapolator.

---

## 1. Timing Benchmark Results

### 1.1 C++ Extrapolator Reference Timings

Real measurements from `TrackExtrapolatorTesterSOA` benchmark:

| Extrapolator | Mean Time (μs) | Throughput | Position Error (mm) |
|--------------|---------------|------------|-------------------|
| **Kisel** | 1.50 | 667K/s | 39.82 (inaccurate!) |
| **Herab** | 1.95 | 513K/s | 5.09 |
| BogackiShampine3 | 2.40 | 417K/s | 0.04 |
| **Reference (CashKarp)** | **2.50** | **400K/s** | **0.00** |
| Verner9 | 2.52 | 397K/s | 0.00 |
| Verner7 | 2.65 | 377K/s | 0.00 |
| Tsitouras5 | 2.75 | 364K/s | 0.00 |

**Source:** `/experiments/next_generation/benchmarking/results/benchmark_results.json`

### 1.2 Speedup Calculation Methodology

#### Reference Baseline: Accurate C++ RK4 Extrapolator

- **Algorithm:** CashKarp adaptive Runge-Kutta integration
- **Reference timing:** **2.50 μs per track** (measured on same hardware)
- **Position error:** 0.00 mm (ground truth)

#### Speedup Formula

```
Speedup = C++_RK4_Time / NN_Time = 2.50 μs / NN_Time (μs)
```

**Interpretation:**
- Speedup > 1: NN is faster than accurate C++ RK4
- Speedup = 2.27: NN is 2.27× faster (mlp_tiny)
- Speedup < 1: NN is slower than C++ RK4

### 1.3 Complete Timing Results - All 60 Extrapolators

| Rank | Model | Type | Time (μs) | Throughput | Speedup |
|------|-------|------|-----------|------------|---------|
| 1 | mlp_tiny | CPU | 1.10 | 0.91M/s | **2.27×** |
| 2 | mlp_tiny_v1 | CPU | 1.10 | 0.91M/s | **2.27×** |
| 3 | pinn_tiny | CPU | 1.18 | 0.85M/s | **2.12×** |
| 4 | mlp_small | CPU | 1.22 | 0.82M/s | **2.06×** |
| 5 | mlp_small_v1 | CPU | 1.26 | 0.79M/s | **1.98×** |
| 6 | pinn_small | CPU | 1.31 | 0.76M/s | **1.91×** |
| 7 | mlp_wide_shallow_v1 | CPU | 1.41 | 0.71M/s | **1.77×** |
| 8 | **Kisel (C++)** | C++ | 1.50 | 0.67M/s | 1.67× |
| 9 | mlp_balanced_v1 | CPU | 1.82 | 0.55M/s | 1.37× |
| 10 | **Herab (C++)** | C++ | 1.95 | 0.51M/s | 1.28× |
| 11-29 | mlp/pinn medium variants | CPU | 2.0-2.2 | 0.45-0.50M/s | 1.1-1.2× |
| 30 | BogackiShampine3 (C++) | C++ | 2.40 | 0.42M/s | 1.04× |
| 31 | **Reference (C++)** | C++ | **2.50** | **0.40M/s** | **1.00×** |
| 32-34 | Other RK variants (C++) | C++ | 2.5-2.8 | 0.36-0.40M/s | 0.9-1.0× |
| 35-60 | rkpinn variants | CPU | 3.6-9.7 | 0.10-0.28M/s | 0.26-0.70× |

### 1.4 Key Observations

1. **7 ML models faster than ALL C++ extrapolators**: mlp_tiny, pinn_tiny, mlp_small, pinn_small, mlp_wide_shallow achieve this
2. **Best ML vs Best Accurate C++**: mlp_tiny (1.10 μs) vs Reference (2.50 μs) = **2.27× speedup**
3. **Best ML vs Fastest C++**: mlp_tiny (1.10 μs) vs Kisel (1.50 μs) = **0.73× as fast**, but Kisel has 39.82mm error!
4. **RK_PINN models are slower than C++ RK4**: 4-stage architecture overhead

### 2.1 Architecture Comparison (10 epochs training)

| Architecture | Mean Val Loss | Best Val Loss | Worst Val Loss |
|--------------|---------------|---------------|----------------|
| **MLP** | 0.00149 | 0.000781 | 0.00246 |
| **PINN** | 0.00452 | 0.00229 | 0.00728 |
| **RK_PINN** | 0.00372 | 0.00109 | 0.00567 |

**Winner at 10 epochs:** MLP (3× better than PINN)

### 2.2 Model Size Comparison

| Size | MLP Loss | PINN Loss | RK_PINN Loss |
|------|----------|-----------|--------------|
| Tiny | 0.000781 | 0.00401 | 0.00567 |
| Small | 0.00142 | 0.00450 | 0.00363 |
| Medium | 0.00153 | 0.00499 | 0.00449 |
| Wide | 0.00222 | 0.00456 | 0.00109 |

### 2.3 Physics Loss Ablation (λ_PDE effect)

| λ_PDE | PINN Loss | RK_PINN Loss |
|-------|-----------|--------------|
| 0.0 (data only) | 0.00286 | **0.00128** |
| 0.1 (weak) | 0.00380 | 0.00217 |
| 1.0 (balanced) | **0.00229** | 0.00449 |
| 10.0 (strong) | 0.00541 | 0.00385 |

**Findings:**
- PINN: Moderate physics loss (λ=1.0) is optimal
- RK_PINN: No physics loss (λ=0.0) performs best at 10 epochs

---

## 3. Convergence Analysis

### 3.1 Are models still improving at epoch 10?

| Model Type | Final Improvement Rate | Still Improving? |
|------------|----------------------|------------------|
| MLP | 32.8% per epoch | ✅ Yes |
| PINN | 33.1% per epoch | ✅ Yes |
| RK_PINN | 32.6% per epoch | ✅ Yes |

**Conclusion:** All architectures are improving at similar rates. 10 epochs is insufficient to determine long-term winner.

### 3.2 Predicted Asymptotic Loss (Exponential Decay Fit)

| Model | Current (10 ep) | Predicted L∞ | Potential Gain |
|-------|-----------------|--------------|----------------|
| MLP_medium | 0.00153 | 0.00044 | 71% |
| PINN_medium | 0.00499 | 0.00148 | 70% |
| RK_PINN_medium | 0.00449 | 0.00102 | 77% |

**Prediction:** MLP likely to remain best, but RK_PINN shows highest improvement potential.

### 3.3 Generalization Gap (Val/Train Ratio)

| Model Type | Gap Ratio | Interpretation |
|------------|-----------|----------------|
| MLP | 0.89 | Slight underfitting |
| PINN | 0.92 | Slight underfitting |
| RK_PINN | 0.85 | Slight underfitting |

**Note:** Gap < 1 indicates models are underfitting (more capacity available). Longer training recommended.

---

## 4. Speed vs Accuracy Trade-off

### 4.1 Pareto Analysis

The best models balance speed and accuracy:

| Model | Time (μs) | Speedup vs C++ | Val Loss | Pareto Optimal? |
|-------|-----------|----------------|----------|-----------------|
| `mlp_tiny` | 1.10 | **2.27×** | 0.000781 | ✅ **Yes** |
| `pinn_tiny` | 1.18 | **2.12×** | 0.00401 | ❌ No (MLP better accuracy) |
| `mlp_small` | 1.22 | **2.06×** | 0.00142 | ✅ Yes |
| `pinn_small` | 1.31 | **1.91×** | 0.00450 | ❌ No |
| C++ Kisel | 1.50 | 1.67× | 39.82mm err | ❌ No (inaccurate) |
| C++ Herab | 1.95 | 1.28× | 5.09mm err | ❌ No |
| C++ Reference | 2.50 | 1.00× | 0.00mm err | Baseline |
| `rkpinn_tiny` | 3.57 | 0.70× | 0.00567 | ❌ No (slower than C++) |

### 4.2 Recommendation by Use Case

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Production (speed + accuracy)** | `mlp_tiny` | 2.27× faster than C++, best accuracy |
| **Maximum accuracy** | `mlp_tiny` or `mlp_small` | Best loss values |
| **If C++ required** | `Reference (CashKarp)` | 2.50 μs/track, zero error |
| **Physics interpretability** | `pinn_medium` | Physics constraints, still faster than C++ |
| **Research/exploration** | Train longer | All models still converging |

---

## 5. Experimental Setup

### 5.1 Training Configuration

- **Dataset:** 50M track samples (RK4 ground truth)
- **Input features:** [x, y, tx, ty, q/p, dz] (6 dimensions)
- **Output features:** [x_f, y_f, tx_f, ty_f] (4 dimensions)
- **Training:** 10 epochs on HTCondor GPU cluster
- **Optimizer:** Adam with cosine annealing LR schedule
- **Batch size:** 4096

### 5.2 Architecture Sizes

| Size | Hidden Layers | Parameters (MLP) |
|------|---------------|------------------|
| Tiny | [64, 64] | ~5k |
| Small | [128, 128] | ~17k |
| Medium | [256, 256, 128] | ~100k |
| Wide | [512, 512, 256, 128] | ~400k |

### 5.3 Experiments Run (29 total)

1. **Core experiments (12):** MLP/PINN/RK_PINN × tiny/small/medium/wide
2. **Physics ablation (8):** λ_PDE ∈ {0, 0.1, 1, 10} for PINN and RK_PINN
3. **Momentum studies (9):** Low/Mid/High momentum ranges × MLP/PINN/RK_PINN

---

## 6. Files and Outputs

### 6.1 Generated Plots

| File | Description |
|------|-------------|
| `plots/architecture_comparison.png` | Val loss by model type + Pareto frontier |
| `plots/size_comparison.png` | Loss vs architecture size |
| `plots/physics_ablation.png` | Effect of λ_PDE |
| `plots/momentum_comparison.png` | Performance by momentum range |
| `plots/loss_curves_architecture.png` | Training/validation curves |
| `plots/convergence_analysis.png` | Epochs vs final loss |
| `plots/timing_benchmarks.png` | **Speedup and throughput plots** |
| `plots/physics_loss_decomposition.png` | Data vs physics loss evolution |
| `plots/convergence_extrapolation.png` | Predicted asymptotic loss |
| `plots/generalization_gap.png` | Train/val gap analysis |

### 6.2 Exported Data

| File | Description |
|------|-------------|
| `results/all_model_results.csv` | Full results table for all 43 models |
| `results/top_models_table.tex` | LaTeX table of top 10 models |

---

## 7. Conclusions

### 7.1 Main Findings

1. **Neural networks are 2.27× faster** than accurate C++ Runge-Kutta extrapolators
2. **MLP outperforms PINN/RK_PINN** at 10 epochs training
3. **Physics loss doesn't help accuracy** with sufficient training data (50M samples)
4. **Tiny models are best**: Smallest models have best accuracy AND speed
5. **More training needed**: All models still improving; 50-100 epochs recommended

### 7.2 Recommendations

1. **Deploy `mlp_tiny`** for production: 2.27× speedup with best accuracy
2. **Investigate longer training** for PINN/RK_PINN to test physics benefits
3. **Test on held-out data** to verify generalization claims
4. **Benchmark on production hardware** to validate speedup claims

### 7.3 Future Work

- [ ] Train for 100 epochs to verify convergence predictions
- [ ] Test on out-of-distribution tracks (edge cases)
- [ ] Implement ONNX export for C++ deployment
- [ ] Benchmark with realistic field map lookups
- [ ] Compare against other LHCb extrapolators (parabolic, linear)

---

## Appendix A: Speedup Calculation Details

### Mathematical Formulation

For a batch of N tracks:

```
T_batch = (1/n_runs) × Σ t_i           # Mean batch time over n_runs
T_per_track = T_batch / N × 10^6       # Per-track time in μs
Throughput = N / T_batch               # Tracks per second
Speedup = 2.50 μs / T_per_track        # Relative to C++ RK4
```

### Example Calculation

For `mlp_tiny` with batch of 10,000 tracks:
- Mean batch time: 11.0 ms
- Per-track time: 11.0 ms / 10,000 = 1.10 μs
- Throughput: 10,000 / 0.0110 s = 909,000 tracks/s = 0.91 M/s
- Speedup: 2.50 μs / 1.10 μs = **2.27×**

### Reference C++ Timing Source

The 2.50 μs reference is measured from:
- `TrackExtrapolatorTesterSOA` benchmark in LHCb Rec framework
- CashKarp RK4 extrapolator (accurate reference)
- Measured on same hardware as ML model benchmarks
- Full magnetic field interpolation from LHCb dipole field map

---

## Appendix B: Analysis Tools

### Main Analysis Notebook
```bash
jupyter notebook experiment_analysis.ipynb
```

### Quick Analysis
```bash
python run_analysis.py --quick
```

### Full Analysis
```bash
python run_analysis.py --full
```

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

---

## V2 Analysis Workflow

V2 training (22 shallow-wide MLP models) is currently running on condor.

### Check Training Status
```bash
condor_q
```

### Analyze V2 When Complete
```bash
python analyze_v2_results.py
```

### Full Comparison with V1
Open `experiment_analysis.ipynb` and re-run all cells after V2 completes.

---

## New Files (January 2026)

| File | Description |
|------|-------------|
| `analyze_v2_results.py` | V2 shallow-wide model analysis script |
| `V1_findings_summary.ipynb` | Executive summary of V1 training results |
