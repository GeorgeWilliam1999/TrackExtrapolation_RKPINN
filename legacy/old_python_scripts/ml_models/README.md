# ML-Based Track Extrapolators

Machine learning models that replace traditional numerical integration for LHCb track extrapolation.

---

## 📊 Current Status

### ✅ Available Models (7 total)

All models trained on 50K samples, tested on 10K held-out validation set.

#### Activation Function Study (3 models)
| Model | Activation | Mean Error | P95 Error | Status |
|-------|------------|------------|-----------|--------|
| `mlp_act_silu.pt` | SiLU ⭐ | **0.21 mm** | **0.54 mm** | **BEST** |
| `mlp_act_tanh.pt` | Tanh | 0.63 mm | 1.68 mm | Baseline |
| `mlp_act_relu.pt` | ReLU | 0.77 mm | 1.78 mm | Good |

**Recommendation:** Use SiLU activation for all future models.

#### Physics-Informed Models (4 models)
| Model | Lambda (λ) | Mean Error | P95 Error | Status |
|-------|------------|------------|-----------|--------|
| `pinn_lambda_0_01.pt` | 0.01 | 18.8 mm | 47.2 mm | ❌ Failed |
| `pinn_lambda_0_05.pt` | 0.05 | 106.7 mm | 273.8 mm | ❌ Failed |
| `pinn_lambda_0_1.pt` | 0.1 | 197.2 mm | 506.7 mm | ❌ Failed |
| `pinn_lambda_0_2.pt` | 0.2 | 328.9 mm | 843.9 mm | ❌ Failed |

**Conclusion:** Current PINN formulation does not work. Physics loss conflicts with true magnetic field dynamics. See [model_investigation.ipynb](../model_investigation.ipynb) for detailed analysis.

### ⏳ Missing Models (Architecture Study)

These were intended to be trained but HTCondor jobs appear incomplete:
- `mlp_tiny.pt` - 64-32 architecture
- `mlp_small.pt` - 128-64 architecture  
- `mlp_medium.pt` - 128-128-64 (default)
- `mlp_large.pt` - 256-256-128-64
- `mlp_xlarge.pt` - 512-512-256-128

**Next Step:** Re-run architecture training jobs on cluster.

---

## 📂 Directory Structure

```
ml_models/
│
├── README.md                           # This file
│
├── data/                               # Training datasets
│   ├── X_analysis.npy                  # 50K training samples (6D input @ z=4000mm)
│   ├── Y_analysis.npy                  # 50K targets (4D output @ z=12000mm)
│   ├── P_analysis.npy                  # Momentum for each track
│   ├── X_weighted_train.npy            # Legacy weighted data
│   ├── Y_weighted_train.npy
│   └── P_weighted_train.npy
│
├── models/                             # Trained models
│   ├── analysis/                       # Latest analysis models
│   │   ├── mlp_act_silu.pt             # Best model (0.21mm)
│   │   ├── mlp_act_tanh.pt
│   │   ├── mlp_act_relu.pt
│   │   ├── pinn_lambda_0_01.pt
│   │   ├── pinn_lambda_0_05.pt
│   │   ├── pinn_lambda_0_1.pt
│   │   ├── pinn_lambda_0_2.pt
│   │   └── *_metadata.json             # Training metadata for each model
│   │
│   ├── production/                     # Production models (for deployment)
│   │   └── (empty - pending training)
│   │
│   ├── mlp_model_cpp_v2.bin            # Legacy C++ format
│   ├── pinn_model_true.bin
│   ├── config.json                     # Model configuration
│   └── full_domain_results.json        # Benchmark results
│
├── python/                             # Training scripts
│   ├── generate_training_data.py       # ⭐ Fast parallel data generation
│   ├── train_analysis_models.py        # Train all analysis variants
│   ├── train_on_gpu.py                 # General GPU training
│   ├── full_domain_training.py         # Full momentum range
│   ├── train_pinn.py                   # PINN training (deprecated)
│   ├── train_true_pinn.py              # True PINN (deprecated)
│   ├── compare_models.py               # Model comparison
│   └── test_pinn_simple.py             # Simple test script
│
├── condor/                             # HTCondor cluster jobs
│   ├── README.md                       # Cluster usage guide
│   ├── train_production.sub            # Production model training
│   ├── train_analysis.sub              # Analysis model training
│   ├── generate_data.sub               # Parallel data generation
│   ├── *.sh                            # Job scripts
│   └── logs/                           # Job outputs
│
├── src/                                # C++ implementation
│   └── TrackMLPExtrapolator.cpp        # LHCb integration (prototype)
│
└── docs/                               # Additional documentation
```

---

## 🚀 Quick Start

### 1. Load and Use Existing Models

See [model_investigation.ipynb](../model_investigation.ipynb) for comprehensive analysis of all available models.

```python
import torch

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('models/analysis/mlp_act_silu.pt', map_location=device)
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)  # input: [N, 6], output: [N, 4]
```

### 2. Generate Training Data

Fast parallel data generation using the Runge-Kutta extrapolator:

```bash
cd python

# Generate 50K samples for analysis (takes ~5-10 min)
python generate_training_data.py \
    --samples 50000 \
    --output ../data/ \
    --name analysis

# Generate larger dataset for production (takes ~30 min for 500K)
python generate_training_data.py \
    --samples 500000 \
    --output ../data/ \
    --name production
```

**Output:**
- `X_analysis.npy` - Input states (N × 6: x, y, tx, ty, q/p, z)
- `Y_analysis.npy` - Target states (N × 4: x', y', tx', ty')
- `P_analysis.npy` - Momentum values (N)

### 3. Train Models Locally

#### Train all analysis variants (recommended for experiments)
```bash
cd python
python train_analysis_models.py
```

This trains:
- 5 architecture variants (tiny, small, medium, large, xlarge)
- 3 activation variants (tanh, relu, silu)
- 4 PINN variants (λ = 0.01, 0.05, 0.1, 0.2)

**Time:** ~20-30 min on GPU for all 12 models

#### Train single custom model
```bash
python train_on_gpu.py \
    --hidden 256 256 128 64 \
    --activation silu \
    --epochs 2000 \
    --lr 0.001 \
    --name my_custom_model
```

### 4. Submit Cluster Jobs (HTCondor)

For large-scale training on GPU cluster:

```bash
cd condor

# Generate large training dataset (parallel jobs)
condor_submit generate_data.sub

# Train all production models
condor_submit train_production.sub

# Monitor jobs
condor_q

# Check specific job output
tail -f logs/prod_medium_*.out
```

See [condor/README.md](condor/README.md) for detailed cluster documentation.

---

## 🧠 Model Architecture

### Network Structure

**TrackMLP** (Data-Driven)
```
Input (6D):  x, y, tx, ty, q/p, z
              ↓
Hidden:      [128] → [128] → [64]  (configurable)
Activation:  SiLU (best), tanh, ReLU
              ↓
Output (4D): x', y', tx', ty'
```

**TrackPINN** (Physics-Informed - deprecated)
```
Same architecture as MLP but with additional physics loss:
  L_total = L_MSE + λ × L_physics
  
  L_physics = L_position + L_bending + L_ty_penalty
```

### Input Features (6D @ z = 4000 mm)
1. **x** [mm] - Horizontal position
2. **y** [mm] - Vertical position
3. **tx** - Horizontal slope (dx/dz)
4. **ty** - Vertical slope (dy/dz)
5. **q/p** [GeV⁻¹] - Signed inverse momentum
6. **z** [mm] - Longitudinal position (currently fixed at 4000)

### Output Features (4D @ z = 12000 mm)
1. **x'** [mm] - Extrapolated horizontal position
2. **y'** [mm] - Extrapolated vertical position
3. **tx'** - Extrapolated horizontal slope
4. **ty'** - Extrapolated vertical slope

### What the Network Learns

The network implicitly learns:
- ✅ Non-uniform magnetic field B(x, y, z) effects
- ✅ Lorentz force curvature: F = q(v × B)
- ✅ Momentum-dependent bending
- ✅ Geometric path length corrections
- ✅ Charge-dependent deflection (+/- bending)

All without explicit physics equations - just from 50K examples!

---

## 📈 Training Details

### Data Generation
- **Source:** LHCb Gaudi framework (`TrackRungeKuttaExtrapolator`)
- **Method:** Parallel processes (multiprocessing)
- **Coverage:** Full phase space
  - Momentum: 0.5 - 100 GeV/c
  - Position: ±1000 mm (x, y)
  - Slopes: ±0.3 (tx, ty)
  - Charge: Both +/- particles
- **Propagation:** z = 4000 mm → 12000 mm (Δz = 8000 mm)

### Training Configuration
- **Framework:** PyTorch 2.9.1 + CUDA 12.8
- **Loss:** Mean Squared Error (MSE)
  ```python
  loss = F.mse_loss(predictions, targets)
  ```
- **Optimizer:** AdamW
  - Learning rate: 0.001 (initial)
  - Weight decay: 1e-5
- **Scheduler:** ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 50 epochs
  - Min LR: 1e-6
- **Batch size:** 1024
- **Epochs:** 2000 (with early stopping)
- **Hardware:** NVIDIA L40S GPU (45GB VRAM)
- **Time:** 2-5 minutes per model on GPU

### Evaluation Metrics
- **Position Error:** $\text{err} = \sqrt{(x' - x_{true})^2 + (y' - y_{true})^2}$
- **Slope Error:** $\|\Delta \mathbf{t}\| = \sqrt{(\Delta t_x)^2 + (\Delta t_y)^2}$
- **Statistics:** Mean, Median, P95, Max

---

## ⚠️ Known Issues

### 1. PINN Models Failed
**Problem:** All PINN variants have 10-1000× worse error than baseline MLP.

**Root Cause:** Physics loss formulation is oversimplified
- Assumes straight-line propagation between collocation points
- Doesn't properly integrate Lorentz force ODE
- Conflicts with true magnetic field dynamics

**Evidence:** See detailed autograd analysis in [model_investigation.ipynb](../model_investigation.ipynb)

**Solution:** Either:
- Abandon PINNs (current recommendation - data-driven works!)
- Redesign with proper ODE integration using Neural ODEs
- Add collocation points along actual trajectory

### 2. Architecture Models Missing
**Problem:** Only activation study models exist, no architecture variants.

**Cause:** HTCondor jobs may have failed or not completed.

**Solution:** Re-run `condor/train_analysis.sub` or train locally with `python/train_analysis_models.py`

### 3. Limited Momentum Range
**Current:** 0.5 - 100 GeV/c  
**LHCb Full Range:** 0.5 - 200 GeV/c

**Next Step:** Generate more data at high momentum and retrain.

---

## 🎯 Next Steps

### Immediate
1. ✅ **Complete** - Identify best model (SiLU activation, 0.21mm)
2. 🔜 **Next** - Train architecture variants (tiny → xlarge)
3. 🔜 **Next** - Extend to full momentum range (up to 200 GeV/c)
4. 🔜 **Next** - Export to ONNX for C++ deployment

### Medium Term
- Generate 1M+ training samples for production
- Multi-step extrapolation (multiple z planes)
- Uncertainty quantification (Bayesian dropout)
- Integration into LHCb Gaudi framework

### Long Term
- Neural ODEs for continuous-time modeling
- Transformer architecture for sequence modeling
- Proper PINN with ODE-integrated physics loss
- Active learning for data efficiency

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `python/generate_training_data.py` | Fast parallel data generation from RK4 |
| `python/train_analysis_models.py` | Train all model variants |
| `python/train_on_gpu.py` | General GPU training script |
| `models/analysis/mlp_act_silu.pt` | **Best model** - use this! |
| `condor/README.md` | HTCondor cluster usage guide |
| `../model_investigation.ipynb` | **Main analysis notebook** |

---

## 🤝 Contributing

When adding new models:
1. Save in `.pt` format with full state dict
2. Include `_metadata.json` with training config
3. Update this README with results
4. Add entry to `experiment_log.csv` in `../experiments/`

---

**Last Updated:** January 2025  
**Model Version:** v1.0 (Analysis Complete)
