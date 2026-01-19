# LHCb Track Extrapolators

Neural network-based track extrapolation for the LHCb experiment, replacing traditional numerical integration methods.

---

## 🎯 Project Overview

**Goal:** Develop fast, accurate ML-based track extrapolators for LHCb reconstruction

**Current Status:** ✅ Proof of concept complete
- **Best Model:** MLP with SiLU activation
- **Performance:** 0.21 mm mean error (vs 0.63 mm tanh, 0.77 mm ReLU)
- **Speed:** ~160× faster than Runge-Kutta (estimated)
- **Physics:** Pure data-driven learning outperforms physics-informed approaches

**Next Steps:** Scale to full detector coverage, optimize for production

---

## 📂 Repository Structure

```
TrackExtrapolators/
│
├── README.md                           # This file - project overview
├── model_investigation.ipynb           # 📊 Main analysis notebook (START HERE)
├── analyze_extrapolators.ipynb         # Legacy analysis
├── extrapolator_results.csv            # Benchmark results
├── full_domain_benchmark.ipynb         # Full coverage analysis
│
├── ml_models/                          # 🧠 Machine Learning Pipeline
│   ├── README.md                       # ML documentation (usage, training)
│   ├── data/                           # Training datasets
│   │   ├── X_analysis.npy              # Input: 50K samples (6D states @ z=4000mm)
│   │   ├── Y_analysis.npy              # Output: extrapolated states @ z=12000mm
│   │   ├── P_analysis.npy              # Momentum for each track
│   │   └── X_weighted_train.npy        # Weighted training data (legacy)
│   │
│   ├── models/                         # Trained models
│   │   ├── analysis/                   # Analysis models (latest)
│   │   │   ├── mlp_act_silu.pt         # ⭐ BEST: 0.21mm error
│   │   │   ├── mlp_act_tanh.pt         # Baseline: 0.63mm
│   │   │   ├── mlp_act_relu.pt         # 0.77mm
│   │   │   ├── pinn_lambda_0_01.pt     # Physics-informed: 18.8mm (failed)
│   │   │   ├── pinn_lambda_0_05.pt     # 106mm (failed)
│   │   │   ├── pinn_lambda_0_1.pt      # 197mm (failed)
│   │   │   └── pinn_lambda_0_2.pt      # 329mm (failed badly)
│   │   └── production/                 # Production models (HTCondor training)
│   │       └── (empty - jobs pending)
│   │
│   ├── python/                         # Training and data generation
│   │   ├── generate_training_data.py   # Fast parallel data generation
│   │   ├── train_analysis_models.py    # Train all analysis variants
│   │   ├── train_on_gpu.py             # GPU-accelerated training
│   │   ├── full_domain_training.py     # Full momentum range
│   │   ├── train_pinn.py               # Physics-informed NN (deprecated)
│   │   └── compare_models.py           # Model comparison utilities
│   │
│   ├── condor/                         # HTCondor cluster jobs
│   │   ├── README.md                   # Cluster usage guide
│   │   ├── train_production.sub        # 12 production models
│   │   ├── train_analysis.sub          # 12 analysis models
│   │   ├── generate_data.sub           # Parallel data generation
│   │   └── logs/                       # Job outputs
│   │
│   └── src/                            # C++ implementations
│       └── TrackMLPExtrapolator.cpp    # LHCb integration (prototype)
│
├── experiments/                        # 🔬 Experiment Archive
│   ├── README.md                       # Experiment tracking
│   ├── experiment_log.csv              # All experiments (dates, configs, results)
│   │
│   ├── baseline/                       # Initial experiments
│   │   ├── v1_positive_qop/            # Single charge training
│   │   └── v2_both_charges/            # Full charge spectrum
│   │
│   ├── architecture/                   # Network architecture studies
│   │   ├── deeper_networks/            # Depth experiments
│   │   ├── wider_networks/             # Width experiments
│   │   └── skip_connections/           # ResNet-style connections
│   │
│   ├── momentum_studies/               # Momentum range experiments
│   │   ├── low_p_05_2gev/              # Low momentum (challenging)
│   │   ├── mid_p_2_10gev/              # Medium momentum
│   │   └── high_p_10_100gev/           # High momentum (easier)
│   │
│   ├── physics_informed/               # PINN experiments
│   │   ├── energy_conservation/        # Energy loss constraints
│   │   └── lorentz_loss/               # Lorentz force penalties
│   │
│   ├── data_augmentation/              # Data sampling strategies
│   │   ├── dense_grid/                 # Uniform sampling
│   │   └── random_sampling/            # Random track generation
│   │
│   ├── field_maps/                     # Magnetic field studies
│   │   ├── simplified/                 # Simplified B-field
│   │   └── simcond/                    # Full simulation conditions
│   │
│   ├── weighted_loss/                  # Loss function experiments
│   │   ├── README.md
│   │   ├── train_weighted.py
│   │   └── training_log.txt
│   │
│   ├── onnx_export/                    # Model export (for deployment)
│   │   ├── README.md
│   │   ├── export_onnx.py
│   │   ├── mlp_full_domain.onnx
│   │   └── pinn_full_domain.onnx
│   │
│   └── production/                     # Production-ready models
│       └── best_model/                 # Finalized model for deployment
│
├── src/                                # 🔧 Traditional C++ Extrapolators (Reference)
│   ├── TrackRungeKuttaExtrapolator.cpp    # Gold standard (slow but accurate)
│   ├── TrackKiselExtrapolator.cpp         # Fast numerical method
│   ├── TrackHerabExtrapolator.cpp         # Alternative fast method
│   ├── TrackLinearExtrapolator.cpp        # Simplest approximation
│   ├── TrackParabolicExtrapolator.cpp     # Second-order approximation
│   ├── TrackFieldExtrapolatorBase.cpp     # Base class
│   └── ...
│
├── tests/                              # Test configurations
│   ├── options/
│   ├── qmtest/
│   └── refs/
│
├── report/                             # 📄 Documentation
│   ├── pinn_track_extrapolation_report.tex
│   └── pinn_track_extrapolation_report.pdf
│
├── plots/                              # Generated figures
├── doc/                                # Release notes
├── lhcb-metainfo/                      # LHCb metadata
└── CMakeLists.txt                      # Build configuration
```

---

## 🚀 Quick Start

### 1. Explore Results (Recommended First Step)

Open the main analysis notebook:
```bash
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators
jupyter notebook model_investigation.ipynb
```

**What you'll find:**
- ✅ Model performance comparison (activation functions, PINN analysis)
- ✅ Physics constraint analysis using autograd
- ✅ Feature sensitivity and gradient analysis
- ✅ Why physics-informed models failed (detailed diagnosis)
- ✅ Decision boundaries and non-linearity visualization

### 2. Train Models Locally

```bash
cd ml_models/python

# Generate training data (fast parallel version)
python generate_training_data.py --samples 50000 --output ../data/ --name analysis

# Train analysis models (all variants)
python train_analysis_models.py

# Or train single model on GPU
python train_on_gpu.py --hidden 256 256 128 64 --epochs 2000 --name custom_model
```

### 3. Submit Cluster Jobs (HTCondor)

```bash
cd ml_models/condor

# Generate large dataset (parallel)
condor_submit generate_data.sub

# Train production models (12 architectures)
condor_submit train_production.sub

# Monitor jobs
condor_q

# Check outputs
tail -f logs/prod_*.out
```

See [`ml_models/condor/README.md`](ml_models/condor/README.md) for detailed cluster usage.

---

## 📊 Key Results Summary

### Current Best Model: **MLP with SiLU Activation**

| Metric | Value |
|--------|-------|
| **Mean Error** | 0.21 mm |
| **Median Error** | 0.15 mm |
| **P95 Error** | 0.54 mm |
| **Max Error** | 4.13 mm |
| **Architecture** | 128-128-64 (3 hidden layers) |
| **Parameters** | 25,924 |
| **Activation** | SiLU (Swish) |

### Activation Function Comparison

| Activation | Mean Error | P95 Error | Speedup vs Tanh |
|------------|------------|-----------|-----------------|
| **SiLU** ⭐ | 0.21 mm | 0.54 mm | - |
| Tanh | 0.63 mm | 1.68 mm | 3× worse |
| ReLU | 0.77 mm | 1.78 mm | 3.7× worse |

### PINN Failure Analysis

All physics-informed models **failed dramatically**:

| PINN λ | Mean Error | Why It Failed |
|--------|------------|---------------|
| 0.01 | 18.8 mm | Wrong physics formulation |
| 0.05 | 106.7 mm | Conflicting constraints |
| 0.1 | 197.2 mm | No proper Lorentz integration |
| 0.2 | 328.9 mm | Higher λ → worse (contradicts true physics!) |

**Root cause:** Physics loss uses oversimplified straight-line approximation instead of proper magnetic field ODE integration. See detailed analysis in [`model_investigation.ipynb`](model_investigation.ipynb).

---

## 🧠 Network Architecture

### Input (6D) @ z = 4000 mm
- `x, y` - Position [mm]
- `tx, ty` - Slopes dx/dz, dy/dz [dimensionless]
- `q/p` - Signed inverse momentum [GeV⁻¹]
- `z` - Longitudinal position [mm] (currently fixed)

### Output (4D) @ z = 12000 mm
- `x', y'` - Extrapolated position [mm]
- `tx', ty'` - Extrapolated slopes

**What the network learns:** The magnetic field propagator over Δz = 8000 mm, including non-uniform B(x,y,z), Lorentz force bending, and geometric effects—all implicitly from data!

---

## 📈 Next Steps

### Immediate (Production Ready)
1. ✅ **Complete** - Identify best architecture (SiLU activation)
2. ⏳ **In Progress** - Train production models on full dataset (HTCondor)
3. 🔜 **Next** - Export to ONNX for C++ inference
4. 🔜 **Next** - Integrate into LHCb framework
5. 🔜 **Next** - Benchmark against Runge-Kutta on real events

### Medium Term (Improved Physics)
- Redesign PINN with proper Lorentz force ODE integration
- Incorporate actual LHCb B(x,y,z) field map
- Add material multiple scattering effects
- Physics-inspired feature engineering (momentum-dependent weights)

### Long Term (Advanced Methods)
- Transformer architecture for multi-step extrapolation
- Neural ODEs for continuous-time modeling
- Uncertainty quantification (Bayesian NNs)
- Active learning for data efficiency

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| [`model_investigation.ipynb`](model_investigation.ipynb) | **Main analysis** - comprehensive model study |
| [`ml_models/README.md`](ml_models/README.md) | ML pipeline documentation |
| [`ml_models/condor/README.md`](ml_models/condor/README.md) | Cluster job submission guide |
| [`experiments/README.md`](experiments/README.md) | Experiment tracking and history |
| `extrapolator_results.csv` | Benchmark comparison (ML vs traditional) |

---

## 🔧 Technical Details

### Data Generation
- **Source:** LHCb Gaudi framework (TrackRungeKuttaExtrapolator)
- **Sampling:** Random uniform in phase space
- **Coverage:** 0.5-100 GeV/c momentum, ±1000 mm position, ±0.3 slope
- **Size:** 50K samples for analysis, scaling to 1M+ for production

### Training
- **Framework:** PyTorch 2.9.1 + CUDA
- **Optimizer:** AdamW with weight decay
- **Scheduler:** ReduceLROnPlateau
- **Loss:** MSE on position + slopes (+ physics loss for PINN)
- **Hardware:** NVIDIA L40S GPUs (45GB VRAM, Capability 8.9)
- **Time:** ~2-5 min per model on GPU

### Evaluation
- **Metric:** Position error = √((x_pred - x_true)² + (y_pred - y_true)²)
- **Validation:** 20% held-out test set
- **Physics checks:** Slope ratios, bending consistency, momentum dependence

---

## 👥 Contributors

- George Scriven (gscriven@nikhef.nl)
- LHCb Reconstruction Group

---

## 📚 References

- LHCb Track Reconstruction: [LHCb-2007-007](https://cds.cern.ch/record/1033584)
- Physics-Informed Neural Networks: [Raissi et al. 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Neural ODEs: [Chen et al. 2018](https://arxiv.org/abs/1806.07366)

---

## 📄 License

LHCb Software - see LHCb collaboration policies

---

**Last Updated:** January 2025  
**Version:** 1.0 (Analysis Phase Complete)
