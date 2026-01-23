# TrackExtrapolators - Neural Network Track Extrapolation

**Clean repository for next-generation track extrapolation experiments**

Reorganized: January 14, 2025  
**Major Update:** January 22, 2026  
Previous work archived in: `legacy/`

---

## 🎯 Project Status

**Current Phase:** ✅ **Model Training in Progress**  
**Goal:** Train physics-informed neural network track extrapolators for LHCb

**Completed:**
- ✅ LHCb software stack properly configured (DetDesc mode)
- ✅ C++ extrapolator tests running successfully  
- ✅ All 9 extrapolators benchmarked across 1210 track states
- ✅ 50M track training dataset generated (`training_50M.npz`)
- ✅ Momentum-split datasets: low/mid/high-p (10M each)
- ✅ MLP, PINN, RK_PINN architectures implemented
- ✅ Real field map integration (`InterpolatedFieldTorch`)
- ✅ PINN training stability fixes applied (see `notes/PINN_STABILITY_FIXES.md`)
- ✅ 30 HTCondor training jobs submitted (cluster 3880818)

**Current Work:**
- 🔄 Training 10 MLP variants (architecture sweep)
- 🔄 Training 10 PINN variants (λ_pde sweep: 1e-5 to 0.1)
- 🔄 Training 10 RK-PINN variants (collocation point sweep)

**See Active Development:** [experiments/next_generation/README.md](experiments/next_generation/README.md)

---

## 📂 Repository Structure

```
TrackExtrapolators/
├── README.md                          # This file
├── CMakeLists.txt                     # C++ build system
│
├── src/                               # C++ Production Code (LHCb framework)
│   ├── TrackRungeKuttaExtrapolator.cpp    # RK4 baseline (to benchmark)
│   ├── TrackKiselExtrapolator.cpp         # Fast analytic method
│   ├── TrackSTEPExtrapolator.cpp          # Reference (highest accuracy)
│   └── TrackExtrapolatorTesterSOA.cpp     # Full benchmark with timing
│
├── tests/                             # LHCb framework tests
│   ├── options/                       # Gaudi configuration files
│   └── qmtest/                        # LHCb test descriptors
│
├── experiments/
│   ├── next_generation/               # 🔥 ACTIVE DEVELOPMENT
│   │   ├── README.md                  # Project status & quick start
│   │   ├── run_all_experiments.py     # Unified experiment runner (NEW)
│   │   │
│   │   ├── models/                    # Model training
│   │   │   ├── train.py               # Main training script
│   │   │   ├── architectures.py       # MLP, PINN, RK_PINN
│   │   │   ├── evaluate.py            # Model evaluation
│   │   │   └── checkpoints/           # Trained models
│   │   │
│   │   ├── data_generation/           # Training data
│   │   │   ├── data/                  # Dataset files
│   │   │   │   ├── training_50M.npz   # 50M tracks (3.7GB)
│   │   │   │   ├── training_low_p.npz # p < 5 GeV
│   │   │   │   ├── training_mid_p.npz # 5 ≤ p < 20 GeV
│   │   │   │   └── training_high_p.npz # p ≥ 20 GeV
│   │   │   └── generate_data.py       # Data generation script
│   │   │
│   │   ├── training/                  # HTCondor job files
│   │   │   └── jobs/                  # .sub files for all experiments
│   │   │
│   │   ├── analysis/                  # Results analysis
│   │   │   └── experiment_analysis.ipynb  # Comprehensive analysis (NEW)
│   │   │
│   │   ├── utils/                     # Utilities
│   │   │   └── magnetic_field.py      # Field map interpolation
│   │   │
│   │   └── notes/                     # Documentation
│   │       └── experimental_protocol.pdf  # Full experiment design
│   │
│   ├── experiment_log.csv             # Experiment tracking
│   └── README.md                      # Historical experiment summary
│
└── legacy/                            # 📦 ARCHIVED (previous work)
    ├── old_notebooks/                 # Analysis notebooks
    ├── old_experiments/               # All previous experiments
    └── OLD_README.md                  # Previous README
```

---

## 🚀 Quick Start

### For Active Development

The main development is in `experiments/next_generation/`. See that README for details.

```bash
cd experiments/next_generation

# Check HTCondor job status
condor_q gscriven

# Run all experiments (local or HTCondor)
python run_all_experiments.py --list        # List available experiments
python run_all_experiments.py --all         # Submit all to HTCondor
python run_all_experiments.py --local       # Run locally (interactive)

# Analyze results after training completes
jupyter notebook analysis/experiment_analysis.ipynb
```

### Training a Single Model

```bash
cd experiments/next_generation/models
conda activate TE

# Train MLP baseline
python train.py --model mlp --preset medium --epochs 100

# Train PINN with physics loss
python train.py --model pinn --preset medium --lambda-pde 1.0 --epochs 100

# Train RK-PINN
python train.py --model rk_pinn --preset medium --epochs 100
```

### Running C++ Tests

This project uses the LHCb software stack. Prerequisites:
- Access to CVMFS (e.g., Nikhef STBC cluster)
- LHCb stack built with DetDesc geometry backend
- Environment: `x86_64_v2-el9-gcc13+detdesc-opt`

```bash
cd /data/bfys/gscriven/TE_stack
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/qmtest/test_extrapolators.qmt
```

---

## 📊 Model Architectures

| Model | Physics | Description |
|-------|---------|-------------|
| **MLP** | Implicit (data) | Standard feedforward, fastest inference |
| **PINN** | Explicit (PDE) | Physics-informed with Lorentz force |
| **RK_PINN** | Explicit (staged) | RK4-inspired multi-stage structure |

**Presets:** `tiny` (5k), `small` (20k), `medium` (100k), `wide` (500k params)

---

## 📊 Historical Results (Legacy)

Previous experiments (in `legacy/`) achieved:

| Model | Activation | Mean Error | Dataset | Notes |
|-------|------------|------------|---------|-------|
| MLP (SiLU) | SiLU | **0.21 mm** | 50K tracks | Best from legacy |
| MLP (Tanh) | Tanh | 0.63 mm | 50K tracks | Baseline |
| PINN | Various | 18-329 mm | 50K tracks | ❌ Failed (wrong field) |

**Note:** Legacy PINN failures were due to using Gaussian field approximation instead of the real
interpolated field map. Current PINN/RK_PINN models use `InterpolatedFieldTorch`.

---

## 🔬 Current Work (January 2026)

### Training Experiments in Progress

**30 HTCondor GPU jobs submitted (cluster 3880818):**

1. **MLP Architecture Sweep** (10 experiments):
   - Presets: tiny, small, medium, large, xlarge, wide, deep
   - Custom: narrow_deep, wide_shallow, balanced

2. **PINN Physics Loss Sweep** (10 experiments):
   - λ_pde values: 1e-5, 1e-4, 1e-3, 1e-2, 0.1
   - Sizes: medium, large, xlarge, wide, deep

3. **RK-PINN Collocation Sweep** (10 experiments):
   - Collocation points: 5, 10, 15, 20
   - Sizes: medium, large, xlarge, wide, deep

### Key Files

| File | Purpose |
|------|---------|
| `experiments/next_generation/run_all_experiments.py` | Unified experiment runner |
| `experiments/next_generation/analysis/experiment_analysis.ipynb` | Results analysis |
| `experiments/next_generation/notes/experimental_protocol.pdf` | Full methodology |
| `experiments/next_generation/models/train.py` | Main training script |

### Design Documents

See `experiments/next_generation/` for:
- [README.md](experiments/next_generation/README.md) - Project status
- [models/README.md](experiments/next_generation/models/README.md) - Architecture details
- [data_generation/README.md](experiments/next_generation/data_generation/README.md) - Data formats

### Recent Updates (January 2026)

- ✅ Fixed PINN/RK_PINN to use `InterpolatedFieldTorch` (real field map)
- ✅ Generated 50M track training dataset
- ✅ Created momentum-split datasets (10M each)
- ✅ Submitted all 29 training experiments to HTCondor
- ✅ Created unified experiment runner and analysis notebook

**Benchmark Results:**
- All 9 extrapolators running successfully
- Test grid: 11×11 = 121 track states (various momenta and angles)
- Total execution: ~0.286s for full benchmark suite
- Methods tested: Reference RK4, BogackiShampine3, Verner7/9, Tsitouras5, Kisel, Herab, Linear, Parabolic

---

## 📋 Key Learnings

### LHCb Software Stack

**Correct way to run tests:**
```bash
# From stack directory (/data/bfys/gscriven/TE_stack)
Rec/run gaudirun.py <path-to-options-file>

# NOT: gaudirun.py <path> (missing environment setup)
```

**Test files:**
- `.qmt` files: QMTest descriptors (reference expected output)
- `.py` files in `tests/options/`: Gaudi configuration scripts
- `.ref` files in `tests/refs/`: Expected output for validation

**Adding new extrapolators** (from supervisor guide):
1. Copy existing extrapolator (e.g., `TrackKiselExtrapolator.cpp`)
2. Rename class and update CMakeLists.txt
3. Implement `propagate()` method (line ~68 in template)
4. Key function signature:
   ```cpp
   StatusCode propagate(
       Gaudi::TrackVector& stateVec,  // [x, y, tx, ty, q/p]
       double zOld, double zNew,
       Gaudi::TrackMatrix* transMat,  // Transport matrix (optional)
       IGeometryInfo const& geometry,
       LHCb::Tr::PID pid,
       const LHCb::Magnet::MagneticFieldGrid* grid
   ) const override;
   ```

**Simplest reference:** `TrackLinearExtrapolator.cpp` - straight-line propagation

---

## 🛠️ Dependencies

### C++ (LHCb Framework)
- Gaudi
- LHCb software stack
- Eigen3 (for ML inference)
- ROOT (for benchmarking)

### Python
```bash
pip install numpy torch tensorboard scikit-learn
```

Optional for benchmarking:
```bash
pip install uproot awkward  # For parsing ROOT files without PyROOT
```

---

## 📝 Experiment Tracking

All experiments logged in [experiments/experiment_log.csv](experiments/experiment_log.csv)

---

## 🤝 Workflow

1. Work in `experiments/next_generation/`
2. Log experiments to `experiment_log.csv`
3. Save models with metadata JSON
4. Update relevant README when completing milestones

---

## ⚠️ Important Notes

### Field Model

**Current status:** Now using real field map (`twodip.rtf`) via `InterpolatedFieldTorch`
- Full 3D field interpolation (Bx, By, Bz all vary with x, y, z)
- Grid: 81×81×146 points, 100mm spacing
- Peak |By| = 1.03 T at z ≈ 5007 mm

### HTCondor Settings

Jobs require these settings for the Nikhef STBC cluster:
```
+UseOS = "el9"
+JobCategory = "short"
```

---

**Last Updated:** January 22, 2026  
**Status:** 29 training experiments submitted, awaiting results
