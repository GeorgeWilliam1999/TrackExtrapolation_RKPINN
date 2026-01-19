# TrackExtrapolators - Next Generation ML Models

**Clean repository for next-generation track extrapolation experiments**

Reorganized: January 14, 2025  
Previous work archived in: `legacy/`

---

## 🎯 Project Status

**Current Phase:** ✅ **C++ Baselines Established**  
**Goal:** Train next-generation ML track extrapolators for LHCb

**Completed:**
- ✅ LHCb software stack properly configured (DetDesc mode)
- ✅ C++ extrapolator tests running successfully  
- ✅ All 9 extrapolators benchmarked across 1210 track states
- ✅ Quantitative accuracy analysis complete
  - **Best:** BogackiShampine3, Verner9 (0.10mm mean error)
  - **Fast:** Herab (0.76mm mean error)
  - **Problematic:** Kisel (39.8mm mean error)

**Next Steps:**
1. Generate training data with validated RK4 parameters
2. Train baseline MLP to beat Herab's 0.76mm accuracy
3. Implement timing benchmarks for ML inference
4. Compare ML performance vs C++ baselines

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
│   ├── ExtrapolatorTester.cpp             # Simple benchmark tool
│   └── TrackExtrapolatorTesterSOA.cpp     # Full benchmark with timing
│
├── tests/                             # LHCb framework tests
│   ├── options/                       # Gaudi configuration files
│   └── qmtest/                        # LHCb test descriptors
│
├── experiments/
│   ├── next_generation/               # 🆕 ACTIVE DEVELOPMENT
│   │   ├── EXPERIMENT_DESIGN.md       # Full experimental plan
│   │   ├── REFERENCES.md              # Literature review
│   │   ├── DATA_AND_MODEL_MANAGEMENT.md   # Infrastructure design
│   │   ├── REVIEW_AND_GAPS.md         # Gap analysis (review)
│   │   ├── GAP_ANALYSIS_FINDINGS.md   # Investigation results
│   │   │
│   │   ├── benchmarking/              # C++ baseline benchmarks
│   │   │   ├── README.md              # How to benchmark
│   │   │   └── benchmark_cpp.py       # Python wrapper
│   │   │
│   │   ├── data_generation/           # Training data creation
│   │   │   ├── generate_data.py       # Main script
│   │   │   └── datasets/              # Generated .npy files
│   │   │
│   │   ├── training/                  # Model training scripts
│   │   ├── analysis/                  # Result analysis notebooks
│   │   ├── deployment/                # ONNX export, C++ integration
│   │   │
│   │   └── utils/                     # Shared utilities
│   │       └── rk4_propagator.py      # Pure Python RK4 integrator
│   │
│   ├── experiment_log.csv             # Experiment tracking
│   └── README.md                      # Experiment guidelines
│
└── legacy/                            # 📦 ARCHIVED (previous work)
    ├── old_notebooks/                 # Analysis notebooks
    ├── old_experiments/               # All previous experiments
    ├── old_python_scripts/ml_models/  # ML training code & trained models
    ├── plots/                         # Old plots
    ├── report/                        # LaTeX report
    ├── lhcb-metainfo/                 # Metadata
    └── OLD_README.md                  # Previous README (0.21mm claims)
```

---

## 🚀 Quick Start

### Prerequisites

This project uses the LHCb software stack. You must have:
- Access to CVMFS (e.g., Nikhef STBC cluster)
- LHCb stack built with DetDesc geometry backend
- Environment: `x86_64_v2-el9-gcc13+detdesc-opt`

**Setup location:** `/data/bfys/gscriven/TE_stack/`

### 1. Run Existing C++ Tests

Verify the LHCb framework is working correctly:

```bash
# From the stack directory
cd /data/bfys/gscriven/TE_stack

# Run the standard extrapolator test
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/qmtest/test_extrapolators.qmt

# Should see: All extrapolators running successfully with accuracy comparisons
```

This tests 9 different extrapolators across a grid of track states.

### 2. Run Comprehensive Benchmarks

```bash
# Run the benchmark configuration
cd /data/bfys/gscriven/TE_stack
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/options/benchmark_extrapolators.py

# Check timing output in the log
# Look for "Timing table" showing execution time per algorithm
```

**Benchmark includes:**
- Reference methods: RK4 (multiple schemes)
- Fast approximations: Kisel, Herab, Linear, Parabolic
- Accuracy: Compared against high-precision STEP integrator
- Test grid: 11×11 = 121 track states per extrapolator

### 3. Extract Performance Metrics

```bash
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/benchmarking

# Parse the benchmark log (when available)
# python parse_benchmark_results.py

# Analyze results
# jupyter notebook analyze_benchmarks.ipynb
```

### 4. Generate Training Data (Future)

```bash
cd experiments/next_generation/data_generation

# Generate training set (10K tracks, 5mm step size)
python generate_data.py --n-tracks 10000 --name train --step-size 5.0

# Generate validation set
python generate_data.py --n-tracks 2000 --name val --step-size 5.0

# Generate test set
python generate_data.py --n-tracks 2000 --name test --step-size 5.0
```

**Output:** `datasets/X_train.npy`, `Y_train.npy`, `P_train.npy` (and val, test)

### 3. Train Model (TODO - Coming Next)

```bash
cd experiments/next_generation/training

# Will create training scripts next
# python train_mlp.py --architecture small
```

---

## 📊 Legacy Results (For Reference)

Previous experiments (now in `legacy/`) achieved:

| Model | Activation | Mean Error | Dataset | Notes |
|-------|------------|------------|---------|-------|
| MLP (SiLU) | SiLU | **0.334 mm** | 50K tracks | Best from legacy |
| MLP (Tanh) | Tanh | 0.63 mm | 50K tracks | Baseline |
| PINN | Various | 18-329 mm | 50K tracks | ❌ Failed |

**Architecture:** [128, 128, 64] with 25,924 parameters  
**Data source:** Python RK4 with analytical field (B₀=1.0T)

---

## 🔬 Active Development: Next Generation

### Current Tasks

- [x] **Setup LHCb environment** - Stack built with DetDesc backend
- [x] **Verify C++ tests** - All extrapolators working correctly
- [x] **Run comprehensive benchmarks** - 9 extrapolators tested
- [ ] **Extract timing metrics** - Parse logs for performance data
- [ ] **Analyze accuracy** - Quantify errors vs reference method
- [ ] Generate pilot dataset (10K tracks) with validated step size
- [ ] Train baseline MLP and compare to legacy (0.334mm claim)
- [ ] Implement uncertainty quantification

### Design Documents

Read in this order:
1. [EXPERIMENT_DESIGN.md](experiments/next_generation/EXPERIMENT_DESIGN.md) - Full plan
2. [GAP_ANALYSIS_FINDINGS.md](experiments/next_generation/GAP_ANALYSIS_FINDINGS.md) - What exists vs what's needed
3. [DATA_AND_MODEL_MANAGEMENT.md](experiments/next_generation/DATA_AND_MODEL_MANAGEMENT.md) - Infrastructure

### Recent Breakthroughs

**LHCb Software Configuration (Jan 14, 2025):**
- ✅ Tests must be run via `Rec/run` script (not direct `gaudirun.py`)
- ✅ Conditions database requires proper PyConf setup with `testfiledb`
- ✅ SSH authentication to CERN GitLab working (port 7999)
- ✅ CVMFS resources accessible (field maps, detector DB, lhcb-metainfo)

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

### From User

> "I THINK it is 5mm" - RK4 step size (need to verify by benchmarking)

> "This is acceptable for now but when we have the true map we will need to re run everything" - Regarding analytical field model

### Critical Next Step

**BENCHMARK THE C++ RK4 FIRST!** 

Without baseline timing, we can't validate the "10× speedup" target.

Expected: ~50-150 μs per track  
Target ML: < 15 μs per track (10× faster)

---

**Last Updated:** January 14, 2025  
**Status:** Repository reorganized, ready for next-generation experiments
