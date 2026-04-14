# TrackExtrapolators — Neural Network Track Extrapolation for LHCb

**Replacing expensive Runge-Kutta track propagation with neural networks**

Restarted: April 2026  
Previous work archived in: `legacy/`

---

## Project Status

**Current Phase:** V1 MLP architecture sweep complete — analysis in progress

### Completed

- LHCb software stack configured (DetDesc mode)
- All 9 C++ extrapolators benchmarked across 1210 track states
- 50M track training dataset generated (fixed dz = 8000 mm)
- V1: 17 MLP architectures trained with MLflow tracking (43 runs, 17 converged)
- 4 deployed `.bin` models for C++ inference
- C++ `TrackMLPExtrapolator` integrated into LHCb framework with Eigen-based inference
- NN field map experiments (separate from track extrapolation)

### V1 Key Observations

- MLP architectures from [64,64] to [1024,512,256] all trained on 5M samples (from 50M pool)
- Training configs and MLflow runs preserved in `experiments/gen_1/V1/`
- **Note:** V1 models trained without fixed random seeds — results not bit-reproducible (now fixed for future training)

---

## Repository Structure

```
TrackExtrapolators/
├── README.md                          # This file
├── CMakeLists.txt                     # Gaudi build configuration
├── requirements.txt                   # Core Python dependencies (pinned)
├── environment.yml                    # Full conda environment spec
│
├── src/                               # C++ production code (LHCb framework)
│   ├── README.md                      # Component reference
│   ├── TrackRungeKuttaExtrapolator.cpp    # Adaptive RK4 (CashKarp/Verner)
│   ├── TrackSTEPExtrapolator.cpp          # RKN (ATLAS STEP algorithm)
│   ├── TrackKiselExtrapolator.cpp         # Kisel polynomial (CBM origin)
│   ├── TrackHerabExtrapolator.cpp         # Hera-B RK5
│   ├── TrackLinearExtrapolator.cpp        # Straight-line (no field)
│   ├── TrackParabolicExtrapolator.cpp     # 2nd-order parabolic
│   ├── TrackParametrizedExtrapolator.cpp  # Basis-function parametrization
│   ├── TrackMasterExtrapolator.cpp        # Orchestrator (delegates)
│   ├── TrackExtrapolatorTesterSOA.cpp     # SOA benchmark with timing
│   └── ...                                # Base classes, material locators, selectors
│
├── ml_models/                         # Deployed neural network models
│   ├── README.md                      # Model inventory & usage
│   ├── models/                        # Binary model files (.bin)
│   └── src/
│       └── TrackMLPExtrapolator.cpp   # Eigen-based NN inference (571 lines)
│
├── experiments/
│   ├── README.md                      # Experiment overview
│   ├── gen_1/                         # ML experiments
│   │   ├── V1/                        # MLP architecture sweep (17 models)
│   │   ├── deployment/                # C++ model export tools
│   │   └── archive/                   # Historical analysis from earlier work
│   └── field_maps/                    # Magnetic field map NN experiments
│
├── tests/                             # LHCb framework tests
│   ├── options/                       # Gaudi configuration scripts
│   ├── qmtest/                        # QMTest descriptors
│   └── refs/                          # Reference output files
│
├── play_time/                         # Scratch / exploration area
│   └── rk_extrapolators_explained.ipynb
│
├── legacy/                            # Archived previous work (pre-2025)
│   ├── old_notebooks/
│   ├── old_experiments/
│   ├── old_python_scripts/
│   ├── plots/
│   └── report/
│
└── doc/
    └── release.notes                  # Official LHCb release notes (through 2016)
```

---

## Model Architectures

| Model | Physics | Description |
|-------|---------|-------------|
| **MLP** | Implicit (data) | Standard feedforward, fastest inference |
| **PINN** | Explicit (PDE) | Physics-informed with Lorentz force (defined in code, not yet trained in V1) |
| **RK_PINN** | Explicit (staged) | RK4-inspired multi-stage structure (defined in code, not yet trained in V1) |

---

## Quick Start

### Python Environment

```bash
# Recreate conda environment
conda env create -f environment.yml

# Or install core deps only
conda activate TE
pip install -r requirements.txt
```

### Training (V1)

```bash
cd experiments/gen_1/V1/models

# Train a single MLP
python train.py --model mlp --hidden_dims 256 256 --epochs 500 --seed 42

# See all options
python train.py --help
```

### Running C++ Extrapolator Tests

Prerequisites: CVMFS access, LHCb stack with DetDesc, `x86_64_v2-el9-gcc13+detdesc-opt`

```bash
cd /data/bfys/gscriven/TE_stack
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/options/test_extrapolators.py
```

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [src/README.md](src/README.md) | C++ source code component reference |
| [ml_models/README.md](ml_models/README.md) | Deployed model inventory |
| [experiments/README.md](experiments/README.md) | Experiment overview |
| [experiments/gen_1/deployment/README.md](experiments/gen_1/deployment/README.md) | C++ model export (binary format spec) |

---

## Dependencies

### C++ (LHCb Framework)
- Gaudi, LHCb software stack
- Eigen3 (ML inference + parametrized extrapolator)
- ROOT (benchmarking)
- GSL (numerical methods)

### Python (Training)

See [requirements.txt](requirements.txt) and [environment.yml](environment.yml).

Key packages: Python 3.10, PyTorch 2.9.1, numpy 2.2.6, mlflow 3.10.1, scipy 1.15.3

---

## Field Model

The real LHCb dipole field map (`twodip.rtf`) is used via `InterpolatedFieldTorch`:
- Full 3D field interpolation (Bx, By, Bz all vary with x, y, z)
- Grid: 81×81×146 points, 100 mm spacing
- Peak |By| = 1.03 T at z ≈ 5007 mm

---

*Last Updated: April 2026*
