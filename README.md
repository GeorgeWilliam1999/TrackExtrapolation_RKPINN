# TrackExtrapolators — Neural Network Track Extrapolation for LHCb

**Replacing expensive Runge-Kutta track propagation with neural networks**

Reorganized: January 14, 2025 | Last Updated: March 2026  
Previous work archived in: `legacy/`

---

## Project Status

**Current Phase:** V5 architecture comparison in progress  
**Best Result:** **0.028 mm** position error at **1.3× faster** than C++ RK4 (fixed dz)  
**Variable dz:** ~1 mm accuracy achieved (V3/V4), architectures under improvement (V5)

### Completed

- LHCb software stack configured (DetDesc mode)
- All 9 C++ extrapolators benchmarked across 1210 track states
- 50M + 100M track training datasets generated
- V1: 53 models trained — MLP, PINN, RK_PINN architecture sweep (cluster 3880818)
- V2: 22 shallow-wide models — speed-optimized architectures (cluster 3891076)
- V3: Variable dz training framework (500–12000 mm step sizes)
- V4: PINN architecture root-cause diagnosis — fundamental IC flaw identified
- V5: 5 new PINN fix architectures designed (QuadraticResidual, ZFrac, PDE-Residual, Compositional)
- C++ `TrackMLPExtrapolator` integrated into LHCb framework with Eigen-based inference

### Key Findings

1. **MLP outperforms PINN** — simple feedforward networks achieve 0.03–0.07 mm accuracy, 10–100× better than physics-informed models
2. **Shallow-wide beats deep-narrow** — 1–2 layer networks with 256–1024 neurons outperform deeper architectures for both accuracy and speed
3. **PINN IC failure diagnosed** — original PINN/RK_PINN architectures ignore `z_frac` input entirely; the network learns a constant mapping (see [V4 diagnosis](experiments/gen_1/V4/PINN_ARCHITECTURE_DIAGNOSIS.md))
4. **10 models faster than C++ RK4** — the fastest (`mlp_v2_single_256`) achieves 3× speedup with 0.065 mm accuracy

---

## Repository Structure

```
TrackExtrapolators/
├── README.md                          # This file
├── CMakeLists.txt                     # Gaudi build configuration
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
│   ├── gen_1/                         # Active ML experiments
│   │   ├── README.md                  # Master experiment README
│   │   ├── MASTER_RESULTS.md          # Comprehensive V1–V4 results
│   │   ├── MASTER_RESULTS.tex         # LaTeX version
│   │   ├── LHCB_STACK_MANUAL.md       # LHCb stack setup guide
│   │   ├── DEPENDENCY_GRAPH.md        # Python module dependencies
│   │   ├── V1/                        # 53 models (deprecated)
│   │   ├── V2/                        # 22 shallow-wide (deprecated)
│   │   ├── V3/                        # Variable dz (active)
│   │   ├── V4/                        # PINN diagnosis + width sweep
│   │   ├── V5/                        # PINN architecture fixes (active)
│   │   ├── deployment/                # C++ model export tools
│   │   └── trained_models/            # All trained checkpoints
│   ├── gen_2/                         # Reference papers for future work
│   └── field_maps/                    # Magnetic field map experiments
│
├── tests/                             # LHCb framework tests
│   ├── options/                       # Gaudi configuration scripts
│   │   ├── benchmark_extrapolators.py
│   │   ├── benchmark_extrapolators_v2.py
│   │   ├── benchmark_many_events.py
│   │   └── test_extrapolators.py
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

## Results Summary

### Best Models (Fixed dz = 8000 mm, V1/V2)

| Model | Position Error | Timing (μs) | Speedup vs C++ RK4 | Recommendation |
|-------|---------------|-------------|---------------------|----------------|
| `mlp_v2_single_256` | 0.065 mm | 0.83 μs | **3.0×** | Best for speed |
| `mlp_v2_shallow_256` | 0.044 mm | 1.50 μs | 1.7× | Balanced |
| `mlp_v2_shallow_512_256` | **0.028 mm** | 1.93 μs | 1.3× | Best accuracy |
| C++ RK4 (CashKarp) | 0 (reference) | 2.50 μs | 1.0× | Baseline |

### Variable dz Models (500–12000 mm, V3/V4)

| Model | Position RMSE | Slope RMSE | Timing (μs) | Notes |
|-------|--------------|------------|-------------|-------|
| MLP shallow_256 | ~1.0 mm | 0.009 | 116.2 | Best V3 variable dz |
| PINN col10 | ~54 mm | 0.00025 | 79.3 | IC flaw — slopes only |

**Reference:** C++ RK4 = 85.2 μs/track for variable dz benchmarks.

### V5 Architectures Under Evaluation

| Architecture | Key Idea | Expected Improvement |
|-------------|----------|---------------------|
| QuadraticResidual | IC + z·c₁ + z²·c₂ polynomial basis | Better trajectory shape |
| PINNZFracInput | z_frac as 7th encoder input | Nonlinear z-dependence |
| PDE-Residual | Autograd Lorentz force loss | Physics consistency |
| CompositionalPINN | Chained short-step predictions | Field-constant regime |

For detailed results across all 75+ models, see [experiments/gen_1/MASTER_RESULTS.md](experiments/gen_1/MASTER_RESULTS.md).

---

## Model Architectures

| Model | Physics | Description | Status |
|-------|---------|-------------|--------|
| **MLP** | Implicit (data) | Standard feedforward, fastest inference | Production-ready |
| **PINN** | Explicit (PDE) | Physics-informed with Lorentz force | IC flaw identified (V4) |
| **RK_PINN** | Explicit (staged) | RK4-inspired multi-stage structure | IC flaw identified (V4) |
| **QuadraticResidual** | Explicit (polynomial) | Quadratic z-dependent correction | V5 — under evaluation |
| **PINNZFracInput** | Explicit (input) | z_frac visible to encoder | V5 — under evaluation |
| **PDE-Residual** | Explicit (autograd) | Lorentz force ODE residual loss | V5 — under evaluation |
| **CompositionalPINN** | Implicit (chained) | N short-step predictions composed | V5 — under evaluation |

---

## Quick Start

### Active Development

All ML experiment work is in `experiments/gen_1/`:

```bash
cd experiments/gen_1

# See the experiment README for full details
cat README.md

# Train a V5 model locally
python V5/training/train_v5.py --config V5/training/configs/mlp_v5_2L_1024_512.json

# Submit V5 jobs to HTCondor
bash V5/cluster/submit_all_v5.sh --submit
```

### Running C++ Extrapolator Tests

Prerequisites: CVMFS access, LHCb stack with DetDesc, `x86_64_v2-el9-gcc13+detdesc-opt`

```bash
cd /data/bfys/gscriven/TE_stack
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/options/test_extrapolators.py
```

See [experiments/gen_1/LHCB_STACK_MANUAL.md](experiments/gen_1/LHCB_STACK_MANUAL.md) for detailed setup instructions.

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [experiments/gen_1/README.md](experiments/gen_1/README.md) | Master experiment overview (V1–V5 status) |
| [experiments/gen_1/MASTER_RESULTS.md](experiments/gen_1/MASTER_RESULTS.md) | Comprehensive results with physics analysis |
| [experiments/gen_1/V4/PINN_ARCHITECTURE_DIAGNOSIS.md](experiments/gen_1/V4/PINN_ARCHITECTURE_DIAGNOSIS.md) | PINN IC failure root-cause analysis (11,000 words) |
| [experiments/gen_1/V5/README.md](experiments/gen_1/V5/README.md) | V5 architecture comparison design |
| [experiments/gen_1/LHCB_STACK_MANUAL.md](experiments/gen_1/LHCB_STACK_MANUAL.md) | LHCb software stack setup & build guide |
| [experiments/gen_1/deployment/README.md](experiments/gen_1/deployment/README.md) | C++ model export (binary format spec) |
| [src/README.md](src/README.md) | C++ source code component reference |
| [ml_models/README.md](ml_models/README.md) | Deployed model inventory |

---

## Dependencies

### C++ (LHCb Framework)
- Gaudi, LHCb software stack
- Eigen3 (ML inference + parametrized extrapolator)
- ROOT (benchmarking)
- GSL (numerical methods)

### Python (Training)
```bash
pip install numpy torch tensorboard scikit-learn
```

---

## Field Model

The real LHCb dipole field map (`twodip.rtf`) is used via `InterpolatedFieldTorch`:
- Full 3D field interpolation (Bx, By, Bz all vary with x, y, z)
- Grid: 81×81×146 points, 100 mm spacing
- Peak |By| = 1.03 T at z ≈ 5007 mm
- All PINN/RK_PINN training uses this field (legacy Gaussian approximation abandoned)

---

*Last Updated: March 2026*
