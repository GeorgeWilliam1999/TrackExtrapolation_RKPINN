# Project Context — Track Extrapolation for LHCb

**Goal:** Replace the C++ adaptive Runge-Kutta track extrapolator with a faster neural network that maintains high accuracy.

**Last updated:** April 2026

---

## The Problem

Charged particles in LHCb are propagated ("extrapolated") through the magnetic field from one detector element to another. This is done many times per event in reconstruction. The current production code is an **adaptive Runge-Kutta** integrator (`TrackRungeKuttaExtrapolator.cpp`, 1444 lines) that:

1. Takes a 5D state vector `[x, y, tx, ty, q/p]` at position `z`
2. Propagates it to a new `z` position through the LHCb dipole field
3. Optionally computes a 5×5 transport (Jacobian) matrix
4. Uses adaptive step-size control with error estimation (Cash-Karp, Verner, etc.)

The field is evaluated via trilinear grid interpolation from a 81×81×146 measurement grid (`twodip.rtf`).

**Bottleneck:** This extrapolator is called O(10⁶) times per event. Even at ~2.5 μs/call, it accounts for a significant fraction of HLT2 reconstruction time.

**Success criteria:** A neural network that:
- Matches the RK4 output to < 0.1 mm position error (ideally < 0.05 mm)
- Is measurably faster than the C++ RK4 (~2.5 μs/track baseline for fixed dz)
- Works with variable dz (the reconstruction dispatches extrapolators by step size)
- Integrates into the Gaudi framework via `TrackMLPExtrapolator`

---

## Physics: Equations of Motion

The state evolves in z (beam direction) according to the Lorentz force:

```
dx/dz  = tx
dy/dz  = ty
dtx/dz = κ · N · [ty·(tx·Bx + Bz) - (1 + tx²)·By]
dty/dz = κ · N · [-tx·(ty·By + Bz) + (1 + ty²)·Bx]
```

where:
- `κ = (q/p) × c_light` (c_light = 2.99792458×10⁻⁴ in LHCb units)
- `N = √(1 + tx² + ty²)`
- `Bx, By, Bz` from the field map (Tesla)
- `q/p` in 1/MeV (charge over momentum)

The dominant field component is `By` (vertical dipole), causing horizontal bending.

---

## LHCb Software Integration Points

### Where the extrapolator is called
- **HLT2 tracking configuration:** `Moore/Hlt/RecoConf/python/RecoConf/hlt2_tracking.py` (line ~563)
- Selects extrapolator based on step size via `TrackMasterExtrapolator` → `TrackDistanceExtraSelector`
- Long-distance steps → `TrackRungeKuttaExtrapolator` (what we replace)
- Short-distance steps → `TrackParabolicExtrapolator` or `TrackLinearExtrapolator`

### The target C++ class
- **Production RK4:** `src/TrackRungeKuttaExtrapolator.cpp` (1444 lines, supports 16 Butcher tableaux)
- **Our NN replacement:** `ml_models/src/TrackMLPExtrapolator.cpp` (571 lines, Eigen-based inference)

### Upstream tracking (context — not what we replace)
- VELO pattern recognition: `Rec/Pr/PrPixel/src/VeloClusterTrackingSIMD.cpp`
- Allen GPU VELO: `Allen/device/velo/search_by_triplet/src/SearchByTriplet.cu`

### Key interfaces
- `ITrackExtrapolator::propagate(stateVec, zOld, zNew, transMat, geometry, pid, grid)`
- State vector: `[x, y, tx, ty, q/p]` (5D), units: mm, dimensionless slopes, c/GeV
- Transport matrix: 5×5 Jacobian (needed for Kalman filter — future work, not in first iteration)

---

## Data Assets

### Magnetic field map
- **File:** `experiments/field_maps/twodip.rtf`
- **Format:** ASCII, 6 columns: `x y z Bx By Bz` (mm, Tesla)
- **Grid:** 81 × 81 × 146 = 957,906 points
- **Range:** x,y ∈ [-4000, 4000] mm (100mm step), z ∈ [-500, 14000] mm (variable step)
- **Peak |By|:** ~1.03 T at z ≈ 5007 mm
- **Used by:** `InterpolatedFieldTorch` in Python training, trilinear interpolation in C++

### Training data
- **V1 dataset:** 50M tracks, fixed dz = 8000 mm, `experiments/gen_1/V1/data_generation/datasets/train_50M.npz`
- **Format:** `X[N,6]` = [x, y, tx, ty, q/p, dz], `Y[N,4]` = [x_out, y_out, tx_out, ty_out], `P[N]` = momentum
- Generated from C++ RK4 ground truth via LHCb stack

### Trained models (V1)
- 17 MLP architectures in `experiments/gen_1/V1/trained_models/`
- MLflow tracking: 43 runs (17 FINISHED, 26 FAILED) in `experiments/gen_1/V1/mlruns/`
- Results summary: `experiments/gen_1/V1/analysis/v1_sweep_results.json`

---

## Architecture Approaches

### 1. MLP (current V1 — data-driven)
- Input: `[x, y, tx, ty, q/p, dz]` → Output: `[x_out, y_out, tx_out, ty_out]`
- Standard feedforward, MSE loss on endpoint
- **Pro:** Simple, fast inference, best accuracy so far
- **Con:** No physics constraint, fixed dz in V1

### 2. PINN (defined in code, not yet trained properly)
- Adds Lorentz force PDE residual loss via autodiff
- Enforces IC at z=0, PDE at collocation points, endpoint match
- **Known issue from earlier work:** IC enforcement flaw — needs redesign

### 3. RK-PINN (from reference papers)
- NN predicts RK stages h^k, combined via Butcher weights
- Can be data-free (only physics loss) or hybrid
- Variable time-step version exists (Stiasny et al.)
- **Relevance:** Directly mirrors the C++ code structure

### Reference papers (attached by user)
1. **Stiasny et al. (2021)** — RK-PINN for power system dynamics, variable dt, data-free
2. **Wang & Perdikaris (2023)** — Long-time integration with physics-informed DeepONets, iterative short-step
3. **Zhai et al. (2023)** — RK4-PINN for parameter estimation, automatic differentiation for physics loss
4. **Zhang et al. (2024)** — RK4 in PINN loss function (replacing AD), FDM spatial derivatives, mMLP architecture

---

## Profiling Data
- `extrapolator_profiling_report.pdf` — C++ sub-operation timing breakdown
- `experiments/field_maps/field_nn/` — extensive field map NN analysis (timing, Pareto curves, Amdahl's law)

---

## Current State (April 2026)

| What | Status |
|------|--------|
| C++ RK extrapolators | ✅ All 9 benchmarked |
| 50M training dataset (fixed dz) | ✅ Generated |
| V1 MLP sweep (17 models) | ✅ Trained, results in MLflow |
| C++ `TrackMLPExtrapolator` | ✅ Exists (571 lines), Eigen inference |
| Field map (twodip.rtf) | ✅ Available |
| NN field map approximation | ✅ Trained models exist |
| Variable dz training | ❌ Not done (V1 = fixed dz only) |
| PINN / RK-PINN training | ❌ Not done (code exists but IC flaw) |
| Model deployment + .bin export | ❌ Pipeline exists but bins removed for clean start |
| Transport matrix (Jacobian) | ❌ Not addressed yet |
| End-to-end integration test | ❌ Not done |

---

## Environment

- **Python:** conda env `TE` — Python 3.10, PyTorch 2.9.1, mlflow 3.10.1
- **C++ stack:** `TE_stack/` — Gaudi/LHCb with DetDesc, `x86_64_v2-el9-gcc13+detdesc-opt`
- **Cluster:** HTCondor with GPU nodes
- **Pinned:** `environment.yml`, `requirements.txt` in repo root
