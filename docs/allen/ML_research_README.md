# ML_research — MLP Extrapolator Research for Allen ParKalman

Research workspace for developing and evaluating machine-learning-based track
extrapolators as replacements for the parametrised extrapolation in the Allen
GPU Kalman filter (HLT1).

## Directory Structure

```
ML_research/
├── baseline_ground_truth.ipynb   # Ground truth baseline notebook (TE kernel)
├── parkalman_workflow_demo.ipynb  # Allen extrapolator comparison demo
├── README.md                     # This file
└── standalone/                   # CPU-compiled test harness
    ├── main.cpp                  # Driver: trajectories + Kalman filter
    ├── Makefile                  # Build system (uses Allen headers from device/)
    ├── MLPExtrapolator.cuh       # MLP model loader and propagation
    ├── MagneticField.cuh         # Analytical LHCb dipole mock
    ├── cuda_compat.h             # CUDA → CPU compatibility shims
    ├── BackendCommon.h           # Compat shim
    ├── FloatOperations.cuh       # Compat shim
    └── *.csv / *.bin             # Generated outputs and model files
```

## Relationship to Allen

This folder does **not** modify any Allen production code. It uses the actual
Allen CUDA headers from `device/kalman/ParKalman/include/` (compiled as CPU
C++20 via compatibility shims) to ensure results are directly comparable to
what runs on GPU.

### Allen headers used (read-only)

| Header | Path |
|--------|------|
| `ButcherTableau.cuh` | `device/kalman/ParKalman/include/` |
| `ExtrapolatorCommon.cuh` | `device/kalman/ParKalman/include/` |
| `ParabolicExtrapolator.cuh` | `device/kalman/ParKalman/include/` |
| `RungeKuttaExtrapolator.cuh` | `device/kalman/ParKalman/include/` |

## Quick Start

```bash
# Build the standalone harness
cd ML_research/standalone
make

# Run RKN baseline (ground truth)
./run_extrapolators

# Run with an MLP model
./run_extrapolators --model your_model.bin
```

This produces CSV files that the notebooks consume:

| File | Content |
|------|---------|
| `trajectories.csv` | Per-step state for all extrapolator methods |
| `kalman_results.csv` | Per-track Kalman fit metrics (RKN baseline) |
| `kalman_hits.csv` | Per-hit filtered residuals and pulls (RKN) |
| `kalman_results_mlp.csv` | Per-track metrics (MLP, if model loaded) |
| `kalman_hits_mlp.csv` | Per-hit metrics (MLP, if model loaded) |

## Notebooks

### `baseline_ground_truth.ipynb`
Establishes the RKN ground truth baseline. Displays:
- Trajectory comparisons across Allen extrapolators
- χ²/ndof distribution with KS test
- Pull distributions (x, y, tx, ty, qop)
- Momentum resolution vs true momentum
- Per-hit filtered residuals by detector region
- `compare_with_baseline()` helper for MLP evaluation

### `parkalman_workflow_demo.ipynb`
Demonstrates the Allen extrapolator outputs with trajectory and residual plots.

### `gen2_true_data_deep_dive.ipynb` + `gen2_true_data_deep_dive.md`
Full audit of the 5 gen_2 `true_gen2_data` MLPs and why they all fail the
physics sanity check in Allen. **Read this before training any new model.**

---

## ⚠️ Conventions that MUST be respected by any new model

The gen_2 `true_gen2_data` sweep (5 MLPs, 5k–430k params) all passed their
training metrics (val_pos_mean < 0.5 mm) but failed catastrophically in the
Allen Kalman filter (χ²/ndof 10⁴–10⁵, pull widths 60+). The root cause was
a mismatch between the training conventions and the deployment conventions.
**These conventions are not negotiable** — the model must be trained on the
exact features it will be called with.

### State vector

Allen's canonical extrapolator state (`device/kalman/ParKalman/include/ExtrapolatorCommon.cuh`):

```
State = { x, y, z, tx, ty, qop }    with  tx = dx/dz,  ty = dy/dz
```

MLP input/output feature order (fixed):

```
input  [6] = [ x, y, tx, ty, qop, dz ]
output [4] = [ x_out, y_out, tx_out, ty_out ]    (qop is conserved by the MLP)
```

### Units — the single most important section

| Quantity | Unit | Notes |
|---|---|---|
| `x, y, x_out, y_out` | mm | |
| `z, dz` | mm | **`dz` must be sampled signed** — see below |
| `tx, ty` | dimensionless | `dx/dz`, `dy/dz` |
| `qop` | **c/MeV** (Allen convention) | `qop = c_light · q / p_MeV` |
| `c_light` | 299.792458 | mm·ns⁻¹·eplus (Gaudi units) |
| Magnetic field | Gaudi units | 1 Tesla = 1e-3 |

### `qop` convention — the bug that killed gen_2

**Allen convention (required):** `qop = c_light · q / p_MeV`

- For a 10 GeV track: `qop ≈ 3.00e-2` (positive charge)
- Standard deviation across Allen's phase space: `σ(qop) ≈ O(1e-2)`

**Previous training convention (WRONG for Allen):** `qop = q / p_MeV`

- For a 10 GeV track: `qop ≈ 1.00e-4`
- Feeding Allen's `qop` into a model trained on this convention places the
  input ~100σ outside the training normalisation — total inference garbage.

The gen_2 `true_gen2_data` dataset (`train_50M_dz25.npz`) was generated with
`qop = q/p`. **This must change for gen_3.** Regenerate with
`qop = c_light · q / p_MeV`. Do NOT add runtime conversions inside the loader
— the model should see the same number during training and inference.

### `dz` must be bidirectional

The Kalman filter runs a backward VELO pass (and in a full Allen fit, also
backward through UT/SciFi). It calls the extrapolator with `dz < 0`. The gen_2
dataset only sampled `dz ∈ [+25, +10 000] mm`, so backward calls were pure
out-of-distribution extrapolation → NaNs → diverged tracks.

**Requirement:** sample `dz ∈ [−10 000, −25] ∪ [+25, +10 000]` mm (skip the
near-zero band). Log-uniform in `|dz|` is strongly preferred — short `dz` is
disproportionately important because the Kalman filter takes many small
backward steps.

### Input-position coverage

The gen_2 dataset samples starting positions in a narrow ±300 mm window
(VELO-like), but output positions extend to ±3200 mm (SciFi). When the Kalman
filter calls the extrapolator *iteratively* layer by layer, its input `x`
after the first jump is already at 300–800 mm — beyond the training
input-box. The model has no training signal there.

**Requirement:** sample input positions from the full Allen track distribution
at every layer. Easiest recipe: run an Allen reconstruction once, harvest
`(state_in, state_out)` pairs at every layer interface, use those as training
data. This automatically covers VELO, UT, SciFi regions in both directions.

### Physics the dataset must include

The gen_2 generator used a vacuum Lorentz ODE only. Real detectors produce:

- **Multiple scattering** (Highland formula): θ_rms ∝ (13.6 MeV / p) · √(x/X₀)
- **Energy loss** (Bethe–Bloch): ~0.1% of p per plane in silicon
- **Fluctuations on `qop`**: the assumption `qop_out = qop_in` is wrong

Without these, `qop` cannot be held fixed — the model must predict
`qop_out` as a 5th output. The Allen MLP binary format already supports
5-output models; the loader will need a minor update.

### Training loss — weight by detector resolution

Plain MSE treats a 0.3 mm error in `x_out` as equivalent at every detector
plane. It is not: σ_VELO_x = 12 μm, σ_SciFi_x = 60 μm, σ_SciFi_y = 500 μm.
A MSE-trained model at 0.3 mm is already 25× the VELO pixel pitch and thus
inadmissible for the innermost stations.

**Requirement:** weight output components by `1/σ_detector²` matched to the
target layer. Target: 10 μm VELO x/y, 100 μm UT/SciFi.

### Jacobian regularisation

The Kalman filter needs `F = ∂state_out / ∂state_in` to propagate covariance.
An MSE-only model has *no* constraint on its Jacobian — finite-differencing
it gives noisy garbage (pull widths ~60 in the gen_2 audit).

**Requirement:** add `λ · ‖∂f_NN/∂x − ∂f_RK4/∂x‖²` to the loss, computed by
forward-mode autodiff (`torch.func.jacfwd`) on a subset of each batch.

### Export / binary format

Current `.bin` format (see `standalone/MLPExtrapolator.cuh`, V1/V2):

```
int model_type         (0=MLP, 1=PINN residual)
int num_layers
per layer:
  int rows, cols
  double weights[rows*cols]  (row-major)
  double biases[rows]
int input_size
double input_mean[input_size]
double input_std[input_size]
int output_size
double output_mean[output_size]
double output_std[output_size]
int activation_len
char activation[activation_len]       ("silu" | "tanh" | "relu" | "sigmoid")
```

**Gaps that must be closed in gen_3:** the format has **no provenance**. Add
a metadata block:

- `qop_convention` string (`"c_over_p"` for Allen, `"q_over_p"` otherwise)
- `dz_signed` bool
- `data_sha256` (hash of training data)
- `git_hash` of generator and training scripts
- `feature_order` string (to catch reordering mistakes)

### Mandatory pre-deployment smoke test

Any new checkpoint must pass this gate before being reviewed:

```bash
cd ML_research/standalone
./run_extrapolators --model your_model.bin
# Then in Python:
from pathlib import Path
import pandas as pd
r = pd.read_csv('kalman_results_mlp.csv').dropna(subset=['chi2'])
assert r.chi2_ndof.mean() < 2.0,       f"chi2/ndof = {r.chi2_ndof.mean():.2f}, fail"
assert abs((r.p_fit/r.p_true - 1).mean()) < 0.01, "dp/p bias > 1%, fail"
assert r.pull_x.std() < 1.5,           f"pull_x.std = {r.pull_x.std():.2f}, fail"
```

If those three assertions pass, the model is worth a full review. If they
fail, the model is broken and no amount of plot-inspection will save it.

### Summary: checklist for gen_3 model producers

- [ ] Data generator uses `qop = c_light · q / p_MeV` (Allen convention)
- [ ] `dz` sampled signed, covering `[−10 km, −25 mm] ∪ [+25 mm, +10 km]`
- [ ] Log-uniform sampling in `|dz|`
- [ ] Input positions cover the full Allen track distribution at every layer
- [ ] Generator includes multiple scattering and energy loss
- [ ] Model predicts 5 outputs including `qop_out` (not 4)
- [ ] Loss is detector-resolution-weighted MSE (not plain MSE)
- [ ] Loss includes Jacobian regularisation term
- [ ] `.bin` carries `qop_convention`, `dz_signed`, `data_sha256`, `git_hash`
- [ ] Training loop runs the 3-assertion smoke test and rejects failing ckpts
- [ ] Consider `NeuralRK4` architecture over plain MLP — already 2.4× better
      at 1/5 the parameters (see
      `TrackExtrapolation/experiments/gen_2/README.md` §v2_fixes)

### Reference: gen_2 failure numbers

For context on what "broken" looks like, the full audit is in
`gen2_true_data_deep_dive.md`. Headline:

| Model | params | χ²/ndof | δp/p bias | Pull(x) | Verdict |
|---|---:|---:|---:|---:|---|
| RKN baseline | — | **0.95** | **−0.04%** | **0.95** | PASS |
| mlp_tiny   | 4 888   | 57 644 | +335% | 57 | FAIL |
| mlp_small  | 17 944  | 139 614 | +592% | 72 | FAIL |
| mlp_medium | 101 016 | 32 368 | +785% | 61 | FAIL |
| mlp_large  | 398 616 | 132 491 | +463% | 62 | FAIL |
| mlp_wide   | 431 000 | 202 393 | +635% | 72 | FAIL |

A 90× parameter scan produced **no monotonic improvement** — this is not a
capacity problem, it is a convention/data problem. Fixing the conventions
above is the prerequisite to any further architecture work.

---

## Environment

Notebooks use the **TE** conda kernel (`/data/bfys/gscriven/conda/envs/TE`).
Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`.

## Baseline Reference Numbers (RKN, 500 tracks, seed=42)

| Metric | Value |
|--------|-------|
| χ²/ndof | 0.948 ± 0.205 |
| δp/p bias | −0.035% |
| δp/p resolution | 1.66% |
| Pull x std | 0.954 |
| Pull y std | 0.932 |
| Diverged | 0 |
