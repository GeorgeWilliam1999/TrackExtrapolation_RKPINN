# V5: PINN Architecture Improvements — Comparative Study

## Motivation

V4 analysis revealed a fundamental flaw in the original PINN architecture:
the encoder sees only the 5-component initial state `[x, y, tx, ty, qop]` but
**not** the fractional z-position (`z_frac`). It produces a single correction
vector that is linearly scaled by `z_frac`, making it structurally incapable of
capturing the nonlinear position trajectories that arise from a
**spatially-varying** magnetic field (By varies 0.4–1.1 T across 8000 mm).

For the full diagnosis, see [V4/PINN_ARCHITECTURE_DIAGNOSIS.md](../V4/PINN_ARCHITECTURE_DIAGNOSIS.md).

V5 implements **all four proposed fixes** in a single unified training framework,
alongside an MLP baseline, for a direct comparison on the same data, same
hyperparameters, same evaluation.

## Architectures

| # | Model Type | Config Key | Key Idea | Data |
|---|-----------|------------|----------|------|
| 1 | **MLP Baseline** | `mlp` | Standard 6→hidden→4, endpoint only | MLP |
| 2 | **QuadraticResidual** | `quadratic` | IC + z·c₁ + z²·c₂ polynomial basis | PINN |
| 3 | **PINNZFracInput** | `zfrac` | z_frac as 7th encoder input, IC residual | PINN |
| 4 | **PDE-Residual PINN** | `pde` | Autograd physics loss (Lorentz force ODE) | MLP or PINN |
| 5 | **Compositional PINN** | `compositional` | Chain N short-step predictions | MLP |

### 1. MLP Baseline (`mlp`)
Standard feedforward network. Input `[x, y, tx, ty, qop, dz]` → output `[x, y, tx, ty]`.
Trained with MSE endpoint loss only. This is V4's MLPV4, carried forward as-is.

### 2. QuadraticResidual (`quadratic`)
Backbone produces two coefficient vectors `c₁` and `c₂`:

$$\hat{s}(z_\text{frac}) = s_0 + z_\text{frac} \cdot c_1 + z_\text{frac}^2 \cdot c_2$$

Guarantees IC at z_frac=0. The quadratic term captures parabolic position
dependence from integrating linear slope changes.

### 3. PINNZFracInput (`zfrac`)
z_frac is concatenated as the 7th input to the encoder:

$$\hat{s}(z_\text{frac}) = s_0 + z_\text{frac} \cdot \text{core}([s_0, q/p, dz, z_\text{frac}])$$

The network can learn arbitrary nonlinear z-dependence since it *sees* z_frac.

### 4. PDE-Residual PINN (`pde`)
Same architecture as #3, but with a **physics-informed loss** using autograd:

1. Forward-pass the model at sampled z_frac points (with `requires_grad`)
2. Compute $d\hat{s}/dz_\text{frac}$ via `torch.autograd.grad`
3. Convert to physical derivatives: $d\hat{s}/dz = (d\hat{s}/dz_\text{frac}) / dz$
4. Evaluate magnetic field $\vec{B}(x, y, z)$ at predicted positions using
   the interpolated field map
5. Compute Lorentz force ODE residual:

$$\mathcal{L}_\text{PDE} = \left\|\frac{d\hat{t}_x}{dz} - \kappa N [t_x t_y B_x - (1+t_x^2)B_y + t_y B_z]\right\|^2 + \text{(analogous for } t_y, x, y\text{)}$$

where $\kappa = c \cdot q/p$, $N = \sqrt{1 + t_x^2 + t_y^2}$.

**Two modes:**
- `"pde_mode": "pure"` — MLP data, loss = IC + endpoint + PDE residual
- `"pde_mode": "hybrid"` — PINN data, loss = IC + endpoint + supervised collocation + PDE residual

### 5. CompositionalPINN (`compositional`)
A single "step model" (6→hidden→4 with residual connection) chained N times:

$$s_{i+1} = s_i + \text{StepModel}([s_i, q/p, dz/N])$$

The total extrapolation `dz` is divided into N equal sub-steps. Each step
learns short-range extrapolation where the field is approximately constant,
and the composition covers the full variable-dz range.

**Configurable N** via `"n_steps"` in config (default 8). Two variants tested: N=8, N=16.

## Experiment Configuration

All models use **[1024, 512]** hidden layers with SiLU activation:

| Config File | Model | Data | Special |
|------------|-------|------|---------|
| `mlp_v5_2L_1024_512` | MLP | MLP (100M) | — |
| `quad_v5_2L_1024_512` | Quadratic | PINN (col10) | — |
| `zfrac_v5_2L_1024_512` | ZFrac | PINN (col10) | — |
| `pde_pure_v5_2L_1024_512` | PDE | MLP (100M) | batch=2048, pure mode |
| `pde_hybrid_v5_2L_1024_512` | PDE | PINN (col10) | batch=2048, hybrid mode |
| `comp_v5_2L_1024_512_N8` | Compositional | MLP (100M) | N=8 sub-steps |
| `comp_v5_2L_1024_512_N16` | Compositional | MLP (100M) | N=16 sub-steps |

**Shared hyperparameters:** lr=0.001, weight_decay=1e-4, warmup=5, patience=20,
epochs=100, seed=42, SiLU activation.

## Training Data

- **MLP data:** `V3/data/training_mlp_v3_100M_v2.npz`
  - `X`: [N, 6] = [x, y, tx, ty, qop, dz]
  - `Y`: [N, 4] = [x, y, tx, ty] at endpoint
- **PINN data:** `V3/data/training_pinn_v3_col10_v2.npz`
  - Same X, Y plus:
  - `z_frac`: [N, 10] = collocation fractions (0.1 to 0.9)
  - `Y_col`: [N, 10, 4] = ground truth at each collocation point

## How to Run

### Local (single model)
```bash
cd experiments/next_generation/
python V5/training/train_v5.py --config V5/training/configs/mlp_v5_2L_1024_512.json
```

### Cluster (all models)
```bash
cd experiments/next_generation/

# Dry run — validates configs, data, field map
bash V5/cluster/submit_all_v5.sh

# Submit to HTCondor
bash V5/cluster/submit_all_v5.sh --submit

# Monitor
condor_q -nobatch
watch -n 30 'condor_q -nobatch'
```

### Collect results after training
```bash
python V5/analysis/collect_results.py
```

## Directory Structure

```
V5/
├── README.md                     ← This file
├── training/
│   ├── train_v5.py               ← Unified training script (all 5 models + field)
│   ├── train_v5_wrapper.sh       ← HTCondor wrapper
│   └── configs/                  ← JSON config per model
│       ├── mlp_v5_2L_1024_512.json
│       ├── quad_v5_2L_1024_512.json
│       ├── zfrac_v5_2L_1024_512.json
│       ├── pde_pure_v5_2L_1024_512.json
│       ├── pde_hybrid_v5_2L_1024_512.json
│       ├── comp_v5_2L_1024_512_N8.json
│       └── comp_v5_2L_1024_512_N16.json
├── cluster/
│   ├── submit_v5_training.sub    ← HTCondor submit file
│   ├── v5_jobs.txt               ← Job list (7 configs)
│   ├── submit_all_v5.sh          ← Pre-flight + submit script
│   └── logs/                     ← Job stdout/stderr/log
├── trained_models/               ← Output (best_model.pt, history.json, etc.)
└── analysis/
    └── collect_results.py        ← Result collection script
```

## Evaluation Criteria

After training, compare all models on:

1. **Endpoint MSE** — primary metric, comparable across all models
2. **Position error** (mm) — denormalized x, y error at endpoint
3. **Slope error** (mrad) — denormalized tx, ty error at endpoint
4. **Trajectory shape** — for PINN models, evaluate at z_frac = 0.1, 0.2, ..., 0.9
5. **Training efficiency** — convergence speed, wall-clock time per epoch

## Expected Outcomes

- **MLP** should match V4 MLP performance (~1mm position, ~0.1mrad slope)
- **Quadratic** should improve trajectory shape over V3 linear PINN
- **ZFrac** should allow nonlinear z-dependence, improving positions
- **PDE-pure** is the most physically principled — may converge slower but should
  respect physics constraints even at unobserved z_frac values
- **PDE-hybrid** combines supervised data efficiency with physics consistency
- **Compositional N=8** learns short-range steps (~1000mm each for typical dz≈8000mm)
- **Compositional N=16** even shorter steps (~500mm each), closer to field-constant regime

## Notes

- `train_v5.py` is fully self-contained — all model classes, field models, and loss
  functions are defined inline (no V3 imports).
- The PDE-residual loss uses `torch.autograd.grad` with `create_graph=True` for
  second-order gradients through the optimizer. This increases memory usage;
  PDE configs use `batch_size=2048` (vs 4096 for others).
- The magnetic field is evaluated using the `InterpolatedFieldTorch` class with
  trilinear interpolation of `field_maps/twodip.rtf` via `grid_sample`.
- `z_ref=2500 mm` is used as the approximate upstream reference z position for
  the PDE residual. This is sufficient because the field mainly varies along z,
  and the field map covers the full LHCb magnet region.
