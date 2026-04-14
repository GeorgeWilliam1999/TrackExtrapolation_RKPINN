# gen_2: MLP + PINN + RK-PINN Track Extrapolation

> **Generation 2** — Systematic comparison of data-driven (MLP) and physics-informed
> (PINN, RK-PINN) approaches for LHCb charged-particle track extrapolation.

## Goal

Replace the C++ adaptive RK4 extrapolator (`TrackRungeKuttaExtrapolator`) with a
neural network that is **faster while maintaining high accuracy**. This generation
trains all three architecture types on the same 50M-track dataset and compares them
under identical conditions.

### Context: Why This Matters

Profiling shows **field evaluations dominate** the extrapolator cost. Two complementary
speedup strategies exist:

| Strategy | How | Status |
|----------|-----|--------|
| **NN field map** | Replace trilinear interpolation with tiny NN (ReLU 1L 32H) | gen_1 field_nn — **done**, 5.7× speedup |
| **End-to-end NN extrapolator** | Replace entire RK4 loop with single NN forward pass | **gen_2 — this experiment** |

The NN extrapolator eliminates *all* field evaluations per step (6 for CashKarp RK45),
potentially achieving an even larger speedup than the NN field map alone.

---

## Experiment Design

### Approaches

| Model | Architecture | Loss | Key Property |
|-------|-------------|------|-------------|
| **MLP** | Feedforward `[in_6] → H×N → [out_4]` | MSE only | Fastest inference, no physics |
| **PINN** | Residual formulation with skip connections | MSE + λ_pde·PDE + λ_ic·IC | Physics-constrained, smooth |
| **RK_PINN** | 4-stage backbone + heads (RK4-inspired) | MSE + λ_pde·PDE + λ_ic·IC | Multi-scale physics, learnable RK weights |

### Sweep Matrix (11 runs)

**MLP** — architecture size sweep (fixed seed=42):

| Config | Hidden dims | Params (approx) | Batch |
|--------|------------|-----------------|-------|
| `mlp/tiny` | [64, 64] | ~5k | 4096 |
| `mlp/small` | [128, 128] | ~20k | 4096 |
| `mlp/medium` | [256, 256, 128] | ~100k | 2048 |
| `mlp/large` | [512, 512, 256] | ~400k | 2048 |
| `mlp/wide` | [512, 512, 256, 128] | ~500k | 2048 |

**PINN** — physics loss weight sweep (medium [256, 256], tanh):

| Config | λ_pde | λ_ic | n_collocation |
|--------|-------|------|--------------|
| `pinn/medium_lam0.1` | 0.1 | 0.01 | 8 |
| `pinn/medium_lam1.0` | 1.0 | 0.1 | 8 |
| `pinn/medium_lam10.0` | 10.0 | 1.0 | 8 |

**RK_PINN** — physics loss weight sweep (medium [256, 256, 128], tanh):

| Config | λ_pde | λ_ic |
|--------|-------|------|
| `rk_pinn/medium_lam0.1` | 0.1 | 0.1 |
| `rk_pinn/medium_lam1.0` | 1.0 | 1.0 |
| `rk_pinn/medium_lam10.0` | 10.0 | 1.0 |

### Training Data

**Source**: `gen_1/V1/data_generation/datasets/train_50M.npz` (shared, not copied)

| Property | Value |
|----------|-------|
| Samples | 50M tracks |
| Input `X` [N,6] | x, y, tx, ty, q/p, **dz** |
| Output `Y` [N,4] | x_out, y_out, tx_out, ty_out |
| dz range | U[100, 10000] mm (variable per sample) |
| Momentum | 1–100 GeV (log-uniform) |
| Field | Interpolated `twodip.rtf` (81×81×146 grid) |
| Generator | Python RK4, 5mm fixed step |

### Success Metrics

| Metric | Target | Stretch |
|--------|--------|---------|
| Position error (mean) | < 0.1 mm | < 0.01 mm |
| Position error (95%) | < 0.5 mm | < 0.1 mm |
| Slope error (mean) | < 1e-4 | < 1e-5 |
| Jacobian ||J||_F | Finite, physically reasonable | Close to RK4 Jacobian |

### Monitoring

All runs tracked via **MLflow** under experiment `gen_2_track_extrapolation`:

- Per-epoch: train_loss, val_loss, data_loss, pde_loss, ic_loss
- Per-epoch: val_pos_mean_mm, val_pos_95_mm, val_slope_mean (+ per-component)
- Final: test metrics, training time, Jacobian statistics
- Artifacts: best_model.pt, normalization.json, config.json, history.json, jacobian.json

**TensorBoard** also enabled for real-time monitoring:
```bash
tensorboard --logdir experiments/gen_2/trained_models
```

---

## Directory Structure

```
gen_2/
├── README.md                          ← this file
├── models/
│   ├── architectures.py → gen_1       (symlink — single source of truth)
│   └── train.py                       (enhanced: physics loss, Jacobian)
├── utils/ → gen_1/V1/utils/           (symlink — magnetic field, RK4)
├── configs/
│   ├── mlp/                           (5 JSON configs)
│   ├── pinn/                          (3 JSON configs)
│   └── rk_pinn/                       (3 JSON configs)
├── condor/
│   ├── submit.sub                     (HTCondor submit file)
│   ├── run_training.sh                (worker script)
│   ├── jobs.txt                       (config list — 11 jobs)
│   └── logs/                          (stdout/stderr)
├── trained_models/                    (output — one dir per run)
└── mlruns/                            (MLflow tracking store)
```

---

## How to Run

### Single run (interactive)

```bash
cd experiments/gen_2/models

# MLP from config
python train.py --config ../configs/mlp/medium.json

# PINN from CLI
python train.py --model pinn --preset medium --lambda_pde 1.0 --epochs 200

# Quick smoke test (1k samples)
python train.py --model mlp --preset tiny --max_samples 1000 --epochs 5 --no-mlflow
```

### All runs (HTCondor batch)

```bash
cd experiments/gen_2/condor
condor_submit submit.sub          # all 11 jobs
condor_q -nobatch                 # check status
```

### Single condor job

```bash
condor_submit submit.sub -append 'queue config_path from (configs/mlp/medium.json)'
```

### View results

```bash
# MLflow UI
cd experiments/gen_2
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000

# TensorBoard
tensorboard --logdir trained_models --port 6006
```

---

## Key Changes from gen_1

| Feature | gen_1/V1 | gen_2 |
|---------|----------|-------|
| Model types | MLP only | MLP + PINN + RK_PINN |
| Training loss | MSE only | MSE + PDE + IC (physics-informed) |
| dz handling | Variable (data), fixed-mean (PINN) | **Per-sample dz** for all models |
| Jacobian | Not computed | Autograd transport matrix |
| Monitoring | MLflow basic | MLflow + TensorBoard, per-component |
| Config | CLI only | **JSON config files** + CLI override |
| Seeds | Added (seed=42) | Inherited, enforced |
| Metrics | pos_mean, slope_mean | + pos_99, slope_95, Jacobian ||F|| |

### Bug Fix: PINN Variable dz

gen_1 PINN used `mean(dz)` for all samples — incorrect for variable-dz training data.
Fixed in `architectures.py`: `forward_at_z()` now accepts per-sample `dz` tensor,
and `forward()` passes `x[:, 5]` (per-sample dz) through to the skip connections.

---

## Next Steps (after gen_2 results)

1. **Compare**: MLP vs PINN vs RK_PINN accuracy-vs-params Pareto plot
2. **Export**: Best model → `.bin` weights for `TrackMLPExtrapolator.cpp`
3. **Benchmark**: C++ inference timing vs RK4 baseline (μs/track)
4. **Integrate**: Deploy in LHCb Gaudi framework for end-to-end test
5. **Jacobian**: Compare NN Jacobian vs RK4 Jacobian for track fitting
