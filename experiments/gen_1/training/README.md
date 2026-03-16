# Training Jobs for Track Extrapolation Neural Networks

This directory contains HTCondor submission scripts for V1/V2 experiments.

**⚠️ NOTE**: V1/V2 models were trained with **fixed dz=8000mm** and cannot generalize to variable step sizes. For V3 training with variable dz, see [V3/training/README.md](../V3/training/README.md).

## Quick Start

```bash
# Submit all V1 experiments (29 jobs, 10 epochs)
./submit_all.sh

# Or use the unified experiment runner (recommended)
cd ..
python run_all_experiments.py --all

# Or submit individual experiments
condor_submit jobs/mlp_medium.sub
```

## Training Status (January 2026)

| Training Round | Cluster | Jobs | Epochs | Status |
|---------------|---------|------|--------|--------|
| V1 (core + ablations) | 3880473-3880501 | 29 | 10 | ✅ Complete |
| V1 (full GPU suite) | 3880818 | 30 | 10 | ✅ Complete |
| V2 (shallow-wide) | 3891076 | 22 | 20 | ✅ Complete |

**Total Models Trained: 53 V1 + 22 V2 = 75 models**

Check status: `condor_q`

## V1 Experiments Overview

### Core Experiments (12 total)

| # | Job | Architecture | Size | Params |
|---|-----|--------------|------|--------|
| 1 | `mlp_tiny` | MLP | [64, 64] | ~5k |
| 2 | `mlp_small` | MLP | [128, 128] | ~20k |
| 3 | `mlp_medium` | MLP | [256, 256, 128] | ~100k |
| 4 | `mlp_wide` | MLP | [512, 512, 256, 128] | ~500k |
| 5 | `pinn_tiny` | PINN | [64, 64] | ~5k |
| 6 | `pinn_small` | PINN | [128, 128] | ~20k |
| 7 | `pinn_medium` | PINN | [256, 256, 128] | ~100k |
| 8 | `pinn_wide` | PINN | [512, 512, 256, 128] | ~500k |
| 9 | `rkpinn_tiny` | RK_PINN | [64, 64] | ~5k |
| 10 | `rkpinn_small` | RK_PINN | [128, 128] | ~20k |
| 11 | `rkpinn_medium` | RK_PINN | [256, 256, 128] | ~100k |
| 12 | `rkpinn_wide` | RK_PINN | [512, 512, 256, 128] | ~500k |

### Physics Loss Ablations (8 total)

| # | Job | λ_pde | λ_ic | Purpose |
|---|-----|-------|------|---------|
| 13 | `pinn_medium_data_only` | 0.0 | 0.0 | No physics (MLP-like) |
| 14 | `pinn_medium_pde_weak` | 0.01 | 0.01 | Weak regularization |
| 15 | `pinn_medium_pde_strong` | 10.0 | 10.0 | Strong physics |
| 16 | `pinn_medium_pde_dominant` | 1.0 | 1.0 | Physics-dominated |
| 17 | `rkpinn_medium_data_only` | 0.0 | 0.0 | No physics |
| 18 | `rkpinn_medium_pde_weak` | 0.01 | 0.01 | Weak regularization |
| 19 | `rkpinn_medium_pde_strong` | 10.0 | 10.0 | Strong physics |
| 20 | `rkpinn_medium_pde_dominant` | 1.0 | 1.0 | Physics-dominated |

### Momentum Range Studies (9 total)

| # | Job | Architecture | Momentum |
|---|-----|--------------|----------|
| 21 | `mlp_medium_low_p` | MLP | 0.5–5 GeV |
| 22 | `mlp_medium_mid_p` | MLP | 5–20 GeV |
| 23 | `mlp_medium_high_p` | MLP | 20–100 GeV |
| 24 | `pinn_medium_low_p` | PINN | 0.5–5 GeV |
| 25 | `pinn_medium_mid_p` | PINN | 5–20 GeV |
| 26 | `pinn_medium_high_p` | PINN | 20–100 GeV |
| 27 | `rkpinn_medium_low_p` | RK_PINN | 0.5–5 GeV |
| 28 | `rkpinn_medium_mid_p` | RK_PINN | 5–20 GeV |
| 29 | `rkpinn_medium_high_p` | RK_PINN | 20–100 GeV |

## Prerequisites

1. **Training Data** must be generated first:
   - Full range: `data_generation/data/training_50M.npz`
   - Low-p: `data_generation/data/training_low_p.npz` (0.5-5 GeV)
   - Mid-p: `data_generation/data/training_mid_p.npz` (5-20 GeV)
   - High-p: `data_generation/data/training_high_p.npz` (20-100 GeV)

2. **Conda Environment** with PyTorch and CUDA support

## Monitoring

```bash
# Check job status
condor_q

# Watch logs
tail -f logs/mlp_medium.out

# Check all running jobs
condor_q -format "%s\n" Args | head -20
```

## Output

Trained models are saved to `trained_models/<experiment_name>/`:
- `best_model.pt` - Best model checkpoint
- `normalization.json` - Input/output normalization parameters
- `config.json` - Training configuration
- `history.json` - Training metrics history
- `model_config.json` - Model architecture details

## Resource Requirements

| Size | Memory | Est. Time (50M samples) |
|------|--------|------------------------|
| tiny | 8 GB | ~2 hours |
| small | 8 GB | ~3 hours |
| medium | 8 GB | ~5 hours |
| wide | 16 GB | ~8 hours |

All jobs request 1 GPU with CUDA capability >= 6.0.

## V2 Shallow-Wide Experiments

Based on timing analysis showing depth has weak correlation (r=0.37) while width has moderate correlation (r=0.60) with inference time, V2 uses shallow (1-2 layers) but wide (256-1024 neurons) architectures.

### V2 Submission File
Located in `../cluster/submit_v2_shallow_wide.sub`

### V2 Model Configurations

| Model | Architecture | Layers | Width | Epochs |
|-------|--------------|--------|-------|--------|
| mlp_v2_single_256 | MLP | 1 | 256 | 20 |
| mlp_v2_single_512 | MLP | 1 | 512 | 20 |
| mlp_v2_single_1024 | MLP | 1 | 1024 | 20 |
| mlp_v2_shallow_256 | MLP | 2 | 256-256 | 20 |
| mlp_v2_shallow_512_256 | MLP | 2 | 512-256 | 20 |
| mlp_v2_shallow_512 | MLP | 2 | 512-512 | 20 |
| mlp_v2_shallow_1024_256 | MLP | 2 | 1024-256 | 20 |
| mlp_v2_shallow_1024_512 | MLP | 2 | 1024-512 | 20 |
| pinn_v2_single_256 | PINN | 1 | 256 | 20 |
| pinn_v2_single_512 | PINN | 1 | 512 | 20 |
| pinn_v2_single_1024 | PINN | 1 | 1024 | 20 |
| pinn_v2_shallow_256 | PINN | 2 | 256-256 | 20 |
| pinn_v2_shallow_512_256 | PINN | 2 | 512-256 | 20 |
| pinn_v2_shallow_512 | PINN | 2 | 512-512 | 20 |
| pinn_v2_shallow_1024_256 | PINN | 2 | 1024-256 | 20 |
| pinn_v2_shallow_1024_512 | PINN | 2 | 1024-512 | 20 |
| rkpinn_v2_single_256 | RK_PINN | 1 | 256 | 20 |
| rkpinn_v2_single_512 | RK_PINN | 1 | 512 | 20 |
| rkpinn_v2_shallow_256 | RK_PINN | 2 | 256-256 | 20 |
| rkpinn_v2_shallow_512_256 | RK_PINN | 2 | 512-256 | 20 |
| rkpinn_v2_shallow_512 | RK_PINN | 2 | 512-512 | 20 |
| rkpinn_v2_shallow_1024_256 | RK_PINN | 2 | 1024-256 | 20 |

### V2 Design Rationale

From timing correlation analysis:
- **Depth ↔ Time**: r = 0.371 (weak) → Keep shallow
- **Width ↔ Time**: r = 0.596 (moderate) → Wide is okay
- **Params ↔ Time**: r = 0.829 (strong) → Limit total params

V2 targets inference time < 2.0 μs/track while maintaining accuracy.
