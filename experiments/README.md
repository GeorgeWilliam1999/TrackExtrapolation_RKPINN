# Track Extrapolation ML Experiments

This directory contains all machine learning experiments for neural network track extrapolation in LHCb.

---

## Directory Structure

```
experiments/
├── README.md                # This file
├── gen_1/                   # First-generation experiments
│   ├── V1/                  # MLP architecture sweep (17 models trained)
│   │   ├── models/          # Architecture code, training script, checkpoints
│   │   ├── data_generation/ # Data generation scripts + 50M-track dataset
│   │   ├── analysis/        # Results analysis, plots, exported .bin files
│   │   ├── utils/           # Magnetic field interpolation, RK4 propagator
│   │   ├── notes/           # LaTeX documents (protocol, field characterisation)
│   │   ├── mlruns/          # MLflow tracking data (43 runs)
│   │   └── training/        # HTCondor job submission (empty)
│   ├── deployment/          # C++ model export pipeline (export_to_cpp.py)
│   └── archive/             # Historical analysis from earlier work
│
├── gen_2/                   # Second-generation: MLP + PINN + RK_PINN
│   ├── models/              # train.py (physics loss), architectures.py (symlink)
│   ├── utils/               # Symlink to gen_1/V1/utils/
│   ├── configs/             # JSON configs: mlp/ pinn/ rk_pinn/
│   ├── condor/              # HTCondor submit files (11 jobs)
│   ├── trained_models/      # Output directory
│   └── mlruns/              # MLflow tracking
│
└── field_maps/              # Magnetic field map experiments
    ├── nn_field_map_sizing.ipynb  # Field map NN sizing analysis
    ├── twodip.rtf                 # LHCb dipole field data
    └── field_nn/                  # NN field map training pipeline
```

---

## gen_1 — Neural Network Track Extrapolators

### V1 — MLP Architecture Sweep

17 MLP architectures trained on 5M samples (from 50M dataset), fixed dz = 8000 mm.  
Training tracked with MLflow (experiment: `V1_MLP_sweep`).

| Aspect | Detail |
|--------|--------|
| Architectures | [64,64] through [1024,512,256], all SiLU |
| Data | 50M tracks, fixed dz=8000mm, 80/10/10 split |
| Training | Cosine LR schedule, early stopping (patience 30), batch 4096 |
| Tracking | MLflow (43 runs total, 17 converged) |
| Code | `V1/models/train.py`, `V1/models/architectures.py` |

**Known issue:** V1 training did not set random seeds. Results are not bit-reproducible. Seed-setting has been added for future runs.

### gen_2 — MLP + PINN + RK_PINN Comparison

Systematic comparison of all three architecture types on the same 50M-track dataset
with variable dz \[100, 10000\] mm. Physics-informed loss for PINN/RK_PINN, Jacobian
evaluation, comprehensive MLflow + TensorBoard monitoring.

| Aspect | Detail |
|--------|--------|
| Models | MLP (5 sizes) + PINN (3 λ) + RK_PINN (3 λ) = 11 runs |
| Data | 50M tracks, variable dz, shared from gen_1 |
| Physics loss | Lorentz PDE residual at collocation points |
| Tracking | MLflow (`gen_2_track_extrapolation`) + TensorBoard |
| Jacobian | Autograd transport matrix evaluated on test set |
| Code | `gen_2/models/train.py`, configs in `gen_2/configs/` |

See [gen_2/README.md](gen_2/README.md) for full experiment plan.

### field_maps — Magnetic Field Approximation

Neural network approximation of the LHCb dipole magnetic field map. Separate from the track extrapolation task — this explores whether the field itself can be represented by a small NN for faster evaluation.

See [field_maps/README.md](field_maps/README.md) and [field_maps/field_nn/README.md](field_maps/field_nn/README.md).

---

*Last Updated: April 2026*
