# Track Extrapolation ML Experiments

This directory contains all machine learning experiments for neural network track extrapolation in LHCb.

---

## Directory Structure

```
experiments/
├── README.md                # This file
├── gen_1/                   # ML extrapolator experiments
│   ├── V1/                  # MLP architecture sweep (17 models trained)
│   │   ├── models/          # Architecture code, training script, checkpoints
│   │   ├── data_generation/ # Data generation scripts + datasets
│   │   ├── analysis/        # Results analysis, plots, exported .bin files
│   │   ├── utils/           # Magnetic field interpolation, RK4 propagator
│   │   ├── notes/           # LaTeX documents (protocol, field characterisation)
│   │   ├── mlruns/          # MLflow tracking data (43 runs)
│   │   └── training/        # HTCondor job submission (empty)
│   ├── deployment/          # C++ model export pipeline (export_to_cpp.py)
│   └── archive/             # Historical analysis from earlier work
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

### field_maps — Magnetic Field Approximation

Neural network approximation of the LHCb dipole magnetic field map. Separate from the track extrapolation task — this explores whether the field itself can be represented by a small NN for faster evaluation.

See [field_maps/README.md](field_maps/README.md) and [field_maps/field_nn/README.md](field_maps/field_nn/README.md).

---

*Last Updated: April 2026*
