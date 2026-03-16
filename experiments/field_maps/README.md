# Magnetic Field Map Experiments

This directory contains experiments related to neural network approximation of the LHCb dipole magnetic field, plus the field map data file used across all training.

---

## Contents

```
field_maps/
├── README.md                      # This file
├── twodip.rtf                     # LHCb dipole field data (81×81×146 grid, 100 mm spacing)
├── nn_field_map_sizing.ipynb      # Notebook: NN sizing analysis for field approximation
└── field_nn/                      # Full NN field map training pipeline
    ├── README.md                  # Pipeline documentation
    ├── generate_configs.py        # Configuration generator
    ├── configs/                   # Training configurations
    ├── training/                  # Training scripts
    ├── cluster/                   # HTCondor job files
    ├── trained_models/            # Trained field map NNs
    └── analysis/                  # Evaluation and comparison
```

---

## Field Map Data (`twodip.rtf`)

The LHCb dipole field measurement data used by `InterpolatedFieldTorch` for all PINN/RK_PINN training:

- **Grid:** 81 × 81 × 146 points
- **Spacing:** 100 mm in each dimension
- **Components:** Full 3D field (Bx, By, Bz)
- **Peak field:** |By| ≈ 1.03 T at z ≈ 5007 mm

This file is also referenced by the track extrapolation experiments in `gen_1/`.

## NN Field Map Sizing (`nn_field_map_sizing.ipynb`)

Analysis notebook exploring whether the field map can be represented by a compact neural network for faster field evaluation during training and inference.

## Field NN Pipeline (`field_nn/`)

Complete training pipeline for neural network field map approximation. See [field_nn/README.md](field_nn/README.md) for details.

---

*Last Updated: March 2026*
