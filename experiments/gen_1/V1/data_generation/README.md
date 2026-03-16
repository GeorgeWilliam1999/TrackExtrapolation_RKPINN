# V1 Data Generation

Data generation scripts for V1 experiments.

## ⚠️ V1 Limitation

**All V1 data was generated with fixed `dz = 8000 mm`**. This is a critical limitation
that was discovered during C++ deployment - models cannot generalize to other step sizes.

## Data Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **dz** | 8000 mm | **FIXED** - major limitation |
| **z_start** | 0 mm | Starting position |
| **z_end** | 8000 mm | Ending position |
| **Samples** | 50M | Full training dataset |
| **Field** | twodip.rtf | Real LHCb dipole map |

## Input/Output Format

```
Input X:  [x, y, tx, ty, q/p, dz]  -> 6 features
          dz = 8000 for ALL samples

Output Y: [x, y, tx, ty]           -> 4 features (at z_end)
```

## Scripts

### `generate_data.py`

Main data generation script using Python RK4 integration.

```bash
python generate_data.py \
    --n-tracks 1000000 \
    --z-start 0 \
    --z-end 8000 \
    --output data/training_1M.npz
```

## Normalization Statistics (V1)

```
Feature    Mean        Std         Issue
------------------------------------------
x          ~0.0        ~300 mm     OK
y          ~0.0        ~200 mm     OK  
tx         ~0.0        ~0.15       OK
ty         ~0.0        ~0.12       OK
q/p        ~0.0        ~0.05       OK
dz         8000.0      ~1e-9       ⚠️ PROBLEM!
```

The near-zero `std[dz]` causes normalization explosion for dz ≠ 8000.

## See Also

- [V3/data_generation](../../V3/data_generation/) - Variable dz data generation (recommended)
