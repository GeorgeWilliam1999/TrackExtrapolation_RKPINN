# V2 Data Generation

Data generation scripts for V2 experiments.

## ⚠️ V2 Limitation

**V2 data still uses fixed `dz = 8000 mm`** - same limitation as V1.

## Data Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **dz** | 8000 mm | **FIXED** - same as V1 |
| **z_start** | 0 mm | Starting position |
| **z_end** | 8000 mm | Ending position |
| **Samples** | 50M | Same dataset as V1 |
| **Field** | twodip.rtf | Real LHCb dipole map |

## Note

V2 used the **same training data as V1**. The improvement in V2 was in the 
**architecture** (residual PINN, shallow-wide), not the data.

## See Also

- [V3/data_generation](../../V3/data_generation/) - Variable dz data generation (recommended)
