# V1 Analysis

Analysis scripts and results for V1 experiments.

## Key Findings

### Model Performance (V1)

| Model | Val Loss | Position Error | Status |
|-------|----------|----------------|--------|
| mlp_tiny | 0.00078 | ~0.5 mm | ✅ Works |
| mlp_medium | 0.00045 | ~0.3 mm | ✅ Works |
| pinn_medium | 0.0012 | ~1.0 mm | ⚠️ IC failure |
| rkpinn_medium | 0.0015 | ~1.2 mm | ⚠️ IC failure |

### PINN Failure Analysis

**Problem**: PINN/RK-PINN output constant values regardless of z_frac.

| z_frac | PINN Output (x) | Expected |
|--------|-----------------|----------|
| 0.0 | 2768 mm | 207 mm |
| 1.0 | 2752 mm | 1039 mm |

**Root Cause**: Network ignored z_frac input, learned direct mapping from initial to final state.

## Recommendations

1. Use MLP for deployment (PINN failed)
2. ❌ Cannot use for variable dz (see V3)
