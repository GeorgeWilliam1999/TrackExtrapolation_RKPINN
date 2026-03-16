# V2 Analysis

Analysis of V2 experiment results.

## Key Findings

### Shallow-Wide Performance

| Architecture | Layers | Neurons | Val Loss | Speedup |
|--------------|--------|---------|----------|---------|
| V1 medium | 3 | 256,256,128 | 0.00095 | 1.7× |
| V2 shallow_512 | 2 | 512,256 | **0.00078** | **1.7×** |
| V2 single_256 | 1 | 256 | 0.0012 | **3.0×** |

**Conclusion**: Shallow-wide (1-2 layers) outperforms deep (4-5 layers).

### PINN Residual Fix

| Model | IC Error (V1) | IC Error (V2) |
|-------|---------------|---------------|
| pinn_medium | 2500+ mm | **< 0.1 mm** |
| rkpinn_medium | 2400+ mm | **< 0.1 mm** |

**Conclusion**: Residual architecture: `Output = IC + z_frac × Correction` works!

### Best V2 Models

| Purpose | Model | Loss | Speed |
|---------|-------|------|-------|
| Best accuracy | mlp_v2_shallow_1024_512 | 0.00072 | 2.1μs |
| Balanced | mlp_v2_shallow_512 | 0.00078 | 1.5μs |
| Fastest | mlp_v2_single_256 | 0.0012 | 0.83μs |

## Remaining Issue

⚠️ All V2 models fail for dz ≠ 8000mm due to fixed dz training.

See [V3](../../V3/) for variable dz support.
