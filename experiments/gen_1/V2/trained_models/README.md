# V2 Trained Models

Trained model checkpoints from V2 experiments.

## V2 Improvements

1. ✅ **PINN IC fixed**: Residual architecture satisfies initial conditions
2. ✅ **Shallow-wide**: Better accuracy than V1 deep networks
3. ❌ **Still fixed dz=8000mm**: Cannot generalize to variable step sizes

## Model Inventory (22 total)

### MLP V2 Models (9)
| Model | Architecture | Params | Val Loss |
|-------|--------------|--------|----------|
| `mlp_v2_single_256` | [256] | 2.8K | 0.0012 |
| `mlp_v2_single_512` | [512] | 5.6K | 0.0010 |
| `mlp_v2_single_1024` | [1024] | 11K | 0.0009 |
| `mlp_v2_shallow_256` | [256, 128] | 36K | 0.00085 |
| `mlp_v2_shallow_512` | [512, 256] | 135K | **0.00078** |
| `mlp_v2_shallow_512_256` | [512, 256] | 135K | 0.00080 |
| `mlp_v2_shallow_1024_256` | [1024, 256] | 270K | 0.00075 |
| `mlp_v2_shallow_1024_512` | [1024, 512] | 540K | 0.00072 |

### PINN V2 Models (7)
| Model | Architecture | λ_pde | Notes |
|-------|--------------|-------|-------|
| `pinn_v2_single_256` | [256] | 1.0 | Residual |
| `pinn_v2_single_512` | [512] | 1.0 | Residual |
| `pinn_v2_shallow_256` | [256, 128] | 1.0 | Residual |
| `pinn_v2_shallow_512` | [512, 256] | 1.0 | Residual |
| `pinn_v2_shallow_1024_256` | [1024, 256] | 1.0 | Residual |
| `pinn_v2_shallow_1024_512` | [1024, 512] | 1.0 | Residual |

### RK-PINN V2 Models (6)
| Model | Architecture | n_coll | Notes |
|-------|--------------|--------|-------|
| `rkpinn_v2_single_256` | [256] | 10 | Residual |
| `rkpinn_v2_single_512` | [512] | 10 | Residual |
| `rkpinn_v2_shallow_256` | [256, 128] | 10 | Residual |
| `rkpinn_v2_shallow_512` | [512, 256] | 10 | Residual |
| `rkpinn_v2_shallow_1024_256` | [1024, 256] | 10 | Residual |

## Best Models

| Metric | Model | Value |
|--------|-------|-------|
| Lowest loss | `mlp_v2_shallow_1024_512` | 0.00072 |
| Best speed/accuracy | `mlp_v2_shallow_512` | 0.00078, 1.5μs |
| Fastest | `mlp_v2_single_256` | 0.83μs |

## Model Files

Each model directory contains:
```
model_name/
├── model.pt          # PyTorch checkpoint
├── config.json       # Training configuration
├── metadata.json     # Normalization statistics ⚠️ std[dz]≈0!
└── results.json      # Training metrics
```

## ⚠️ Deployment Warning

V2 models have `input_std[dz] ≈ 1e-9` which causes normalization explosion
for dz ≠ 8000. See [V3](../../V3/) for models with variable dz support.

## See Also

- [V3/trained_models](../../V3/trained_models/) - V3 models (coming soon)
