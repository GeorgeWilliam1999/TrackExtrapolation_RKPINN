# V1 Trained Models

Trained model checkpoints from V1 experiments.

## ⚠️ V1 Limitations

1. **Fixed dz=8000mm**: Cannot generalize to variable step sizes
2. **PINN IC failure**: PINN/RK-PINN models don't satisfy initial conditions

## Model Inventory (53 total)

### MLP Models (15)
| Model | Architecture | Params | Notes |
|-------|--------------|--------|-------|
| `mlp_tiny_v1` | [64, 64] | 5K | |
| `mlp_small_v1` | [128, 128] | 20K | |
| `mlp_medium_v1` | [256, 256, 128] | 100K | |
| `mlp_large_v1` | [512, 512, 256, 128] | 500K | |
| `mlp_wide_v1` | [512, 512, 256, 128] | 500K | |
| `mlp_balanced_v1` | [256, 256, 128] | 100K | |
| `mlp_narrow_deep_v1` | [64, 64, 64, 64] | 17K | |
| `mlp_wide_shallow_v1` | [512, 256] | 135K | |
| `mlp_medium_low_p` | [256, 256, 128] | 100K | p < 5 GeV |
| `mlp_medium_mid_p` | [256, 256, 128] | 100K | 5-20 GeV |
| `mlp_medium_high_p` | [256, 256, 128] | 100K | p > 20 GeV |

### PINN Models (16)
| Model | λ_pde | Notes |
|-------|-------|-------|
| `pinn_standard_v1` | 1.0 | ⚠️ IC failure |
| `pinn_medium_data_only` | 0.0 | MLP-like |
| `pinn_medium_pde_weak` | 0.01 | |
| `pinn_medium_pde_strong` | 10.0 | |
| `pinn_medium_pde_dominant` | 1.0 | |

### RK-PINN Models (11)
| Model | n_coll | Notes |
|-------|--------|-------|
| `rkpinn_balanced_v1` | 10 | ⚠️ IC failure |
| `rkpinn_coll5_v1` | 5 | |
| `rkpinn_coll10_v1` | 10 | |
| `rkpinn_coll20_v1` | 20 | |

## Model Files

Each model directory contains:
```
model_name/
├── model.pt          # PyTorch checkpoint
├── config.json       # Training configuration
├── metadata.json     # Normalization statistics
└── results.json      # Training metrics
```

## Usage

```python
import torch
from models.architectures import create_model

# Load model
checkpoint = torch.load('mlp_medium_v1/model.pt')
model = create_model('mlp', [256, 256, 128])
model.load_state_dict(checkpoint['model_state_dict'])
```

## See Also

- [V2/trained_models](../../V2/trained_models/) - V2 shallow-wide models
