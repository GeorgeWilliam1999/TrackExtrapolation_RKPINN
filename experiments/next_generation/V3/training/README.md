# V3 Training

Training scripts and configurations for V3 models with variable dz support.

## Quick Start

### 1. Generate Training Data First

Before training, generate V3 data with variable dz:

```bash
# Submit data generation job
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation
condor_submit V3/cluster/submit_datagen_v3.sub

# Monitor job
condor_q

# Data will be saved to:
# V3/data_generation/data/training_v3_100M.npz
```

### 2. Train Models

**Local (single model):**
```bash
python models/train.py \
    --config V3/training/configs/mlp_v3_shallow_256.json
```

**Cluster (all models):**
```bash
condor_submit V3/cluster/submit_v3_training.sub
```

## Available Configurations

### MLP Models
| Config | Architecture | Parameters | Description |
|--------|-------------|------------|-------------|
| `mlp_v3_shallow_256` | [256, 256, 128] | ~100K | Baseline shallow-wide |
| `mlp_v3_shallow_512` | [512, 512, 256] | ~400K | Larger capacity |
| `mlp_v3_deep_128` | [128, 128, 128, 128, 64] | ~75K | Deep narrow |
| `mlp_v3_deep_256` | [256, 256, 256, 128, 64] | ~200K | Deep wide |

### PINN Models (Residual Architecture)
| Config | Architecture | λ_pde | λ_ic | Description |
|--------|-------------|-------|------|-------------|
| `pinn_v3_shallow_256` | [256, 256, 128] | 1.0 | 1.0 | PINN with residual IC |
| `pinn_v3_deep_128` | [128, 128, 128, 128, 64] | 1.0 | 1.0 | Deep PINN |

### RK-PINN Models
| Config | Architecture | n_coll | Description |
|--------|-------------|--------|-------------|
| `rkpinn_v3_shallow_256` | [256, 256, 128] | 10 | RK4-inspired loss |
| `rkpinn_v3_deep_128` | [128, 128, 128, 128, 64] | 10 | Deep RK-PINN |

## Configuration Format

Example JSON config (`mlp_v3_shallow_256.json`):
```json
{
    "name": "mlp_v3_shallow_256",
    "description": "V3 MLP - variable dz support",
    
    "model_type": "mlp",
    "hidden_dims": [256, 256, 128],
    "activation": "silu",
    
    "data_path": ".../V3/data_generation/data/training_v3_100M.npz",
    "train_fraction": 0.8,
    "val_fraction": 0.1,
    
    "batch_size": 4096,
    "epochs": 100,
    "learning_rate": 0.001,
    "scheduler": "cosine",
    
    "patience": 20,
    "checkpoint_dir": ".../trained_models"
}
```

## Training Output

Each trained model saves:
```
trained_models/mlp_v3_shallow_256/
├── model.pt              # PyTorch checkpoint
├── config.json           # Training config
├── metadata.json         # Normalization stats (IMPORTANT!)
├── results.json          # Final metrics
└── tensorboard/          # Training logs
```

### Critical: Normalization Statistics

The `metadata.json` contains normalization parameters:
```json
{
    "input_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 6250.0],
    "input_std": [300.0, 200.0, 0.15, 0.12, 0.05, 3300.0],
    "output_mean": [0.0, 0.0, 0.0, 0.0],
    "output_std": [500.0, 350.0, 0.2, 0.15]
}
```

**V3 Key Difference**: `input_std[5]` (dz) should be ~3300, NOT ~1e-9!

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir trained_models/mlp_v3_shallow_256/tensorboard
```

### Cluster Jobs
```bash
# Check job status
condor_q

# View job output
tail -f V3/cluster/logs/train_mlp_v3_shallow_256_*.out
```

## After Training: Deployment

1. **Export to C++ binary**:
```bash
python deployment/export_to_cpp.py \
    trained_models/mlp_v3_shallow_256 \
    ml_models/models/mlp_v3_shallow_256.bin
```

2. **Run QMTest**:
```bash
cd /data/bfys/gscriven/TE_stack
make fast/Rec/test ARGS='-R extrapolators -V'
```

## Directory Structure
```
V3/training/
├── README.md (this file)
├── train_v3_wrapper.sh     # Cluster job wrapper
└── configs/
    ├── mlp_v3_shallow_256.json
    ├── mlp_v3_shallow_512.json
    ├── mlp_v3_deep_128.json
    ├── mlp_v3_deep_256.json
    ├── pinn_v3_shallow_256.json
    ├── pinn_v3_deep_128.json
    ├── rkpinn_v3_shallow_256.json
    └── rkpinn_v3_deep_128.json
```
