# V2 Models

Neural network architectures for V2 experiments.

## V2 Improvements over V1

### 1. PINN Residual Architecture (New!)

Fixes the V1 PINN IC failure:

```
V1 (broken): Output = Network(x, y, tx, ty, q/p, z_frac)
V2 (fixed):  Output = IC + z_frac × Network(x, y, tx, ty, q/p, z_frac)
```

This ensures:
- At z_frac=0: Output = IC (exact initial condition)
- At z_frac=1: Output = IC + correction (final state)

### 2. Shallow-Wide Architecture

V1 deep networks underperformed. V2 uses shallow-wide:

| Architecture | Layers | Width | V1 Loss | V2 Loss |
|--------------|--------|-------|---------|---------|
| Deep-narrow | 4-5 | 64-128 | 0.0045 | - |
| Shallow-wide | 1-3 | 256-1024 | - | **0.0008** |

## Model Sizes (V2)

| Preset | Architecture | Parameters |
|--------|--------------|------------|
| single_256 | [256] | ~2.8K |
| single_512 | [512] | ~5.6K |
| single_1024 | [1024] | ~11K |
| shallow_256 | [256, 128] | ~36K |
| shallow_512 | [512, 256] | ~135K |
| shallow_1024_256 | [1024, 256] | ~270K |
| shallow_1024_512 | [1024, 512] | ~540K |

## Scripts

- `architectures.py` - Model class definitions (includes residual PINN)
- `train.py` - Training loop and utilities

## Usage

```bash
python train.py --model mlp --hidden_dims 512 512 --activation silu
python train.py --model pinn --hidden_dims 512 256 --lambda_pde 1.0
```

## ⚠️ Remaining Issue

V2 models still trained with **fixed dz=8000mm** - see V3 for variable dz support.
