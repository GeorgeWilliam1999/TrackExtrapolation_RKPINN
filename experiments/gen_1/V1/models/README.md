# V1 Models

Neural network architectures and training scripts for V1 experiments.

## ⚠️ V1 Issues

1. **PINN IC Failure**: PINN/RK-PINN architectures failed to learn z_frac dependence
2. **Fixed dz**: All models trained with dz=8000mm only

## Architectures

### MLP (Multi-Layer Perceptron)
Standard feedforward network: `Input[6] → Hidden → Output[4]`

### PINN (Physics-Informed Neural Network)
Adds physics loss from Lorentz force equations.
**Issue**: Failed to satisfy Initial Condition (IC) constraint.

### RK-PINN (Runge-Kutta PINN)
Multi-stage physics loss inspired by RK4 integration.
**Issue**: Same IC failure as PINN.

## Model Sizes

| Preset | Architecture | Parameters |
|--------|--------------|------------|
| tiny | [64, 64] | ~5K |
| small | [128, 128] | ~20K |
| medium | [256, 256, 128] | ~100K |
| wide | [512, 512, 256, 128] | ~500K |

## Scripts

- `architectures.py` - Model class definitions
- `train.py` - Training loop and utilities

## Usage

```bash
python train.py --model mlp --preset medium --epochs 100
python train.py --model pinn --preset medium --lambda-pde 1.0
```

## See Also

- [V2/models](../../V2/models/) - Residual PINN architecture (fixes IC issue)
