# Weighted Loss Training Experiment

## Objective

Improve low-momentum track extrapolation accuracy by weighting the loss function inversely proportional to momentum.

## Motivation

Analysis shows that errors are dominated by low-momentum tracks (p < 5 GeV):
- Low-p tracks bend sharply in the magnetic field
- The MLP struggles to capture this non-linear behavior
- Standard MSE loss treats all samples equally, but low-p tracks are harder

## Method

We weight each sample's contribution to the loss by a function of momentum:

```
L_weighted = (1/N) * Σ w_i * MSE_i
```

Where `w_i` is computed from momentum `p_i`:

| Weight Type | Formula | Description |
|-------------|---------|-------------|
| uniform | w = 1 | Standard MSE (baseline) |
| inverse_p_0.5 | w = 1/√p | Mild upweighting of low-p |
| inverse_p_1.0 | w = 1/p | Linear inverse weighting |
| inverse_p_2.0 | w = 1/p² | Strong upweighting of low-p |
| log_p | w = 1/(1+log(p)) | Logarithmic weighting |

## Results

### Experiment 1: power=2.0 (2024-12-21)
**Result: WORSE performance across all momentum bins**

| Momentum Bin | Original MLP | Weighted MLP | Change |
|--------------|--------------|--------------|--------|
| 2-5 GeV | 1.01 mm | 2.72 mm | -170% |
| 5-10 GeV | 0.68 mm | 1.68 mm | -148% |
| 10-20 GeV | 0.60 mm | 1.81 mm | -203% |
| 20-50 GeV | 0.59 mm | 1.89 mm | -221% |
| 50-100 GeV | 0.57 mm | 1.80 mm | -217% |
| **OVERALL** | **0.69 mm** | **2.00 mm** | **-189%** |

**Conclusion**: Power=2 is too aggressive and destabilizes training.

## Key Insights

1. The original MLP already generalizes well to low momentum
2. Over-weighting low-p samples causes underfitting at high-p without improving low-p
3. The physics-informed approach (PINN) is more effective than weighted data loss

## Recommendations

1. Try milder weighting (power=0.5 or 1.0)
2. Consider curriculum learning (train high-p first, then gradually add low-p)
3. Use separate models for different momentum ranges
4. The PINN approach shows better low-p performance (1.28 mm vs 1.48 mm MLP)

## Files

- `train_weighted_v2.py` - Training script
- `mlp_weighted.bin` - Weighted loss model

## Notes

- Training data: 15,000 samples, log-uniform momentum 2-100 GeV
- Architecture: 6→256→256→128→64→4 MLP with tanh
- Epochs: 500 with cosine annealing LR
