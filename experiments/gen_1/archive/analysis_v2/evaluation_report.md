# Model Evaluation Results

Generated: 2026-01-15 12:34:30

Total models evaluated: 21


## All Models Ranked by Position Error

| Rank | Model | Type | Params | Pos (μm) | Pos 95% (μm) | Speed (ms/1k) |
|------|-------|------|--------|----------|--------------|---------------|
| 1 | rkpinn_wide_v1 | rk_pinn | 533,012 | 9.2 | 21.6 | 19.14 |
| 2 | rkpinn_large_v1 | rk_pinn | 201,236 | 13.0 | 30.9 | 19.49 |
| 3 | rkpinn_medium_v1 | rk_pinn | 51,476 | 14.4 | 36.0 | 18.06 |
| 4 | mlp_tiny_v1 | mlp | 2,660 | 22.8 | 59.6 | 4.23 |
| 5 | rkpinn_wide_shallow_v1 | rk_pinn | 135,444 | 26.9 | 63.8 | 17.51 |
| 6 | mlp_xlarge_v1 | mlp | 430,980 | 27.4 | 62.2 | 14.01 |
| 7 | mlp_wide_shallow_v1 | mlp | 35,204 | 33.7 | 82.2 | 6.14 |
| 8 | rkpinn_small_v1 | rk_pinn | 34,964 | 36.4 | 84.3 | 15.73 |
| 9 | mlp_small_v1 | mlp | 9,412 | 53.4 | 130.3 | 5.92 |
| 10 | mlp_wide_v1 | mlp | 135,940 | 55.0 | 125.1 | 6.82 |
| 11 | mlp_deep_v1 | mlp | 58,948 | 57.8 | 130.6 | 13.65 |
| 12 | rkpinn_tiny_v1 | rk_pinn | 9,300 | 60.6 | 142.6 | 9.29 |
| 13 | rkpinn_balanced_v1 | rk_pinn | 114,068 | 62.1 | 143.2 | 19.10 |
| 14 | mlp_balanced_v1 | mlp | 57,316 | 72.3 | 156.2 | 8.93 |
| 15 | mlp_medium_v1 | mlp | 25,924 | 74.5 | 181.5 | 8.36 |
| 16 | mlp_narrow_deep_v1 | mlp | 15,140 | 75.4 | 177.1 | 11.50 |
| 17 | rkpinn_xlarge_v1 | rk_pinn | 531,220 | 84.6 | 198.9 | 23.94 |
| 18 | mlp_large_v1 | mlp | 100,996 | 101.7 | 239.9 | 8.97 |
| 19 | rkpinn_deep_v1 | rk_pinn | 84,500 | 150.7 | 362.0 | 23.79 |
| 20 | rkpinn_narrow_deep_v1 | rk_pinn | 21,780 | 246.7 | 541.0 | 16.43 |
| 21 | test_mlp_tiny | mlp | 2,660 | 529122.5 | 995377.3 | 4.10 |

## Best Model Per Type

| Type | Best Model | Position Error (μm) | Parameters |
|------|------------|---------------------|------------|
| rk_pinn | rkpinn_wide_v1 | 9.2 | 533,012 |
| mlp | mlp_tiny_v1 | 22.8 | 2,660 |

## Baseline Comparison

| Model | Position (μm) | vs Herab (760 μm) | vs BS3 (100 μm) | vs Target (10 μm) |
|-------|---------------|-------------------|-----------------|-------------------|
| rkpinn_wide_v1 | 9.2 | ✓ 83x better | ✓ 11x better | ✓ 1.1x better |
| rkpinn_large_v1 | 13.0 | ✓ 58x better | ✓ 8x better | ✗ 1.3x worse |
| rkpinn_medium_v1 | 14.4 | ✓ 53x better | ✓ 7x better | ✗ 1.4x worse |
| mlp_tiny_v1 | 22.8 | ✓ 33x better | ✓ 4x better | ✗ 2.3x worse |
| rkpinn_wide_shallow_v1 | 26.9 | ✓ 28x better | ✓ 4x better | ✗ 2.7x worse |
| mlp_xlarge_v1 | 27.4 | ✓ 28x better | ✓ 4x better | ✗ 2.7x worse |
| mlp_wide_shallow_v1 | 33.7 | ✓ 23x better | ✓ 3x better | ✗ 3.4x worse |
| rkpinn_small_v1 | 36.4 | ✓ 21x better | ✓ 3x better | ✗ 3.6x worse |
| mlp_small_v1 | 53.4 | ✓ 14x better | ✓ 2x better | ✗ 5.3x worse |
| mlp_wide_v1 | 55.0 | ✓ 14x better | ✓ 2x better | ✗ 5.5x worse |