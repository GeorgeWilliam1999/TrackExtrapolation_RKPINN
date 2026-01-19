# Model Comparison Results

Generated: 2026-01-15 12:10:47

## All Models Ranked by Position Error

| Rank | Model | Type | Architecture | Pos Error (mm) | Pos 95% (mm) | Train Time |
|------|-------|------|--------------|----------------|--------------|------------|
| 1 | rkpinn_wide_v1 | rk_pinn | 512-256 | 0.0092 | 0.0216 | 992.8m |
| 2 | rkpinn_large_v1 | rk_pinn | 256-256-128 | 0.0130 | 0.0308 | 964.5m |
| 3 | rkpinn_medium_v1 | rk_pinn | 128-128-64 | 0.0143 | 0.0355 | 969.9m |
| 4 | test_mlp_tiny | mlp | 64-32 | 532.2219 | 1005.8702 | 0.0m |

## Baseline Comparison

| Model | Position Error (mm) | vs Herab | vs BS3 |
|-------|---------------------|----------|--------|
| rkpinn_wide_v1 | 0.0092 | ✓ Better | ✓ Better |
| rkpinn_large_v1 | 0.0130 | ✓ Better | ✓ Better |
| rkpinn_medium_v1 | 0.0143 | ✓ Better | ✓ Better |
| test_mlp_tiny | 532.2219 | ✗ Worse | ✗ Worse |