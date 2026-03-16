# V2 Results

Training results for V2 shallow-wide experiments.

## Files

- `v2_model_results.csv` - Complete training metrics for 22 V2 models
- `v2_pareto_models.csv` - Pareto-optimal models (speed vs accuracy)

## Best V2 Models

| Model | Val Loss | Time | Speedup | Notes |
|-------|----------|------|---------|-------|
| mlp_v2_shallow_1024_512 | 0.00072 | 2.1μs | 1.2× | Best accuracy |
| mlp_v2_shallow_512 | 0.00078 | 1.5μs | 1.7× | Balanced |
| mlp_v2_single_256 | 0.0012 | 0.83μs | **3.0×** | Fastest |

## Improvement over V1

V2 shallow-wide (2 layers) outperforms V1 deep networks (4-5 layers):
- ~15% better validation loss
- Similar or faster inference
