# V2 Cluster Jobs

HTCondor submission files for V2 experiments.

## Submission Files

| File | Purpose | Jobs |
|------|---------|------|
| `submit_v2_shallow_wide.sub` | Shallow-wide MLP/PINN | 22 |
| `submit_v2_pinn_retrain.sub` | PINN with residual architecture | 7 |
| `v2_jobs.txt` | Job configuration list | - |

## Job Status

| Cluster ID | Jobs | Epochs | Status |
|------------|------|--------|--------|
| 3891076 | 22 | 20 | ✅ Complete |
| 3933584 | 7 | 20 | ✅ Complete (PINN retrain) |

## Usage

```bash
# Submit V2 shallow-wide training
condor_submit submit_v2_shallow_wide.sub

# Submit PINN retrain with residual architecture
condor_submit submit_v2_pinn_retrain.sub

# Check status
condor_q
```

## Resource Requirements

- CPUs: 4
- Memory: 16-32GB
- GPUs: 1
- Runtime: ~2-4 hours per model
