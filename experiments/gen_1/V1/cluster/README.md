# V1 Cluster Jobs

HTCondor submission files for V1 experiments.

## Main Submission File

`submit_full_suite_gpu.sub` - Submits all V1 training jobs (30 models)

## Job Status

| Cluster ID | Jobs | Epochs | Status |
|------------|------|--------|--------|
| 3880473-3880501 | 29 | 10 | ✅ Complete |
| 3880818 | 30 | 10 | ✅ Complete |

## Usage

```bash
# Submit all V1 jobs
condor_submit submit_full_suite_gpu.sub

# Check status
condor_q

# View logs
tail -f logs/*.out
```

## Resource Requirements

- CPUs: 4
- Memory: 16GB
- GPUs: 1
- Runtime: ~2-4 hours per model
