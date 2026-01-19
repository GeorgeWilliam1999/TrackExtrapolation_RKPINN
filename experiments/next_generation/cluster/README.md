# HTCondor Cluster Training

Guide for training neural network track extrapolators on the NIKHEF stoomboot cluster.

## Quick Start

```bash
# Submit all training jobs (30 models)
condor_submit submit_full_suite_gpu.sub

# Monitor jobs
condor_q

# Check specific job
condor_q -analyze <JOB_ID>
```

## GPU Resources

NIKHEF has excellent GPU resources available:

| Node | GPUs | Type | VRAM |
|------|------|------|------|
| wn-lot-008/009 | 2 each | Tesla V100-PCIE | 32 GB |
| wn-pijl-002/003 | 2 each | NVIDIA L40S | 45 GB |
| wn-pijl-004-007 | 4 each | NVIDIA L40S | 45 GB |

**Total: 22 GPUs** - Request with `request_gpus = 1`

## Model Types

We train **three** distinct architectures:

| Model | Jobs | Description | Key Parameter |
|-------|------|-------------|---------------|
| **MLP** | 10 | Pure data loss, architecture sweep | N/A |
| **PINN** | 10 | Autodiff PDE residual loss | `--lambda_pde` |
| **RK-PINN** | 10 | Collocation-based physics | `--n_collocation`, `--lambda_pde` |

**Total: 30 jobs**

## Files

- `submit_full_suite_gpu.sub` - Main HTCondor job definitions (30 jobs)
- `submit_training.sub` - Single job template
- `submit_training_gpu.sub` - GPU job template  
- `run_training.sh` - Script executed on worker nodes
- `monitor_training.sh` - Check job status
- `logs/` - Job output and error logs

## Customizing Jobs

Edit `submit_full_suite_gpu.sub` to add experiments:

```
# Arguments: MODEL_TYPE PRESET EXPERIMENT_NAME [EXTRA_ARGS...]
arguments = mlp wide my_experiment --epochs 200 --lr 1e-4
queue 1
```

### Available Options

```bash
python train.py --help

--model {mlp,pinn,rk_pinn}  # Model architecture
--preset {tiny,small,medium,large,xlarge,wide,deep}
--epochs N                   # Number of epochs
--lr FLOAT                   # Learning rate
--batch_size N               # Batch size
--lambda_pde FLOAT           # Physics loss weight (PINN/RK-PINN only)
--n_collocation N            # Collocation points (RK-PINN only)
--activation {silu,relu,tanh,gelu}
--dropout FLOAT
--hidden_dims INT [INT ...]  # Custom architecture
```

### Example Experiments

```bash
# MLP with custom architecture
arguments = mlp medium my_mlp --hidden_dims 256 256 128 --epochs 150

# PINN with strong physics constraint
arguments = pinn large my_pinn --lambda_pde 0.1 --epochs 100

# RK-PINN with many collocation points
arguments = rk_pinn medium my_rkpinn --n_collocation 20 --lambda_pde 1e-3
```

## Monitoring

```bash
# All your jobs
condor_q

# Detailed status
condor_q -analyze

# Job history
condor_history -limit 20

# Watch logs in real-time
tail -f logs/gpu_suite_*.out

# Check GPU utilization on a node
ssh wn-pijl-002 nvidia-smi
```

## Collecting Results

After training completes:

```bash
cd ../models
python evaluate_all_models.py
```

Results will be in `../analysis/results/`.

## Troubleshooting

### Job stuck in queue
```bash
condor_q -analyze <JOB_ID>
```
Usually means no GPU available - jobs will start when resources free up.

### Out of memory
Reduce batch size: `--batch_size 4096` or `--batch_size 2048`

### CUDA errors
Check if GPU is actually available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Permission denied
Ensure the data and model directories are accessible from worker nodes (they're on NFS).
