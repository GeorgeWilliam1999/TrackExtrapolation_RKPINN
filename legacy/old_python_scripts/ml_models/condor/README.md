# HTCondor Job Submission for Track Extrapolation ML

This directory contains HTCondor submit files and scripts for parallel data generation and model training on the Nikhef STBC cluster.

## Overview

HTCondor is a distributed computing system that manages job scheduling across cluster nodes. We use it to:
1. Generate large training datasets in parallel
2. Train multiple model architectures simultaneously on GPU nodes

## Directory Structure

```
condor/
├── README.md                 # This file
├── generate_data.sub         # HTCondor submit file for data generation
├── generate_data.sh          # Wrapper script for data generation job
├── train_models.sub          # HTCondor submit file for production training
├── train_model.sh            # Wrapper script for production training job
├── train_analysis.sub        # HTCondor submit file for analysis models (NEW)
├── train_analysis_job.sh     # Wrapper script for analysis training (NEW)
└── logs/                     # Job output logs (created automatically)
    ├── *.out                 # Standard output
    ├── *.err                 # Standard error
    └── *.log                 # HTCondor log
```

## Quick Start for Analysis Models

```bash
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models/condor

# Submit 12 analysis training jobs (for model_investigation.ipynb)
condor_submit train_analysis.sub

# Monitor jobs
condor_q -nobatch

# Check progress
tail -f logs/analysis_arch_tiny_*.out
tail -f logs/analysis_pinn_*.out
```

Models saved to `../models/analysis/` with PyTorch state dicts (.pt) and metadata (.json).

## Prerequisites

1. **Conda environment** with required packages:
   ```bash
   conda activate TE
   ```

2. **HTCondor** installed and configured on cluster
   ```bash
   condor_version  # Check HTCondor installation
   condor_q        # Check job queue
   ```

3. **GPU nodes** available (for training jobs)
   ```bash
   condor_status -constraint 'CUDACapability >= 7.0'
   ```

## Usage

### 1. Data Generation

Generate training data in parallel using multiple CPU cores:

```bash
# Create logs directory
mkdir -p logs

# Submit data generation job
condor_submit generate_data.sub

# Monitor job progress
condor_q
watch -n 5 condor_q

# Check output
tail -f logs/data_gen_0.out
```

**Configuration Options** (`generate_data.sub`):
- `request_cpus = 8`: Number of CPU cores per job
- `request_memory = 8GB`: Memory allocation
- `queue 1`: Number of parallel jobs (increase for more data)

To generate more data, modify the submit file:
```bash
# Edit generate_data.sub
queue 5  # Generate 5x100K = 500K samples total
```

### 2. Model Training on GPU

Train multiple model architectures in parallel on GPU nodes:

```bash
# Submit training jobs
condor_submit train_models.sub

# Monitor GPU jobs
condor_q -constraint 'RequestGpus > 0'

# Check specific job output
tail -f logs/train_large_*.out
```

**Pre-configured Architectures** (`train_models.sub`):
- `large`: 512-512-256-128 (~837K parameters)
- `xlarge`: 1024-512-256-128 (~2.1M parameters)
- `deep`: 256-256-256-256-128 (~431K parameters)
- `ultra`: 1024-1024-512-256-128 (~3.1M parameters)
- `wide`: 2048-1024-512-256 (~5.2M parameters)

**Add Custom Architectures**:
Edit `train_models.sub` and add a line:
```bash
queue Model,HiddenLayers from (
    large,512-512-256-128
    custom,2048-2048-1024-512-256  # Your architecture
)
```

### 3. Monitoring Jobs

```bash
# List all your jobs
condor_q

# Detailed job status
condor_q -better-analyze <job_id>

# Job history
condor_history -limit 10

# Remove a job
condor_rm <job_id>

# Remove all your jobs
condor_rm <username>
```

### 4. Collecting Results

After jobs complete:

```bash
# Check generated data
ls -lh ../data/X_train*.npy

# Check trained models
ls -lh ../models/mlp_*_condor.bin

# View model metadata
cat ../models/mlp_large_condor_metadata.json | python -m json.tool

# Merge datasets (if multiple data generation jobs)
cd ../python
python -c "
import numpy as np
from glob import glob

X_parts = [np.load(f) for f in sorted(glob('../data/X_train_part*.npy'))]
Y_parts = [np.load(f) for f in sorted(glob('../data/Y_train_part*.npy'))]
P_parts = [np.load(f) for f in sorted(glob('../data/P_train_part*.npy'))]

X = np.vstack(X_parts)
Y = np.vstack(Y_parts)
P = np.concatenate(P_parts)

np.save('../data/X_train.npy', X)
np.save('../data/Y_train.npy', Y)
np.save('../data/P_train.npy', P)

print(f'Merged {len(X)} samples')
"
```

## Resource Requirements

### Data Generation Job
- **CPUs**: 8 cores (adjustable)
- **Memory**: 8 GB
- **Disk**: 2 GB
- **Time**: ~10-20 min for 100K samples

### Training Job (GPU)
- **GPUs**: 1x CUDA-capable (≥7.0)
- **CPUs**: 4 cores
- **Memory**: 16 GB
- **Disk**: 5 GB
- **Time**: ~30-60 min for 2000 epochs (depends on architecture)

## Troubleshooting

### Job Stays Idle
```bash
# Check why job is held
condor_q -better-analyze <job_id>

# Check available resources
condor_status -constraint 'CUDACapability >= 7.0'
```

### GPU Not Found
```bash
# Verify GPU requirement in submit file
requirements = (CUDACapability >= 7.0)

# Check if CUDA is loaded
nvidia-smi
```

### Environment Issues
```bash
# Ensure conda is activated in wrapper script
source /data/bfys/gscriven/conda/bin/activate TE

# Test environment manually
./train_model.sh large 512-512-256-128
```

### Out of Memory
```bash
# Increase memory request in .sub file
request_memory = 32GB

# Or reduce batch size in training script
--batch 256
```

## Advanced Usage

### Priority and Nice Values
```bash
# Lower priority (nice to other users)
+NiceUser = True

# Set custom priority
priority = 10
```

### Email Notifications
```bash
# Add to .sub file
notification = Complete
notify_user = your.email@nikhef.nl
```

### DAG Workflows
For complex workflows (generate data → train → evaluate):

```bash
# Create DAG file (workflow.dag)
JOB A generate_data.sub
JOB B train_models.sub
PARENT A CHILD B

# Submit DAG
condor_submit_dag workflow.dag
```

## Performance Tips

1. **Batch size**: Increase for better GPU utilization (512-2048)
2. **Workers**: Match to CPU cores for data loading
3. **Pinned memory**: Enabled in GPU training for faster transfers
4. **Mixed precision**: Can add for faster training (not yet implemented)

## HTCondor Documentation

- **Nikhef HTCondor Guide**: https://wiki.nikhef.nl/grid/HTCondor
- **Official HTCondor Manual**: https://htcondor.readthedocs.io/
- **Submit File Reference**: https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html

## Contact

For cluster-specific issues, contact Nikhef support:
- Email: helpdesk@nikhef.nl
- Wiki: https://wiki.nikhef.nl/

For code issues, see main repository README.
