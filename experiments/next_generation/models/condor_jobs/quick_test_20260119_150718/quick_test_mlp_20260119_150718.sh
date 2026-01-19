#!/bin/bash
# Training job: quick_test_mlp_20260119_150718
# Generated: 2026-01-19T15:07:18.977496

set -e  # Exit on error

echo "=========================================="
echo "Training Job: quick_test_mlp_20260119_150718"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Activate conda environment
source /data/bfys/gscriven/miniforge3/etc/profile.d/conda.sh
conda activate TE

# Set up environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Check GPU availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"

# Create output directory
mkdir -p /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/models/condor_jobs/quick_test_20260119_150718/checkpoints
mkdir -p /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/models/condor_jobs/quick_test_20260119_150718/logs

# Run training
echo ""
echo "Starting training..."
echo ""

python /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/models/train.py \
    --model mlp \
    --preset tiny \
    --name quick_test_mlp_20260119_150718 \
    --epochs 10 \
    --data_path ../data_generation/data/batch_0.npz \
    --checkpoint_dir /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/models/condor_jobs/quick_test_20260119_150718/checkpoints \
    --max_samples 10000 \
    --patience 5 \
    --min_delta 1e-07

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
