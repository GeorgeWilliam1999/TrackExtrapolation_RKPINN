#!/bin/bash
# =============================================================================
# V3 Training Job Wrapper Script
# Executed on HTCondor worker node
# =============================================================================

MODEL_NAME="$1"

# Base directory
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation"
cd "$BASE_DIR"

echo "=============================================="
echo "V3 Training Job: $MODEL_NAME"
echo "=============================================="
echo "Hostname:   $(hostname)"
echo "Date:       $(date)"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Activate environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

# Check CUDA
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
nvidia-smi || echo "nvidia-smi not available"

# Run appropriate training based on model name
echo ""
echo "Starting training..."
echo ""

case "$MODEL_NAME" in
    mlp_v3_shallow_256)
        python V3/training/train_mlp.py \
            --data_path V3/data/training_mlp_v3_100M_v2.npz \
            --model mlp \
            --hidden_dims 256 256 128 \
            --activation silu \
            --batch_size 4096 \
            --epochs 100 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --patience 20 \
            --name mlp_v3_shallow_256 \
            --checkpoint_dir V3/trained_models \
            --tensorboard
        ;;
    mlp_v3_shallow_512)
        python V3/training/train_mlp.py \
            --data_path V3/data/training_mlp_v3_100M_v2.npz \
            --model mlp \
            --hidden_dims 512 512 256 \
            --activation silu \
            --batch_size 4096 \
            --epochs 100 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --patience 20 \
            --name mlp_v3_shallow_512 \
            --checkpoint_dir V3/trained_models \
            --tensorboard
        ;;
    mlp_v3_deep_128)
        python V3/training/train_mlp.py \
            --data_path V3/data/training_mlp_v3_100M_v2.npz \
            --model mlp \
            --hidden_dims 128 128 128 128 64 \
            --activation silu \
            --batch_size 4096 \
            --epochs 100 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --patience 20 \
            --name mlp_v3_deep_128 \
            --checkpoint_dir V3/trained_models \
            --tensorboard
        ;;
    mlp_v3_deep_256)
        python V3/training/train_mlp.py \
            --data_path V3/data/training_mlp_v3_100M_v2.npz \
            --model mlp \
            --hidden_dims 256 256 256 256 128 \
            --activation silu \
            --batch_size 4096 \
            --epochs 100 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --patience 20 \
            --name mlp_v3_deep_256 \
            --checkpoint_dir V3/trained_models \
            --tensorboard
        ;;
    pinn_v3_col5)
        python V3/training/train_pinn.py --config V3/training/configs/pinn_v3_res_256_col5.json
        ;;
    pinn_v3_col10)
        python V3/training/train_pinn.py --config V3/training/configs/pinn_v3_res_256_col10.json
        ;;
    pinn_v3_col20)
        python V3/training/train_pinn.py --config V3/training/configs/pinn_v3_res_256_col20.json
        ;;
    pinn_v3_col50)
        python V3/training/train_pinn.py --config V3/training/configs/pinn_v3_res_256_col50.json
        ;;
    *)
        echo "ERROR: Unknown model name: $MODEL_NAME"
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    echo "=============================================="
fi

exit $EXIT_CODE
