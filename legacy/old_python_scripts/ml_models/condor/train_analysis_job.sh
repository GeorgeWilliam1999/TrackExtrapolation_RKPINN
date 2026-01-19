#!/bin/bash
#
# Wrapper script for training analysis models on HTCondor
#
# Usage: ./train_analysis_job.sh <job_type> <params>
#   job_type: arch, pinn, or activation
#   params: name:hidden_layers or weight:hidden_layers

set -e

JOB_TYPE=$1
PARAMS=$2
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models"
PYTHON_DIR="${BASE_DIR}/python"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${BASE_DIR}/models/analysis"

# Activate conda environment
source /data/bfys/gscriven/conda/bin/activate TE

echo "=========================================="
echo "Analysis Model Training Job"
echo "=========================================="
echo "Job Type: ${JOB_TYPE}"
echo "Params: ${PARAMS}"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"
cd "${PYTHON_DIR}"

# Parse parameters
NAME=$(echo ${PARAMS} | cut -d: -f1)
HIDDEN=$(echo ${PARAMS} | cut -d: -f2 | tr '-' ' ')

case ${JOB_TYPE} in
    arch)
        echo "Training architecture: ${NAME}"
        python train_on_gpu.py \
            --data "${DATA_DIR}" \
            --dataset analysis \
            --hidden ${HIDDEN} \
            --epochs 1000 \
            --batch 256 \
            --lr 0.001 \
            --output "${OUTPUT_DIR}" \
            --name "mlp_${NAME}"
        ;;
    
    pinn)
        echo "Training PINN with λ=${NAME}"
        python -c "
import sys
sys.path.insert(0, '.')
from train_analysis_models import *

# Load data
X = np.load('${DATA_DIR}/X_analysis.npy')
Y = np.load('${DATA_DIR}/Y_analysis.npy')
P = np.load('${DATA_DIR}/P_analysis.npy')

np.random.seed(42)
n = len(X)
idx = np.random.permutation(n)
n_test, n_val = int(n*0.05), int(n*0.1)
X_train, Y_train = X[idx[n_test+n_val:]], Y[idx[n_test+n_val:]]
X_val, Y_val = X[idx[n_test:n_test+n_val]], Y[idx[n_test:n_test+n_val]]
X_test, Y_test, P_test = X[idx[:n_test]], Y[idx[:n_test]], P[idx[:n_test]]

hidden = [${HIDDEN// /, }]
model = TrackPINN(hidden).to(device)
model, history = train_model(model, X_train, Y_train, X_val, Y_val, 
                             epochs=1000, use_physics_loss=True, physics_weight=${NAME})
metrics, _, _ = evaluate_model(model, X_test, Y_test, P_test)
save_model(model, history, metrics, 'pinn_lambda_${NAME//./_}', '${OUTPUT_DIR}')
print(f'Mean error: {metrics[\"pos_mean\"]:.2f} mm')
"
        ;;
    
    activation)
        echo "Training with ${NAME} activation"
        python -c "
import sys
sys.path.insert(0, '.')
from train_analysis_models import *

# Load data
X = np.load('${DATA_DIR}/X_analysis.npy')
Y = np.load('${DATA_DIR}/Y_analysis.npy')
P = np.load('${DATA_DIR}/P_analysis.npy')

np.random.seed(42)
n = len(X)
idx = np.random.permutation(n)
n_test, n_val = int(n*0.05), int(n*0.1)
X_train, Y_train = X[idx[n_test+n_val:]], Y[idx[n_test+n_val:]]
X_val, Y_val = X[idx[n_test:n_test+n_val]], Y[idx[n_test:n_test+n_val]]
X_test, Y_test, P_test = X[idx[:n_test]], Y[idx[:n_test]], P[idx[:n_test]]

hidden = [${HIDDEN// /, }]
model = TrackMLP(hidden, activation='${NAME}').to(device)
model, history = train_model(model, X_train, Y_train, X_val, Y_val, epochs=1000)
metrics, _, _ = evaluate_model(model, X_test, Y_test, P_test)
save_model(model, history, metrics, 'mlp_act_${NAME}', '${OUTPUT_DIR}')
print(f'Mean error: {metrics[\"pos_mean\"]:.2f} mm')
"
        ;;
esac

echo "=========================================="
echo "Job complete!"
echo "=========================================="
