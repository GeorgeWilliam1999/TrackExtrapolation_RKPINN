#!/bin/bash
# V3 Data Generation Pipeline Monitor
#
# This script monitors the distributed data generation and runs the next steps.
#
# Usage:
#   ./monitor_datagen.sh
#
# Steps:
# 1. Wait for trajectory generation (100 jobs) to complete
# 2. Merge trajectory chunks
# 3. Submit MLP sample extraction (10 jobs)
# 4. Submit PINN sample extraction (4 jobs)
# 5. Wait for extraction to complete
# 6. Merge MLP samples

set -e

cd "$(dirname "$0")/.."  # Go to V3 directory

TRAJ_CLUSTER=${1:-3935225}  # Pass cluster ID as argument

echo "=============================================================="
echo "V3 Data Generation Pipeline Monitor"
echo "=============================================================="
echo "Trajectory generation cluster: $TRAJ_CLUSTER"
echo ""

# Function to wait for jobs
wait_for_jobs() {
    local cluster=$1
    local name=$2
    echo "Waiting for $name jobs (cluster $cluster)..."
    
    while true; do
        # Count running/idle jobs
        status=$(condor_q $cluster 2>/dev/null | grep "Total for query" || echo "0 jobs")
        
        if [[ "$status" == *"0 jobs"* ]] || [[ -z "$(condor_q $cluster 2>/dev/null | grep $cluster)" ]]; then
            echo "  ✓ All $name jobs completed!"
            break
        fi
        
        # Extract counts
        echo "  Status: $status"
        sleep 60
    done
}

# Step 1: Wait for trajectory generation
echo ""
echo "[Step 1] Waiting for trajectory generation..."
wait_for_jobs $TRAJ_CLUSTER "trajectory"

# Step 2: Merge trajectories
echo ""
echo "[Step 2] Merging trajectory chunks..."
n_chunks=$(ls data/chunks/trajectories_*.npz 2>/dev/null | wc -l)
echo "  Found $n_chunks trajectory chunks"

if [ $n_chunks -lt 90 ]; then
    echo "  ERROR: Expected ~100 chunks, found $n_chunks. Some jobs may have failed."
    echo "  Check logs in cluster/logs/"
    exit 1
fi

python data_generation/merge_trajectories.py \
    --input_dir data/chunks \
    --output data/trajectories_10k.npz

# Step 3: Submit MLP extraction
echo ""
echo "[Step 3] Submitting MLP sample extraction..."
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation
MLP_CLUSTER=$(condor_submit V3/cluster/submit_mlp_extract_distributed.sub | grep "submitted to cluster" | awk '{print $NF}' | tr -d '.')
echo "  Submitted MLP extraction cluster: $MLP_CLUSTER"

# Step 4: Submit PINN extraction
echo ""
echo "[Step 4] Submitting PINN sample extraction..."
PINN_CLUSTER=$(condor_submit V3/cluster/submit_pinn_extract.sub | grep "submitted to cluster" | awk '{print $NF}' | tr -d '.')
echo "  Submitted PINN extraction cluster: $PINN_CLUSTER"

# Step 5: Wait for extractions
echo ""
echo "[Step 5] Waiting for sample extraction..."
wait_for_jobs $MLP_CLUSTER "MLP extraction"
wait_for_jobs $PINN_CLUSTER "PINN extraction"

# Step 6: Merge MLP samples
echo ""
echo "[Step 6] Merging MLP samples..."
cd V3
python data_generation/merge_samples.py \
    --input_dir data/chunks \
    --pattern "mlp_samples_*.npz" \
    --output data/training_mlp_v3.npz

# Summary
echo ""
echo "=============================================================="
echo "Data Generation Complete!"
echo "=============================================================="
echo ""
echo "Generated files:"
ls -lh data/*.npz
echo ""
echo "Next step: Submit training jobs"
echo "  condor_submit V3/cluster/submit_pinn_collocation_study.sub"
