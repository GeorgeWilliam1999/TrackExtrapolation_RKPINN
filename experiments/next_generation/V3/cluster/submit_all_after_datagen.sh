#!/bin/bash
#===============================================================================
# Submit all training jobs after data generation completes
# Run this after data_gen job finishes successfully
#===============================================================================

cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/V3/cluster

# Verify data exists
echo "Checking for v2 data files..."
for f in ../data/training_mlp_v3_100M_v2.npz \
         ../data/training_pinn_v3_col5_v2.npz \
         ../data/training_pinn_v3_col10_v2.npz \
         ../data/training_pinn_v3_col20_v2.npz \
         ../data/training_pinn_v3_col50_v2.npz; do
    if [ -f "$f" ]; then
        echo "  ✓ $(basename $f) ($(ls -lh $f | awk '{print $5}'))"
    else
        echo "  ✗ $(basename $f) NOT FOUND!"
        echo "ERROR: Data generation not complete. Run submit_data_gen.sub first."
        exit 1
    fi
done

echo ""
echo "All data files present. Submitting training jobs..."
echo ""

# Clean old models
echo "Cleaning old model directories..."
rm -rf ../trained_models/mlp_v3_*
rm -rf ../trained_models/pinn_v3_*

# Submit MLP jobs
echo "Submitting MLP training jobs..."
condor_submit submit_shallow_256.sub
condor_submit submit_shallow_512.sub
condor_submit submit_deep_128.sub
condor_submit submit_deep_256.sub

# Submit PINN jobs
echo "Submitting PINN training jobs..."
condor_submit submit_pinn_col5.sub
condor_submit submit_pinn_col10.sub
condor_submit submit_pinn_col20.sub
condor_submit submit_pinn_col50.sub

echo ""
echo "All jobs submitted. Monitor with: condor_q gscriven"
