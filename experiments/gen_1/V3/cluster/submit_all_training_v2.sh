#!/bin/bash
#===============================================================================
# Submit all V3 training jobs after data extraction completes
#===============================================================================

cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/V3/cluster

echo "=============================================="
echo "Submitting all V3 training jobs"
echo "=============================================="

# Check data files exist
if [ ! -f ../data/training_mlp_v3_100M_v2.npz ]; then
    echo "ERROR: MLP data file not found!"
    ls -lh ../data/*_v2.npz 2>/dev/null
    exit 1
fi

echo "Data files available:"
ls -lh ../data/*_v2.npz
echo ""

# Submit MLP training jobs
echo "Submitting MLP training jobs..."
for arch in shallow_256 shallow_512 deep_128 deep_256; do
    condor_submit submit_mlp_${arch}.sub
done

# Submit PINN training jobs
echo ""
echo "Submitting PINN training jobs..."
for col in 5 10 20 50; do
    condor_submit submit_pinn_col${col}_train.sub
done

echo ""
echo "All training jobs submitted!"
condor_q gscriven
