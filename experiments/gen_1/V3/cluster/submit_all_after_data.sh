#!/bin/bash
#===============================================================================
# Submit all V3 training jobs after data extraction completes
# This script waits for the extraction job to finish, then submits training
#===============================================================================

cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation

# Wait for data files to exist
echo "Waiting for data extraction to complete..."
while [ ! -f V3/data/training_mlp_v3_100M_v2.npz ] || \
      [ ! -f V3/data/training_pinn_v3_col50_v2.npz ]; do
    sleep 60
    echo "  Still waiting... $(date)"
done

echo ""
echo "Data files ready!"
ls -lh V3/data/*_v2.npz
echo ""

# Submit MLP training jobs
echo "Submitting MLP training jobs..."
for arch in shallow_256 shallow_512 deep_128 deep_256; do
    condor_submit V3/cluster/submit_${arch}.sub
    echo "  Submitted: $arch"
done

# Submit PINN training jobs
echo "Submitting PINN training jobs..."
for col in 5 10 20 50; do
    condor_submit V3/cluster/submit_pinn_col${col}.sub
    echo "  Submitted: pinn_col${col}"
done

echo ""
echo "All training jobs submitted!"
condor_q gscriven
