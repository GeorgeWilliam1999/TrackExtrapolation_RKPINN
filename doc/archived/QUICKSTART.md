# Quick Start Guide: Training Large, Accurate Models

This guide shows you how to quickly train large, accurate models using the new parallel workflow.

## Option 1: Local Training (Your Machine)

### Step 1: Generate Data (Once, ~10 minutes for 100K samples)
```bash
conda activate TE
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models/python

# Fast parallel data generation
python generate_training_data.py --samples 100000 --output ../data/
```

### Step 2: Train on GPU (~30-60 minutes per model)
```bash
# Large model (837K parameters)
python train_on_gpu.py --data ../data/ --model large --epochs 2000

# Extra-large model (2.1M parameters)
python train_on_gpu.py --data ../data/ --model xlarge --epochs 2000

# Deep model (431K parameters)
python train_on_gpu.py --data ../data/ --model deep --epochs 2000
```

**All models are saved to** `ml_models/models/` **with metadata!**

---

## Option 2: Cluster Training (Nikhef STBC - Parallel, Faster)

### Step 1: Submit Data Generation to Cluster
```bash
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models/condor

# Submit job (generates 100K samples)
condor_submit generate_data.sub

# Monitor progress
watch -n 5 condor_q
tail -f logs/data_gen_0.out
```

### Step 2: Submit Training Jobs (5 models in parallel!)
```bash
# Once data is ready, submit training
condor_submit train_models.sub

# Monitor GPU jobs
condor_q -constraint 'RequestGpus > 0'

# Check specific model training
tail -f logs/train_large_*.out
```

### Step 3: Collect Results
```bash
# Check trained models
ls -lh ../models/mlp_*_condor.bin

# View results
cat ../models/mlp_large_condor_metadata.json | python -m json.tool
```

---

## Quick Reference

### Data Generation Parameters
```bash
--samples 100000      # Number of training samples
--workers 8           # CPU cores (auto-detects if omitted)
--output ../data/     # Output directory
--name train          # Dataset name prefix
```

### Training Parameters
```bash
--data ../data/       # Data directory
--model large         # Architecture: large, xlarge, deep
--hidden 512 512 256  # Custom architecture (alternative to --model)
--epochs 2000         # Training epochs
--batch 512           # Batch size (larger = faster on GPU)
--lr 0.001            # Learning rate
--name my_model       # Custom model name
```

### HTCondor Commands
```bash
condor_submit job.sub       # Submit job
condor_q                    # Check queue
condor_q -better-analyze ID # Why job is waiting
condor_rm ID                # Remove job
condor_history -limit 10    # Recent jobs
```

---

## Expected Performance

### Data Generation Speed
- **Single core**: ~50-100 samples/s
- **8 cores**: ~400-800 samples/s
- **32 cores**: ~1500-3000 samples/s

### Training Time (2000 epochs, GPU)
- **Large** (512-512-256-128): ~30 min
- **XLarge** (1024-512-256-128): ~45 min
- **Deep** (256x5): ~40 min

### Model Accuracy (Expected)
- **Large**: <1.5 mm mean error
- **XLarge**: <1.2 mm mean error
- **Deep**: <1.4 mm mean error

---

## Next Steps

1. **Generate Data**: Start with 100K samples
2. **Train Models**: Try all 3 architectures
3. **Evaluate**: Run `analyze_extrapolators.ipynb`
4. **Iterate**: Generate more data if needed (200K-500K)
5. **Deploy**: Best model → production

For detailed documentation:
- **Main README**: `../README.md`
- **HTCondor Guide**: `condor/README.md`
- **ML Details**: `ml_models/README.md`
