# Field Map NN Grid Search

**Goal**: Find the optimal neural network architecture to approximate the LHCb magnetic field map `(x, y, z) -> (Bx, By, Bz)`, replacing the current trilinear grid interpolation with a faster NN surrogate.

## Motivation

From profiling the LHCb track extrapolation pipeline ([profiling notebook](../../play_time/rk_extrapolators_explained.ipynb)):

| Metric | Value |
|--------|-------|
| Field lookup cost | ~0.109 us per call (trilinear interpolation on 3D grid) |
| Lookups per track | ~69 (CashKarp RK4(5): ~11.5 steps x 6 stages) |
| Field fraction | ~76% of total CashKarp propagation cost |
| Per-track cost | ~8.35 us (of which ~6.3 us is field lookups) |

The field map is **static per run** -- an ideal target for a neural network approximation. If the NN is faster than trilinear interpolation, the overall extrapolation speeds up proportionally (Amdahl's Law).

## Experimental Design

### Data

**Source**: `twodip.rtf` -- regular 3D grid, 81 x 81 x 146 = 957,906 points.
- Input: `(x, y, z)` in mm -- detector coordinates
- Output: `(Bx, By, Bz)` in Tesla -- magnetic field vector
- Grid covers x in [-4000, 4000], y in [-4000, 4000], z in [-500, 14000] mm with 100 mm spacing

### Train / Validation Split

**Training set**: All 957,906 grid vertices. We use 100% of the grid because:
1. During RK stepping, the NN is queried at arbitrary off-grid positions
2. The grid vertices are the only exact data we have
3. What matters is interpolation quality between vertices, not fitting a held-out subset

**Validation set**: 100,000 random off-grid points sampled uniformly within the grid bounds. The ground truth for these points is computed via **trilinear interpolation** on the grid -- this is exactly the function the NN must replace (the C++ `fieldVectorLinearInterpolation`).

This design means:
- We train the NN to perfectly learn the grid vertex values
- We evaluate whether it can also reproduce the trilinear interpolation between vertices
- The validation directly measures how well the NN would substitute for the existing C++ field service

### Architecture Grid Search

We search over **30 architectures**: 5 widths x 3 depths x 2 activations.

| Parameter | Values |
|-----------|--------|
| Hidden width | 32, 64, 128, 256, 512 |
| Hidden depth | 1, 2, 3 layers |
| Activation | ReLU, SiLU |

All models are simple MLPs: `Linear(3, H) -> Act -> [Linear(H, H) -> Act ->]* Linear(H, 3)`

**Why these choices?**
- **Width range**: From tiny (32 = 259 params) to large (512 = 265k params for 1 layer). The sizing notebook showed single-layer nets up to 256 wide fit comfortably in L1 cache.
- **Depth range**: 1 layer is essential for single-sample speed (each additional layer adds H^2 FLOPs). We include 2-3 layers to map the accuracy/speed frontier.
- **ReLU vs SiLU**: ReLU is fast (1 FLOP/neuron) but produces piecewise-linear approximations. SiLU is smooth (4 FLOPs/neuron) and often more accurate for smooth fields like B(x,y,z). The grid search quantifies this trade-off.

### Training Configuration

| Setting | Value | Justification |
|---------|-------|---------------|
| Optimizer | Adam | Standard, works well for small MLPs |
| Learning rate | 1e-3 | Standard starting point, decayed via cosine |
| LR schedule | Cosine annealing (to ~1e-5) | Smooth decay, good convergence |
| Batch size | 4096 | Fits in GPU memory, good gradient statistics |
| Max epochs | 200 | Sufficient for convergence on ~1M samples |
| Early stopping | Patience = 20 epochs | Stop if validation MSE stops improving |
| Normalisation | Per-dimension mean/std on inputs & outputs | Standard practice |

### Evaluation Metrics

| Metric | Unit | Target | Description |
|--------|------|--------|-------------|
| MAE | Gauss | < 0.1 | Mean absolute error (1 T = 10,000 Gauss) |
| p99 | Gauss | < 1.0 | 99th percentile error |
| Max | Gauss | < 10.0 | Worst-case error |
| FLOPs | count | < 57 | Must beat trilinear (~57 FLOPs) for speed |
| Params | count | < 8192 | Must fit in 32 KB L1 cache (float32) |

The accuracy targets come from the requirement that field errors should not significantly degrade track reconstruction. The speed targets come from the sizing notebook analysis.

### Outputs Per Model

Each trained model saves to `trained_models/<config_name>/`:
- `model.pt` -- PyTorch state dict (best validation checkpoint)
- `config.json` -- Training configuration used
- `model_config.json` -- Architecture details, param count, FLOP count, L1 cache fit
- `normalization.json` -- Input/output normalisation constants
- `history.json` -- Full per-epoch training history + final metrics
- `exports/field_nn.onnx` -- ONNX model for C++ deployment
- `exports/field_nn_weights.h` -- C header with embedded weights for direct integration

## Directory Structure

```
field_nn/
  README.md                 # This file
  generate_configs.py       # Script to create all 30 config JSONs + job list
  training/
    trilinear.py            # Vectorised trilinear interpolation (validation ground truth)
    train_field_nn.py       # Main training script (--config <json>)
    train_field_nn_wrapper.sh  # HTCondor worker-node wrapper
  configs/                  # 30 JSON config files (one per architecture)
    field_nn_relu_1L_32H.json
    field_nn_relu_1L_64H.json
    ...
    field_nn_silu_3L_512H.json
  cluster/
    submit_field_nn.sub     # HTCondor submit file
    field_nn_jobs.txt       # Job list (one config name per line)
    submit_all.sh           # Master submit script with dry-run validation
    logs/                   # HTCondor stdout/stderr/log files
  trained_models/           # Output directory (one subfolder per model)
  analysis/                 # Post-training analysis (notebook, to be created)
```

## How to Run

### 1. Generate configs

```bash
cd /data/bfys/gscriven/TrackExtrapolation/experiments/field_maps/field_nn
python generate_configs.py
```

### 2. Local test (2 epochs, verify everything works)

```bash
python training/train_field_nn.py --config configs/field_nn_relu_1L_32H.json --epochs 2
```

### 3. Dry run (validate all configs, check environment)

```bash
bash cluster/submit_all.sh
```

### 4. Submit to Nikhef STBC cluster

```bash
bash cluster/submit_all.sh --submit
```

### 5. Monitor

```bash
condor_q -nobatch
watch 'condor_q -nobatch | tail -40'
```

### 6. Check results

```bash
# Quick check: which models finished?
for d in trained_models/*/; do
    if [ -f "$d/history.json" ]; then
        mae=$(python -c "import json; h=json.load(open('${d}history.json')); print(f'{h[\"final_metrics\"][\"mae_gauss\"]:.2f}')")
        echo "$(basename $d): MAE = $mae Gauss"
    fi
done
```

## Cluster Details

- **Cluster**: Nikhef STBC (stoomboot)
- **GPUs**: Tesla V100-PCIE-32GB, NVIDIA L40S (45GB)
- **OS**: Enterprise Linux 9
- **Conda env**: `TE` at `/data/bfys/gscriven/conda/envs/TE/` (Python 3.10, PyTorch + CUDA)
- **Filesystem**: Shared NFS (`/data/bfys/gscriven/`) -- no file transfer needed
- **Resources per job**: 1 GPU, 4 CPUs, 16 GB RAM, 10 GB disk

## Next Steps (after this experiment)

1. **Analysis notebook**: Heatmaps of MAE vs (width, depth), Pareto plot of accuracy vs speed
2. **Track-quality evaluation**: Integrate best NN into C++ Gaudi framework, compare track propagation accuracy to real field (separate gen_2 experiment)
3. **SIMD deployment**: Embed best architecture weights in C++ with AVX2 intrinsics for single-sample inference
