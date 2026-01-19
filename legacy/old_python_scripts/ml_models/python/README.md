# Python Training Scripts

Training and evaluation scripts for ML-based track extrapolators.

---

## 📂 File Overview

| Script | Purpose | Status |
|--------|---------|--------|
| `generate_training_data.py` | ⭐ Generate data from RK4 | ✅ Active |
| `train_analysis_models.py` | ⭐ Train all analysis variants | ✅ Active |
| `train_on_gpu.py` | ⭐ General GPU training | ✅ Active |
| `full_domain_training.py` | Full momentum range training | ✅ Active |
| `compare_models.py` | Model comparison utilities | ✅ Active |
| `train_large_models.py` | Large architecture training | ⏳ Testing |
| `train_pinn.py` | PINN training (deprecated) | ❌ Deprecated |
| `train_true_pinn.py` | True PINN (deprecated) | ❌ Deprecated |
| `test_pinn_simple.py` | Simple test script | 🔧 Utility |
| `test_pipeline.sh` | End-to-end test | 🔧 Utility |

---

## 🚀 Quick Start

### 1. Generate Training Data

```bash
# Generate 50K samples for analysis
python generate_training_data.py \
    --samples 50000 \
    --output ../data/ \
    --name analysis

# Generate larger dataset for production
python generate_training_data.py \
    --samples 500000 \
    --output ../data/ \
    --name production \
    --workers 8
```

**Options:**
- `--samples N` - Number of tracks to generate
- `--output DIR` - Output directory
- `--name NAME` - Prefix for output files
- `--workers N` - Parallel processes (default: CPU count)

**Output files:**
- `X_<name>.npy` - Input states (N × 6)
- `Y_<name>.npy` - Target states (N × 4)
- `P_<name>.npy` - Momentum values (N)

### 2. Train All Analysis Models

```bash
# Train all 12 variants (activation + architecture + PINN)
python train_analysis_models.py

# Or with custom options
python train_analysis_models.py \
    --epochs 3000 \
    --device cuda:1 \
    --output ../models/analysis_v2/
```

This trains:
- **Architecture variants:** tiny, small, medium, large, xlarge
- **Activation variants:** tanh, relu, silu
- **PINN variants:** λ = 0.01, 0.05, 0.1, 0.2

### 3. Train Single Custom Model

```bash
python train_on_gpu.py \
    --hidden 256 256 128 64 \
    --activation silu \
    --epochs 2000 \
    --lr 0.001 \
    --weight-decay 1e-5 \
    --batch-size 1024 \
    --name my_custom_model \
    --device cuda
```

**Common options:**
- `--hidden N N N` - Hidden layer sizes
- `--activation {tanh,relu,silu}` - Activation function
- `--epochs N` - Training epochs
- `--lr FLOAT` - Learning rate
- `--batch-size N` - Batch size
- `--name STR` - Model name
- `--device {cuda,cpu}` - Training device

---

## 📝 Script Details

### `generate_training_data.py`

Fast parallel data generation using LHCb's Runge-Kutta extrapolator.

**How it works:**
1. Generate random track states (x, y, tx, ty, q/p) at z = 4000 mm
2. Extrapolate each to z = 12000 mm using TrackRungeKuttaExtrapolator
3. Save input/output pairs as NumPy arrays

**Phase space coverage:**
- Position: x, y ∈ [-1000, 1000] mm
- Slopes: tx, ty ∈ [-0.3, 0.3]
- Momentum: p ∈ [0.5, 100] GeV/c
- Charge: Both +1 and -1

### `train_analysis_models.py`

Trains all model variants for the analysis notebook.

**Models trained:**
```
Architecture (5):
  mlp_tiny.pt      - [64, 32]
  mlp_small.pt     - [128, 64]
  mlp_medium.pt    - [128, 128, 64]
  mlp_large.pt     - [256, 256, 128, 64]
  mlp_xlarge.pt    - [512, 512, 256, 128]

Activation (3):
  mlp_act_tanh.pt  - tanh activation
  mlp_act_relu.pt  - ReLU activation
  mlp_act_silu.pt  - SiLU activation ⭐ BEST

PINN (4):
  pinn_lambda_0_01.pt - λ = 0.01
  pinn_lambda_0_05.pt - λ = 0.05
  pinn_lambda_0_1.pt  - λ = 0.1
  pinn_lambda_0_2.pt  - λ = 0.2
```

**Output:**
- `.pt` files (PyTorch state dict)
- `_metadata.json` (training config and metrics)

### `train_on_gpu.py`

General-purpose GPU training script with full configurability.

**Features:**
- AdamW optimizer with weight decay
- ReduceLROnPlateau scheduler
- Early stopping
- Checkpoint saving
- TensorBoard logging (optional)
- Mixed precision training (optional)

**Example:**
```bash
python train_on_gpu.py \
    --hidden 128 128 64 \
    --activation silu \
    --epochs 2000 \
    --lr 0.001 \
    --batch-size 1024 \
    --name production_v1
```

### `compare_models.py`

Model comparison and evaluation utilities.

**Functions:**
- `load_all_models()` - Load all models from directory
- `evaluate_model()` - Compute metrics on test set
- `compare_momentum_bins()` - Performance vs momentum
- `plot_error_distributions()` - Error histograms

**Usage:**
```python
from compare_models import evaluate_model, load_all_models

models = load_all_models('../models/analysis/')
for name, model in models.items():
    metrics = evaluate_model(model, X_test, Y_test)
    print(f"{name}: {metrics['mean_error']:.3f} mm")
```

### `full_domain_training.py`

Training focused on full momentum range coverage (0.5-200 GeV/c).

**Features:**
- Extended momentum range data
- Momentum-stratified validation
- Detailed per-bin metrics

---

## ⚠️ Deprecated Scripts

### `train_pinn.py` and `train_true_pinn.py`

**Status:** ❌ Deprecated

These scripts implement physics-informed neural networks with simplified Lorentz force constraints. **Analysis shows the physics loss formulation is incorrect** and makes predictions worse.

**Problem:** Physics loss assumes straight-line propagation, which conflicts with actual curved trajectories in the magnetic field.

**Recommendation:** Use standard data-driven training (`train_on_gpu.py` with SiLU activation).

If you need physics constraints, consider:
- Neural ODEs (proper ODE integration)
- Trajectory-aware collocation points
- Proper B-field integration in loss

---

## 🔧 Development

### Running Tests

```bash
# Test full pipeline
./test_pipeline.sh

# Quick sanity check
python test_pinn_simple.py
```

### Adding New Models

1. Create training script or modify `train_on_gpu.py`
2. Save model with:
   ```python
   torch.save(model.state_dict(), f'{name}.pt')
   with open(f'{name}_metadata.json', 'w') as f:
       json.dump(config, f)
   ```
3. Add to `../experiments/experiment_log.csv`
4. Update README with results

### Environment Setup

```bash
# Create conda environment
conda create -n TE python=3.10
conda activate TE

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install numpy matplotlib pandas scipy jupyter
```

---

## 📊 Output Formats

### Model Files (`.pt`)

PyTorch state dict format:
```python
{
    'layer_0.weight': tensor(...),
    'layer_0.bias': tensor(...),
    'layer_1.weight': tensor(...),
    ...
}
```

Load with:
```python
model = TrackMLP(hidden_sizes=[128, 128, 64])
model.load_state_dict(torch.load('model.pt'))
```

### Metadata Files (`.json`)

```json
{
    "name": "mlp_act_silu",
    "architecture": [128, 128, 64],
    "activation": "silu",
    "input_dim": 6,
    "output_dim": 4,
    "epochs": 2000,
    "final_loss": 0.0001,
    "mean_error_mm": 0.207,
    "p95_error_mm": 0.543,
    "training_time_s": 142.5,
    "date": "2025-01-13"
}
```

### Data Files (`.npy`)

NumPy array format:
- `X_*.npy`: Shape (N, 6) - float32
- `Y_*.npy`: Shape (N, 4) - float32
- `P_*.npy`: Shape (N,) - float32

---

**Last Updated:** January 2025
