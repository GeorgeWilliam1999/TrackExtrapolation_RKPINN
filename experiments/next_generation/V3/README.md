# V3: Variable Step-Size ML Track Extrapolators

**Status:** 🔧 In Development  
**Goal:** Train ML models that can extrapolate tracks over arbitrary step sizes (variable dz)  
**Key Innovation:** Models trained with variable `dz` to enable flexible extrapolation distances

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [V1/V2 Issues Discovered](#v1v2-issues-discovered)
3. [V3 Solution](#v3-solution)
4. [Model Architecture](#model-architecture)
5. [Data Generation](#data-generation)
6. [Training Configuration](#training-configuration)
7. [C++ Integration Workflow](#c-integration-workflow)
8. [File Structure](#file-structure)
9. [Normalization Details](#normalization-details)

---

## Problem Statement

### The dz Limitation

In V1 and V2 experiments, all models were trained with a **fixed propagation distance**:
- Training data: `z_start = 0 mm`, `z_end = 8000 mm` → **dz = 8000 mm**
- All 50 million training samples used identical `dz = 8000`

This causes a critical issue when deploying to C++ where extrapolation distances vary:

```cpp
// LHCb test uses different z ranges:
const double z1 = 3000.;  // z_start
const double z2 = 7000.;  // z_end
// dz = 4000 mm (NOT 8000!)
```

### Normalization Catastrophe

The 6th input feature (`dz`) had these normalization statistics during V1/V2 training:
```
input_mean[5] = 8000.0
input_std[5]  = 9.999e-09  (essentially zero!)
```

When normalizing: `(dz - mean) / std = (4000 - 8000) / 1e-9 = -4e12` → **Numerical explosion!**

### V2 QMTest Results (Before Fix)

| Extrapolator | Output (x, y, tx, ty) | Status |
|-------------|----------------------|--------|
| Reference (RK4) | (-4144, -2209, -1.43, -0.49) | ✅ Correct |
| MLP_v2_shallow_256 | (3.58e12, -8.32e12, -1.64e8, 1.52e9) | ❌ Exploded |

After patching std to use mean value:
| Extrapolator | Output | Status |
|-------------|--------|--------|
| MLP_v2_shallow_256 | (-539, -181, -0.01, 0.04) | ❌ Wrong (model can't generalize) |

---

## V1/V2 Issues Discovered

### Issue 1: PINN Initial Condition Failure (V1)

**Problem:** PINN and RK_PINN architectures failed to satisfy the Initial Condition (IC) constraint.

| z_frac | PINN Output | Expected |
|--------|-------------|----------|
| 0.0 (IC) | x = 2768 mm | x = 207 mm |
| 1.0 | x = 2752 mm | x = 1039 mm |

**Root Cause:** The forward pass set `x[:, 5] = 1.0` for all inputs during training. The network ignored z_frac and learned a direct mapping from initial to final state.

**Solution (V2):** Residual architecture: `Output = InitialState + z_frac × NetworkCorrection`

### Issue 2: Fixed dz Training (V1 & V2)

**Problem:** All models trained with `dz = 8000 mm` only.

**Consequence:**
- `input_std[dz] ≈ 0` → Division by zero during normalization
- Model cannot extrapolate different distances
- Deployed model produces garbage for `dz ≠ 8000`

**Solution (V3):** Train with variable `dz ∈ [500, 12000] mm`

### Issue 3: Input Feature Representation

**V1/V2 Input:** `[x, y, tx, ty, q/p, dz]` where `dz` was constant

**V3 Input Options:**
1. `[x, y, tx, ty, q/p, dz]` - Raw distance (requires larger network for scale invariance)
2. `[x, y, tx, ty, q/p, z_frac]` - Normalized fraction `z_frac = dz / dz_max`
3. `[x, y, tx, ty, q/p, log(dz)]` - Log-scale distance (handles large range better)

**V3 Choice:** Raw `dz` with proper normalization from diverse training data.

---

## V3 Solution

### Key Changes

1. **Variable dz Training Data:**
   ```python
   # V1/V2: Fixed dz
   dz = 8000  # Always!
   
   # V3: Variable dz
   dz = np.random.uniform(500, 12000)  # Random for each sample
   ```

2. **Proper Normalization:**
   ```python
   # V3 expected statistics (after training with variable dz):
   input_mean[5] ≈ 6250     # Mean of uniform(500, 12000)
   input_std[5]  ≈ 3300     # Std of uniform distribution
   ```

3. **Larger Dataset:**
   - V1/V2: 50M samples (fixed dz)
   - V3: 100M+ samples (variable dz, more diversity needed)

4. **q/p as Active Feature:**
   - Ensure model learns momentum-dependent bending
   - Include diverse momentum range: 0.5-100 GeV

---

## Model Architecture

### Input Features (6 dimensions)
| Index | Feature | Units | Range (Training) | Description |
|-------|---------|-------|------------------|-------------|
| 0 | x | mm | [-300, 300] | Horizontal position at z_start |
| 1 | y | mm | [-250, 250] | Vertical position at z_start |
| 2 | tx | - | [-0.3, 0.3] | Horizontal slope dx/dz |
| 3 | ty | - | [-0.25, 0.25] | Vertical slope dy/dz |
| 4 | q/p | 1/MeV | [-4e-4, 4e-4] | Charge/momentum (±0.5-100 GeV) |
| 5 | dz | mm | [500, 12000] | Propagation distance |

### Output Features (4 dimensions)
| Index | Feature | Units | Description |
|-------|---------|-------|-------------|
| 0 | x | mm | Horizontal position at z_end |
| 1 | y | mm | Vertical position at z_end |
| 2 | tx | - | Horizontal slope at z_end |
| 3 | ty | - | Vertical slope at z_end |

**Note:** `q/p` is conserved (no material interactions) so not predicted.

### Network Architectures (V3)

| Model | Hidden Layers | Params | Activation | Notes |
|-------|---------------|--------|------------|-------|
| mlp_v3_shallow_256 | [256, 256] | ~70k | SiLU | Baseline |
| mlp_v3_shallow_512 | [512, 512] | ~270k | SiLU | Higher capacity |
| mlp_v3_deep_256 | [256, 256, 256, 128] | ~140k | SiLU | Deeper for dz learning |
| pinn_v3_residual_256 | [256, 256] | ~70k | SiLU | IC-guaranteed via residual |

---

## Data Generation

### Strategy: Trajectory-Based Segment Extraction

**Key Insight:** Generate full trajectories ONCE, then extract O(N²) training samples!

```
Full Trajectory (5mm RK4 steps, ~3000 points for 15km):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z=0                          z=7500                     z=15000

From ONE trajectory, extract thousands of segments:
  [z=500 → z=2000]     dz=1500mm   ─────▶  MLP sample
  [z=1200 → z=8700]    dz=7500mm   ─────▶  MLP sample
  [z=3000 → z=7000]    dz=4000mm   ─────▶  MLP sample (matches QMTest!)
  ...
```

**Efficiency:**
- 10,000 trajectories × 100 segments each = **1M training samples**
- BUT each segment includes supervised collocation points for PINN!

### V3 Data Generation Scripts

**1. Generate Full Trajectories:**
```bash
# Generate 10k full trajectories with 5mm resolution
python generate_trajectories.py \
    --n_trajectories 10000 \
    --z_start 0 \
    --z_end 15000 \
    --step_size 5 \
    --output trajectories_10k.npz
```

**2. Extract Variable dz Samples:**
```bash
# Extract 100M (state, dz) → state samples
python extract_segments.py \
    --input trajectories_10k.npz \
    --n_samples 100000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --output training_mlp_v3.npz

# Extract samples WITH collocation for PINN
python extract_segments.py \
    --input trajectories_10k.npz \
    --n_samples 10000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --collocation_points 10 \
    --output training_pinn_v3.npz
```

### Data Formats

**MLP Training Data:**
```python
# training_mlp_v3.npz
{
    'X': np.array([n_samples, 6]),  # [x, y, tx, ty, q/p, dz]
    'Y': np.array([n_samples, 4]),  # [x, y, tx, ty] at z_end
}
```

**PINN Training Data (with supervised collocation):**
```python
# training_pinn_v3.npz
{
    'X': np.array([n_samples, 6]),        # [x, y, tx, ty, q/p, dz]
    'Y': np.array([n_samples, 4]),        # [x, y, tx, ty] at z_end
    'z_frac': np.array([n_samples, n_col]),  # Collocation z_frac values
    'Y_col': np.array([n_samples, n_col, 4]), # TRUE states at collocation!
}
```

### PINN Supervised Collocation

**This is the key innovation for V3 PINN:**

```python
# Traditional PINN: Physics loss at random z_frac (unsupervised)
z_frac = torch.rand(batch_size)
pred = pinn(input, z_frac)
physics_residual = compute_ode_residual(pred, z_frac, dz)  # No ground truth!

# V3 PINN: Supervised collocation from trajectory data
z_frac = batch['z_frac']        # [batch, n_col] known z_frac values
Y_col = batch['Y_col']          # [batch, n_col, 4] TRUE states!
pred = pinn(input, z_frac)      # [batch, n_col, 4]
collocation_loss = F.mse_loss(pred, Y_col)  # Direct supervision!
```

**Advantages:**
1. No physics residual needed (have ground truth at all collocation points!)
2. Faster convergence (direct supervision vs soft physics constraint)
3. Still enforces trajectory consistency via IC + endpoints + intermediate points

### Condor Submission

```bash
# Step 1: Generate trajectories
condor_submit V3/cluster/submit_generate_trajectories.sub

# Step 2: Extract training samples  
condor_submit V3/cluster/submit_extract_segments.sub
```

---

## Training Configuration

### V3 Models

| Model | Architecture | Training Data | Loss Function |
|-------|--------------|---------------|---------------|
| **MLP** | [256,256] → 4 | (input, dz) → endpoint | MSE on endpoint |
| **PINN** | [256,256] → 4 | (input, dz, z_frac) → state | MSE on collocation + IC + endpoint |

### MLP Training Config

```python
CONFIG_MLP_V3 = {
    # Data
    "data_path": "data/training_mlp_v3.npz",
    
    # Architecture
    "model_type": "mlp",
    "hidden_dims": [256, 256],
    "activation": "silu",
    
    # Training
    "batch_size": 8192,
    "epochs": 30,
    "learning_rate": 0.001,
    "scheduler": "cosine",
}
```

### PINN Training Config (with supervised collocation)

```python
CONFIG_PINN_V3 = {
    # Data
    "data_path": "data/training_pinn_v3.npz",  # Includes collocation!
    
    # Architecture  
    "model_type": "pinn_residual",
    "hidden_dims": [256, 256],
    "activation": "silu",
    
    # Training
    "batch_size": 4096,  # Smaller due to collocation expansion
    "epochs": 50,
    "learning_rate": 0.001,
    
    # PINN Loss Weights
    "lambda_ic": 10.0,        # Initial condition (z_frac=0)
    "lambda_endpoint": 1.0,   # Final state (z_frac=1)
    "lambda_collocation": 1.0, # Intermediate points (supervised!)
}
```

### PINN Loss Function (V3 - Supervised)

```python
def pinn_loss_v3(model, batch):
    """
    V3 PINN loss with supervised collocation.
    
    No physics residual needed - we have ground truth at collocation points!
    """
    # Unpack batch
    input_state = batch['X']        # [B, 6] = [x, y, tx, ty, qop, dz]
    endpoint = batch['Y']           # [B, 4] = final state
    z_frac = batch['z_frac']        # [B, N_col] = collocation fractions
    Y_col = batch['Y_col']          # [B, N_col, 4] = TRUE states at collocation
    
    B, N_col = z_frac.shape
    
    # 1. IC Loss: At z_frac=0, output must equal input (residual = 0)
    ic_pred = model(input_state, z_frac=torch.zeros(B, 1))
    ic_target = input_state[:, :4]  # [x, y, tx, ty]
    ic_loss = F.mse_loss(ic_pred, ic_target)
    
    # 2. Endpoint Loss: At z_frac=1, output must equal final state
    endpoint_pred = model(input_state, z_frac=torch.ones(B, 1))
    endpoint_loss = F.mse_loss(endpoint_pred, endpoint)
    
    # 3. Collocation Loss: At intermediate z_frac, output must match trajectory
    #    This is SUPERVISED, not physics-based!
    col_pred = model(input_state.unsqueeze(1).expand(-1, N_col, -1), 
                     z_frac=z_frac)  # [B, N_col, 4]
    collocation_loss = F.mse_loss(col_pred, Y_col)
    
    # Total loss (no physics residual needed!)
    total = (cfg.lambda_ic * ic_loss + 
             cfg.lambda_endpoint * endpoint_loss +
             cfg.lambda_collocation * collocation_loss)
    
    return total, {'ic': ic_loss, 'endpoint': endpoint_loss, 'col': collocation_loss}
```

### Training Commands

```bash
# Train MLP
python train_v3.py --config configs/mlp_v3_shallow_256.json

# Train PINN
python train_v3.py --config configs/pinn_v3_residual_256.json

# Submit to cluster
condor_submit V3/cluster/submit_v3_training.sub
```

---

## C++ Integration Workflow

### 1. Export Trained Model to Binary

```bash
python deployment/export_to_cpp.py \
    trained_models/mlp_v3_shallow_256 \
    ml_models/models/mlp_v3_shallow_256.bin
```

### 2. Binary File Format

```
[int]    num_layers
For each layer:
    [int]    rows
    [int]    cols
    [double] weights[rows * cols]
    [double] biases[rows]
[int]    input_size
[double] input_mean[input_size]
[double] input_std[input_size]
[int]    output_size
[double] output_mean[output_size]
[double] output_std[output_size]
[int]    activation_len
[char]   activation[activation_len]
```

### 3. Update Test Configuration

```python
# tests/options/test_extrapolators.py
from TrackExtrapolators.TrackExtrapolatorsConf import TrackMLPExtrapolator

extrapolators += [
    TrackMLPExtrapolator("MLP_v3",
        ModelPath="/path/to/mlp_v3_shallow_256.bin",
        Activation="silu"),
]
```

### 4. Run QMTest

```bash
cd /data/bfys/gscriven/TE_stack
make fast/Rec/test ARGS='-R extrapolators -V'
```

---

## File Structure

```
V3/
├── README.md                 # This file
├── data_generation/
│   ├── generate_variable_dz.py    # Main data generation script
│   ├── submit_datagen.sub         # Condor submission for data gen
│   └── README.md                  # Data generation docs
├── training/
│   ├── train_v3.py               # V3 training script
│   ├── configs/                   # Training configurations
│   │   ├── mlp_v3_shallow_256.json
│   │   ├── mlp_v3_shallow_512.json
│   │   └── pinn_v3_residual_256.json
│   └── README.md
├── cluster/
│   ├── submit_v3_training.sub    # Condor job submission
│   ├── v3_jobs.txt               # Job list
│   └── README.md
├── models/
│   └── README.md                 # Model architecture details
├── analysis/
│   ├── analyze_v3_results.py     # Analysis scripts
│   └── compare_v1_v2_v3.py       # Cross-version comparison
└── trained_models/               # Output directory for trained models
    └── .gitkeep
```

---

## Normalization Details

### V3 Normalization (Expected after training)

**Input Normalization:**
| Feature | Mean | Std | Notes |
|---------|------|-----|-------|
| x | ~0 | ~170 mm | Position centered at beam |
| y | ~0 | ~140 mm | Position centered at beam |
| tx | ~0 | ~0.087 | Slope ~uniform in [-0.3, 0.3] |
| ty | ~0 | ~0.087 | Slope ~uniform in [-0.25, 0.25] |
| q/p | ~0 | ~6e-4 | Charge/momentum in 1/MeV |
| dz | ~6250 | ~3300 | **Variable!** uniform(500, 12000) |

**Output Normalization:**
| Feature | Mean | Std |
|---------|------|-----|
| x | ~0 | ~700-1500 mm | Depends on dz |
| y | ~0 | ~600-1200 mm | Depends on dz |
| tx | ~0 | ~0.087 | Slopes similar to input |
| ty | ~0 | ~0.087 | Slopes similar to input |

### Normalization in Forward Pass

```python
# Python (training)
x_norm = (x - input_mean) / input_std
y_pred_norm = model(x_norm)
y_pred = y_pred_norm * output_std + output_mean

# C++ (inference)
Eigen::VectorXd x_norm = (input.array() - input_mean.array()) / input_std.array();
Eigen::VectorXd y_norm = forward(x_norm);
Eigen::VectorXd y = y_norm.array() * output_std.array() + output_mean.array();
```

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V1 | Jan 2026 | Initial training, fixed dz=8000, PINN IC issues |
| V2 | Jan 2026 | Shallow-wide architectures, residual PINN, still fixed dz |
| V3 | Jan 2026 | Variable dz training, proper normalization, C++ deployment |

---

## Known Limitations & V4 Plans

V3 benchmark results show that while the variable-dz approach is conceptually correct, the current architectures are **too narrow** to match V2's accuracy on fixed-dz extrapolation. Key observations:

- **V2 best (fixed dz=8000):** 0.028 mm position error with [512, 256] architecture
- **V3 best (variable dz):** ~0.25 mm position error with [128, 128, 128, 128, 64] architecture
- Variable dz is a fundamentally harder regression problem — the model must learn a family of extrapolations parameterized by dz, requiring significantly more capacity.
- V2's wider/shallower networks ([512, 256]) outperformed V3's deeper/narrower ones.

**V4 will address this with very wide architectures** (e.g. [1024, 512], [2048, 1024]) applied to the variable-dz problem. The hypothesis is that width (not depth) is the dominant factor for this smooth, physics-governed function.

---

## Authors

- G. Scriven (LHCb Collaboration)
- January 2026
