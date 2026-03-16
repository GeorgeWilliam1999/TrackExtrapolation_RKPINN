# V3 Development Plan

## Overview

This document outlines the complete V3 development plan for variable-dz ML track extrapolators.

**Goal:** Train MLP and PINN models that can extrapolate tracks over arbitrary distances (500-12000mm), replacing multiple RK4 steps with a single neural network call.

**Key Innovation:** 
- Variable dz training data from trajectory segments
- Supervised collocation for PINN (ground truth at intermediate points)
- Architecture optimizations for fast inference

---

## Table of Contents

1. [V3 vs V1/V2 Differences](#v3-vs-v1v2-differences)
2. [Data Generation Strategy](#data-generation-strategy)
3. [Model Architectures](#model-architectures)
4. [Training Configurations](#training-configurations)
5. [Collocation Point Study](#collocation-point-study)
6. [Inference Optimization](#inference-optimization)
7. [Evaluation Plan](#evaluation-plan)
8. [Timeline and Milestones](#timeline-and-milestones)
9. [File Structure](#file-structure)

---

## V3 vs V1/V2 Differences

| Aspect | V1/V2 | V3 |
|--------|-------|-----|
| **dz** | Fixed 8000mm | Variable 500-12000mm |
| **Data source** | Endpoint only | Trajectory segments |
| **PINN collocation** | Unsupervised (physics loss) | Supervised (ground truth!) |
| **Samples** | 50M | 100M+ (from 10k trajectories) |
| **Inference** | Fails for dz≠8000 | Works for any dz in range |

### Why V1/V2 Failed

```
V1/V2: input_std[dz] ≈ 0  (all samples had dz=8000)
       → (dz - 8000) / 0 = ∞  for any dz ≠ 8000
       → Numerical explosion in C++ deployment
       
V3:    input_std[dz] ≈ 3300  (variable dz)
       → Proper normalization for all dz values
```

---

## Data Generation Strategy

### Step 1: Generate Full Trajectories

Generate high-resolution trajectories covering the full detector:

```bash
python V3/data_generation/generate_trajectories.py \
    --n_trajectories 10000 \
    --z_start 0 \
    --z_end 15000 \
    --step_size 5 \
    --p_min 0.5 \
    --p_max 100 \
    --output data/trajectories_10k.npz
```

**Output format:**
```python
{
    'T': np.array([n_traj], dtype=object),  # Each element: [n_steps, 6]
                                             # Format: [z, x, y, tx, ty, qop]
    'P': np.array([n_traj]),                # Momentum in GeV
}
```

**Statistics:**
- 10k trajectories × 3000 points × 6 features × 4 bytes ≈ **700 MB**
- Each trajectory: z=0 → z=15000mm, 5mm steps

### Step 2: Extract Training Samples

From trajectories, extract random segments with variable dz:

```bash
# MLP training data (100M samples, endpoint only)
python V3/data_generation/extract_segments.py \
    --input data/trajectories_10k.npz \
    --n_samples 100000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --output data/training_mlp_v3.npz

# PINN training data (10M samples, with collocation)
python V3/data_generation/extract_segments.py \
    --input data/trajectories_10k.npz \
    --n_samples 10000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --collocation_points 10 \
    --output data/training_pinn_v3_col10.npz
```

**MLP data format:**
```python
{
    'X': np.array([n_samples, 6]),  # [x, y, tx, ty, qop, dz]
    'Y': np.array([n_samples, 4]),  # [x, y, tx, ty] at endpoint
}
```

**PINN data format (with supervised collocation):**
```python
{
    'X': np.array([n_samples, 6]),           # [x, y, tx, ty, qop, dz]
    'Y': np.array([n_samples, 4]),           # [x, y, tx, ty] at endpoint
    'z_frac': np.array([n_samples, n_col]),  # Collocation z_frac values
    'Y_col': np.array([n_samples, n_col, 4]) # TRUE states at collocation!
}
```

### Why This Approach?

From ONE trajectory, we can extract O(N²) training samples:

```
Trajectory with 3000 points (z=0 → z=15000, 5mm steps):
    
    Segment [i=100, j=500]:   z=500 → z=2500,   dz=2000mm
    Segment [i=200, j=1800]:  z=1000 → z=9000,  dz=8000mm
    Segment [i=600, j=1400]:  z=3000 → z=7000,  dz=4000mm  ← matches QMTest!
    ... thousands more combinations
```

**Efficiency:**
- 10k trajectories → 100M+ unique (state, dz) pairs
- Data generation is embarrassingly parallel
- Collocation points are FREE (just index into trajectory)

---

## Model Architectures

### MLP (Baseline)

Direct mapping from (state, dz) → final_state:

```
Input:  [x, y, tx, ty, qop, dz]  (6 features)
        ↓
    [Linear 6→256, SiLU]
        ↓
    [Linear 256→256, SiLU]
        ↓
    [Linear 256→4]
        ↓
Output: [x, y, tx, ty]  (4 features)
```

**Inference:** Single forward pass, ~70k FLOPs

### PINN Residual (Recommended)

The key insight: output = initial_condition + z_frac × network_correction

```
Input:  [x₀, y₀, tx₀, ty₀, qop, dz, z_frac]  (7 features)
        ↓
    ┌─────────────────────────────────────┐
    │  Core Network (shared encoder):     │
    │  [Linear 6→256, SiLU]               │  ← Only (state, dz), NOT z_frac!
    │  [Linear 256→256, SiLU]             │
    │  [Linear 256→4]                     │
    │  → "correction" vector              │
    └─────────────────────────────────────┘
        ↓
    Output = [x₀, y₀, tx₀, ty₀] + z_frac × correction
```

**Why this architecture?**

1. **IC Guaranteed:** At z_frac=0, output = initial_state (no training needed!)
2. **Smooth interpolation:** z_frac linearly scales the correction
3. **Fast inference:** At z_frac=1, output = state + correction (same as MLP!)

```python
class PINNResidual(nn.Module):
    def __init__(self, hidden_dims=[256, 256]):
        super().__init__()
        # Core network processes (state, dz) only - NOT z_frac
        layers = []
        in_dim = 6  # [x, y, tx, ty, qop, dz]
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))  # Output: correction
        self.core = nn.Sequential(*layers)
    
    def forward(self, state_dz, z_frac):
        """
        Args:
            state_dz: [B, 6] = [x, y, tx, ty, qop, dz]
            z_frac: [B, 1] or [B, N_col] for batched collocation
        """
        initial = state_dz[:, :4]  # [x, y, tx, ty]
        correction = self.core(state_dz)  # [B, 4]
        
        # Handle batched z_frac for collocation
        if z_frac.dim() == 2 and z_frac.size(1) > 1:
            # [B, N_col, 1] × [B, 1, 4] → [B, N_col, 4]
            initial = initial.unsqueeze(1)
            correction = correction.unsqueeze(1)
            z_frac = z_frac.unsqueeze(-1)
        
        return initial + z_frac * correction
```

### PINN with z_frac as Input (Alternative)

For comparison, test architecture where z_frac is a regular input:

```
Input:  [x, y, tx, ty, qop, dz, z_frac]  (7 features)
        ↓
    [Linear 7→256, SiLU]
        ↓
    [Linear 256→256, SiLU]
        ↓
    [Linear 256→4]
        ↓
Output: [x, y, tx, ty]
```

**Pros:** More flexible (network can learn non-linear z_frac dependence)
**Cons:** Must learn IC constraint, slightly more parameters

---

## Training Configurations

### MLP Configurations

| Config | Hidden Dims | Params | Notes |
|--------|-------------|--------|-------|
| mlp_v3_shallow_256 | [256, 256] | ~70k | Baseline |
| mlp_v3_shallow_512 | [512, 512] | ~270k | Higher capacity |
| mlp_v3_deep_256 | [256, 256, 256, 128] | ~140k | Deeper |
| mlp_v3_deep_512 | [512, 512, 256] | ~400k | Deep + wide |

### PINN Configurations (with collocation study)

| Config | Hidden Dims | Collocation Points | Notes |
|--------|-------------|-------------------|-------|
| pinn_v3_res_256_col5 | [256, 256] | 5 | Minimal collocation |
| pinn_v3_res_256_col10 | [256, 256] | 10 | Baseline |
| pinn_v3_res_256_col20 | [256, 256] | 20 | Dense collocation |
| pinn_v3_res_256_col50 | [256, 256] | 50 | Very dense |
| pinn_v3_res_512_col10 | [512, 512] | 10 | Wider network |
| pinn_v3_input_256_col10 | [256, 256] | 10 | z_frac as input |

### Training Hyperparameters

```python
CONFIG_MLP = {
    "batch_size": 8192,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "warmup_epochs": 3,
    "grad_clip": 1.0,
}

CONFIG_PINN = {
    "batch_size": 4096,      # Smaller due to collocation expansion
    "epochs": 50,            # More epochs for constraint learning
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    
    # Loss weights
    "lambda_ic": 10.0,       # High weight for IC (even though residual guarantees it)
    "lambda_endpoint": 1.0,
    "lambda_collocation": 1.0,
}
```

---

## Collocation Point Study

### Research Question

How many supervised collocation points are needed for good PINN training?

### Hypothesis

With supervised collocation (ground truth at intermediate points), fewer collocation points should be needed compared to unsupervised physics-based PINN.

### Experiment Design

Train identical PINN architectures with varying collocation points:

| Experiment | Collocation Points | z_frac Values | Expected Training Cost |
|------------|-------------------|---------------|------------------------|
| col_5 | 5 | [0.1, 0.3, 0.5, 0.7, 0.9] | 1× baseline |
| col_10 | 10 | [0.1, 0.2, ..., 0.9] | ~2× baseline |
| col_20 | 20 | [0.05, 0.10, ..., 0.95] | ~4× baseline |
| col_50 | 50 | [0.02, 0.04, ..., 0.98] | ~10× baseline |

### Metrics to Track

1. **Training loss convergence rate** - Does more collocation = faster convergence?
2. **Validation loss** - Does more collocation = better generalization?
3. **IC satisfaction** - How well is z_frac=0 → initial_state maintained?
4. **Interpolation quality** - Error at unseen z_frac values
5. **C++ deployment accuracy** - Final QMTest error vs reference

### Expected Outcome

With supervised collocation, we expect:
- col_5: May be sufficient (good interpolation with sparse supervision)
- col_10: Good balance of accuracy and training cost
- col_20+: Diminishing returns (already have ground truth)

---

## Inference Optimization

### Key Insight: PINN Residual = MLP at z_frac=1

For the residual PINN architecture:

```
Output = IC + z_frac × Correction

At inference (z_frac=1):
Output = IC + Correction
       = [x₀, y₀, tx₀, ty₀] + Core_Network([x₀, y₀, tx₀, ty₀, qop, dz])
```

**This is identical cost to MLP!** The z_frac multiplication becomes a simple addition.

### C++ Implementation

```cpp
class TrackPINNExtrapolator : public TrackFieldExtrapolatorBase {
    // Same as MLP, but with explicit IC addition
    
    StatusCode propagate(State& state, double zNew, ...) const override {
        double dz = zNew - state.z();
        
        // Build input (same as MLP)
        Eigen::VectorXd input(6);
        input << state.x(), state.y(), state.tx(), state.ty(),
                 state.qOverP(), dz;
        
        // Forward pass
        Eigen::VectorXd correction = m_model.forward(input);  // [4]
        
        // PINN residual: output = IC + correction
        // (For MLP, correction IS the output; for PINN, we add IC)
        state.setX(state.x() + correction(0));   // IC + correction
        state.setY(state.y() + correction(1));
        state.setTx(state.tx() + correction(2));
        state.setTy(state.ty() + correction(3));
        state.setZ(zNew);
        
        return StatusCode::SUCCESS;
    }
};
```

### Export Format

Add flag to binary model file indicating PINN vs MLP:

```
[int32]  model_type  (0=MLP, 1=PINN_RESIDUAL)
[int32]  n_layers
... (rest same as before)
```

### Performance Comparison

| Model | Forward Pass FLOPs | Inference Time | Memory |
|-------|-------------------|----------------|--------|
| MLP [256,256] | ~70k | ~10μs | ~300KB |
| PINN Residual [256,256] | ~70k + 4 adds | ~10μs | ~300KB |
| PINN z_frac-input [256,256] | ~71k | ~10μs | ~300KB |

**Conclusion:** All architectures have essentially identical inference cost. The choice should be based on training effectiveness and accuracy.

---

## Evaluation Plan

### Phase 1: Data Validation

Before training, verify data quality:

```python
# Check dz distribution
plt.hist(X[:, 5], bins=100)  # Should be uniform in [500, 12000]

# Check normalization statistics
print(f"dz mean: {X[:, 5].mean():.1f}")  # Should be ~6250
print(f"dz std: {X[:, 5].std():.1f}")   # Should be ~3300

# Verify collocation data
for i in range(10):
    # Check that Y_col[i, j] matches trajectory at z_frac[i, j]
    ...
```

### Phase 2: Training Metrics

Track during training:
- Train/val loss curves
- Per-feature MSE (x, y, tx, ty separately)
- IC violation (for PINN)
- Gradient norms

### Phase 3: Python Evaluation

After training, evaluate in Python:

```python
# Load test data (10% held out)
X_test, Y_test = load_test_data()

# Predict
Y_pred = model(X_test)

# Metrics
mse = ((Y_pred - Y_test) ** 2).mean(dim=0)
print(f"MSE: x={mse[0]:.4f}, y={mse[1]:.4f}, tx={mse[2]:.6f}, ty={mse[3]:.6f}")

# Check specific dz values matching QMTest
dz_4000_mask = np.abs(X_test[:, 5] - 4000) < 100
evaluate(X_test[dz_4000_mask], Y_test[dz_4000_mask])
```

### Phase 4: C++ Integration Test

Deploy to LHCb stack and run QMTest:

```bash
# Export model
python deployment/export_to_cpp.py \
    V3/trained_models/mlp_v3_shallow_256 \
    ml_models/models/mlp_v3_shallow_256.bin

# Build and test
cd /data/bfys/gscriven/TE_stack
make fast/Rec/test ARGS='-R extrapolators -V'
```

**Success criteria:**
- No NaN/Inf outputs
- Position error < 1mm vs RK4 reference
- Slope error < 1e-4 vs RK4 reference

### Phase 5: Benchmark

Compare timing vs RK4:

```bash
# Run benchmark script
Rec/run gaudirun.py tests/options/benchmark_extrapolators.py
```

**Metrics:**
- Time per extrapolation (μs)
- Total time for 1M extrapolations
- Speedup factor vs TrackRungeKuttaExtrapolator

---

## Timeline and Milestones

### Week 1: Data Generation

- [ ] Generate 10k full trajectories
- [ ] Extract 100M MLP training samples
- [ ] Extract 10M PINN samples with 5, 10, 20, 50 collocation points
- [ ] Validate data statistics

### Week 2: MLP Training

- [ ] Train mlp_v3_shallow_256 (baseline)
- [ ] Train mlp_v3_shallow_512
- [ ] Train mlp_v3_deep_256
- [ ] Evaluate and select best MLP

### Week 3: PINN Collocation Study

- [ ] Train pinn_v3_res_256_col5
- [ ] Train pinn_v3_res_256_col10
- [ ] Train pinn_v3_res_256_col20
- [ ] Train pinn_v3_res_256_col50
- [ ] Analyze collocation impact

### Week 4: C++ Integration & Benchmark

- [ ] Export best MLP to C++
- [ ] Export best PINN to C++
- [ ] Run QMTest validation
- [ ] Benchmark vs RK4
- [ ] Write results report

---

## File Structure

```
V3/
├── PLAN.md                          # This file
├── README.md                        # Overview and quick start
│
├── data_generation/
│   ├── generate_trajectories.py     # Step 1: Full trajectories
│   ├── extract_segments.py          # Step 2: MLP/PINN samples
│   ├── generate_variable_dz.py      # Alternative: direct generation
│   └── README.md
│
├── data/                            # Generated data (gitignored)
│   ├── trajectories_10k.npz
│   ├── training_mlp_v3.npz
│   ├── training_pinn_v3_col5.npz
│   ├── training_pinn_v3_col10.npz
│   ├── training_pinn_v3_col20.npz
│   └── training_pinn_v3_col50.npz
│
├── models/
│   ├── architectures.py             # MLP, PINN definitions
│   ├── pinn_residual.py             # PINN residual architecture
│   └── README.md
│
├── training/
│   ├── train_mlp.py                 # MLP training script
│   ├── train_pinn.py                # PINN training with collocation
│   ├── losses.py                    # Loss functions
│   ├── configs/
│   │   ├── mlp_v3_shallow_256.json
│   │   ├── mlp_v3_shallow_512.json
│   │   ├── mlp_v3_deep_256.json
│   │   ├── pinn_v3_res_256_col5.json
│   │   ├── pinn_v3_res_256_col10.json
│   │   ├── pinn_v3_res_256_col20.json
│   │   └── pinn_v3_res_256_col50.json
│   └── README.md
│
├── cluster/
│   ├── submit_generate_trajectories.sub
│   ├── submit_extract_segments.sub
│   ├── submit_train_mlp.sub
│   ├── submit_train_pinn.sub
│   └── README.md
│
├── analysis/
│   ├── analyze_collocation_study.py
│   ├── compare_architectures.py
│   ├── plot_results.py
│   └── README.md
│
├── trained_models/                  # Training outputs
│   ├── mlp_v3_shallow_256/
│   ├── mlp_v3_shallow_512/
│   ├── pinn_v3_res_256_col5/
│   ├── pinn_v3_res_256_col10/
│   └── ...
│
└── results/
    ├── collocation_study.csv
    ├── architecture_comparison.csv
    └── qmtest_results.csv
```

---

## Summary: V3 Key Points

1. **Variable dz is essential** - Train on dz ∈ [500, 12000] mm for real deployment

2. **Trajectory-based data generation** - Generate once, extract millions of samples

3. **Supervised collocation** - Ground truth at intermediate points (no physics residual needed!)

4. **PINN residual architecture** - Guarantees IC, same inference cost as MLP

5. **Collocation study** - Determine minimal collocation points needed

6. **Same inference cost** - MLP and PINN have identical runtime performance

7. **Target: 10-100× speedup** over RK4 for long extrapolations (>2000mm)

---

## Authors

- G. Scriven (LHCb Collaboration)
- January 2026
