# PINN Training Stability Fixes

**Date:** January 23, 2026  
**Author:** G. Scriven  
**Issue:** PINN models failing to train - producing NaN/Inf values

## Problem Summary

The PINN (Physics-Informed Neural Network) models were failing during training with:
- `Best: mm` shown in monitoring (missing value = NaN)
- Physics loss: `inf`
- IC loss: `~10^26` (astronomically large)
- All gradients: NaN or Inf
- Training time: ~10 hours for only 2-4 epochs (should be ~100s/epoch)

Meanwhile, MLP models (no physics loss) trained normally, and RK_PINN models struggled but made some progress.

## Root Cause Analysis

### Debug Output (from `debug_pinn.py`)

```
Forward output:
  Min: -2.276050e+13, Max: 3.974403e+12
  Mean: -4.699084e+12, Std: 1.055872e+13

Physics loss components:
  ic: 1.334923e+26 ✓
  pde: inf ⚠️ INVALID

Gradient statistics:
  network.0.weight: norm=nan ⚠️ NaN
  network.0.bias:   norm=inf ⚠️ Large
```

### Failure Chain

1. **Random Weight Initialization Issue**
   - Standard Xavier initialization produces weights that, combined with the denormalization step, output values ~10^13 on the first forward pass
   - The network output is denormalized: `y = y_norm * output_std + output_mean`
   - With `output_std ≈ 700mm` and random `y_norm ≈ O(10^10)`, output becomes enormous

2. **Physics Loss Explosion**
   - **IC Loss:** `MSE(y_at_z0, initial_state)` where y_at_z0 ~ 10^13 and initial_state ~ 100mm
   - Squared error: `(10^13 - 100)^2 ≈ 10^26`
   - **PDE Loss:** Involves computing gradients, magnetic field, and squared residuals
   - Chain rule through bad values → overflow to Inf

3. **Gradient Catastrophe**
   - Backpropagating through `loss = 10^26` or `loss = inf` produces NaN gradients
   - Optimizer step with NaN gradients corrupts all weights
   - Model is unrecoverable after first batch

4. **Scale Mismatch**
   - Position errors in mm (~0.05-500mm) vs slope errors (~0.001-0.1)
   - PDE residuals mix different physical quantities without normalization
   - Physics loss dominates data loss by >100x even when "working"

## Fixes Applied

### 1. Input Normalization Fix (`architectures.py` - CRITICAL)

The **root cause** was a design inconsistency in how the z/dz column is handled:

```python
# Training data has: [x0, y0, tx0, ty0, qop, dz] where dz=8000mm (constant)
# PINN forward pass replaces column 5 with z_frac (0-1) for trajectory queries
# But normalization was computed on dz=8000, so:
#   normalized_z = (1.0 - 8000) / 1e-8 = -8×10¹¹  ← DISASTER!
```

**Fix:** Override `set_normalization` in PINN to skip normalizing column 5:

```python
def set_normalization(self, X, Y, eps=1e-8):
    self.input_mean = X.mean(dim=0).clone()
    self.input_std = X.std(dim=0).clone() + eps
    # ... output normalization ...
    
    # Don't normalize column 5 (z/dz) - it's replaced with z_frac during forward
    self.input_mean[5] = 0.0
    self.input_std[5] = 1.0
```

This ensures z_frac values (0-1) pass through unchanged to the network.

### 2. Stable Weight Initialization (`architectures.py`)

```python
def _init_weights(self) -> None:
    for m in self.network.modules():
        if isinstance(m, nn.Linear):
            is_output = (m.out_features == 4)
            if is_output:
                # Output layer: near-zero → denormalized output ≈ output_mean
                nn.init.uniform_(m.weight, -0.001, 0.001)
            else:
                # Hidden layers: smaller than default for stability
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

**Rationale:** With near-zero output layer weights, the initial network output is close to zero. After denormalization, this gives predictions near `output_mean` (the average target), which is a reasonable starting point.

### 2. Normalized Physics Loss (`architectures.py`)

```python
# Characteristic scales for normalization
pos_scale = 500.0   # mm - typical position variation
slope_scale = 0.1   # typical slope magnitude
curvature_scale = 1e-5  # typical dtx/dz magnitude

# Normalized IC loss
ic_pos_err = ((y_at_z0[:, :2] - initial_state[:, :2]) / pos_scale).pow(2).mean()
ic_slope_err = ((y_at_z0[:, 2:] - initial_state[:, 2:]) / slope_scale).pow(2).mean()
ic_loss = ic_pos_err + ic_slope_err
```

**Rationale:** Dividing by characteristic scales makes all loss components dimensionless and ~O(1), preventing any single term from dominating.

### 3. Value Clamping (`architectures.py`)

```python
# Clamp predictions to physical range
y_clamped[:, 0] = y[:, 0].clamp(-3000, 3000)  # x position in mm
y_clamped[:, 1] = y[:, 1].clamp(-3000, 3000)  # y position in mm
y_clamped[:, 2] = y[:, 2].clamp(-1, 1)        # tx slope
y_clamped[:, 3] = y[:, 3].clamp(-1, 1)        # ty slope
```

**Rationale:** Prevents magnetic field evaluation at unphysical positions (which could return garbage or crash), and limits the domain of squared residuals.

### 4. NaN/Inf Protection (`architectures.py` and `train.py`)

```python
# In compute_physics_loss:
if torch.isnan(ic_loss) or torch.isinf(ic_loss):
    ic_loss = torch.tensor(0.0, device=device, requires_grad=True)

# In train_epoch:
if torch.isnan(loss) or torch.isinf(loss):
    n_skipped += 1
    continue  # Skip this batch
```

**Rationale:** Instead of crashing, skip problematic batches and continue training. Most batches should be fine once other fixes are in place.

### 5. Gradient Clipping (`train.py`)

```python
# After loss.backward()
if grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**Rationale:** Even with all other fixes, occasional large gradients can occur. Clipping at `max_norm=1.0` prevents any single step from destabilizing the model.

### 6. Physics Loss Warmup (`train.py`)

```python
# Physics loss warmup: ramp from 0 to 1 over warmup_epochs
warmup_epochs = config.get('physics_warmup_epochs', 10)
if epoch < warmup_epochs:
    physics_scale = epoch / warmup_epochs
else:
    physics_scale = 1.0

# Apply scaled physics loss
loss = data_loss + physics_scale * physics_loss
```

**Rationale:** Allow the network to first learn a reasonable approximation from data alone. Once it can make sensible predictions, gradually introduce physics constraints. This prevents the "cold start" problem where random predictions create huge physics losses.

## New Configuration Options

Added to `DEFAULT_CONFIG` in `train.py`:

```python
'physics_warmup_epochs': 10,  # Gradually increase physics loss over N epochs
'grad_clip': 1.0,             # Gradient clipping threshold (0 to disable)
```

## Why Different Models Behaved Differently

| Model | Physics Loss | Behavior | Explanation |
|-------|--------------|----------|-------------|
| MLP | None | Works ✓ | Only data loss, no explosion mechanism |
| PINN | IC + PDE | Fails ✗ | Full physics loss explodes immediately |
| PINN (λ=0) | None | Starts bad, then crashes | No physics but output still goes wild |
| RK_PINN | RK-integrated | Struggles but survives | RK structure provides implicit regularization |

The RK_PINN architecture is more stable because:
1. It predicts at multiple intermediate z points (not just endpoints)
2. The RK combination weights provide ensemble-like averaging
3. Errors at one stage don't fully propagate to the final output

## Testing the Fix

Run the debug script to verify:

```bash
cd experiments/next_generation
conda activate TE
python debug_pinn.py
```

Expected output after fix:
- Forward output: values ~O(100) instead of ~O(10^13)
- IC loss: ~O(1) instead of ~O(10^26)
- PDE loss: finite value instead of inf
- Gradients: reasonable norms (~0.01-10) instead of NaN

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.
2. Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing.
