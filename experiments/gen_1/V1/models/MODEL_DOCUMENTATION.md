# Model Architecture Documentation

## Overview

This document describes the three neural network architectures for track extrapolation:

| Model | Data Loss | IC Loss | PDE Loss | Architecture |
|-------|-----------|---------|----------|--------------|
| **MLP** | ✓ MSE | ❌ | ❌ | Simple feedforward |
| **PINN** | ✓ MSE | ✓ | ✓ | Single network with autodiff |
| **RK_PINN** | ✓ MSE | ✓ | ✓ | Multi-head with RK-style weighting |

---

## Training Data Format

All models use the same input/output format:

**Input X** `[batch, 6]`:
| Index | Name | Description | Units |
|-------|------|-------------|-------|
| 0 | x₀ | Initial x position | mm |
| 1 | y₀ | Initial y position | mm |
| 2 | tx₀ | Initial x-slope (dx/dz) | - |
| 3 | ty₀ | Initial y-slope (dy/dz) | - |
| 4 | q/p | Charge / momentum | 1/MeV |
| 5 | dz | Propagation distance | mm |

**Output Y** `[batch, 4]`:
| Index | Name | Description | Units |
|-------|------|-------------|-------|
| 0 | x | Final x position | mm |
| 1 | y | Final y position | mm |
| 2 | tx | Final x-slope | - |
| 3 | ty | Final y-slope | - |

---

## 1. MLP (Multi-Layer Perceptron)

### Architecture
```
Input(6) → [Linear → SiLU]×N → Linear(4) → Output
```

Example with `hidden_dims=[128, 128, 64]`:
```
[x₀,y₀,tx₀,ty₀,q/p,dz] → Linear(6→128) → SiLU 
                       → Linear(128→128) → SiLU 
                       → Linear(128→64) → SiLU 
                       → Linear(64→4) → [x,y,tx,ty]
```

### Loss Function
```
L = MSE(y_pred, y_true)
```
Pure data-driven. No physics constraints.

### Use Case
- Fast inference baseline
- When physics isn't critical
- Comparison benchmark

---

## 2. PINN (Physics-Informed Neural Network)

### Architecture
Same feedforward structure as MLP, but with `tanh` activation (smoother for autodiff):
```
Input(6) → [Linear → Tanh]×N → Linear(4) → Output
```

### Key Difference: Network Interprets Input Differently
- **MLP**: Last input is `dz` (fixed propagation distance)
- **PINN**: Last input is `z_query ∈ [0, 1]` (normalized query position)

The PINN learns the **entire trajectory** as a continuous function y(z).

### Loss Function
```
L = L_data + λ_IC · L_IC + λ_PDE · L_PDE
```

#### L_IC (Initial Condition Loss)
```python
# Query network at z=0
y_at_z0 = network([x₀, y₀, tx₀, ty₀, q/p, z=0])

# Must match initial state
L_IC = MSE(y_at_z0, [x₀, y₀, tx₀, ty₀])
```

#### L_PDE (Physics Residual Loss)
At each collocation point z_i ∈ {0.1, 0.2, ..., 1.0}:

```python
# Forward pass with gradient tracking
y = network([x₀, y₀, tx₀, ty₀, q/p, z_i])

# Compute ∂y/∂z via autodiff
dy_dz = torch.autograd.grad(y, z_i)

# Expected derivatives from Lorentz equations
dy_dz_expected = lorentz_equations(y, B(z_i), κ)

# Residual
L_PDE += ||dy_dz - dy_dz_expected||²
```

### The Lorentz Force Equations (LHCb Form)
```
dx/dz  = tx
dy/dz  = ty
dtx/dz = κ · √(1+tx²+ty²) · [tx·ty·Bx - (1+tx²)·By + ty·Bz]
dty/dz = κ · √(1+tx²+ty²) · [(1+ty²)·Bx - tx·ty·By - tx·Bz]
```
where `κ = (q/p) × c_light = (q/p) × 2.998×10⁻⁴`

---

## 3. RK_PINN (Runge-Kutta PINN)

### Key Idea
Instead of a single network predicting the full trajectory, RK_PINN uses:
1. **Shared feature extractor** - processes initial state
2. **Multiple stage heads** - each specialized for a specific z position
3. **Learnable combination weights** - initialized to RK4 coefficients

### Architecture Diagram
```
                                         ┌─────────────────────────────────────┐
                                         │  Stage Head 1 (z=0.25)              │
                                         │  Linear(129→64) → Tanh → Linear(4)  │──→ [x,y,tx,ty]₁
                                         └─────────────────────────────────────┘
                                                          ↑
[x₀,y₀,tx₀,ty₀,q/p,dz]                                   │ cat([features, 0.25])
        │                                                │
        ▼                               ┌────────────────┴─────────────────────┐
┌──────────────────────┐                │  Stage Head 2 (z=0.50)              │
│  Feature Extractor   │                │  Linear(129→64) → Tanh → Linear(4)  │──→ [x,y,tx,ty]₂
│  Linear(6→128)→Tanh  │───features───→└─────────────────────────────────────┘
│  Linear(128→128)→Tanh│     (128)                       │
└──────────────────────┘                                 │ cat([features, 0.50])
                                                         │
                                        ┌────────────────┴─────────────────────┐
                                        │  Stage Head 3 (z=0.75)              │
                                        │  Linear(129→64) → Tanh → Linear(4)  │──→ [x,y,tx,ty]₃
                                        └─────────────────────────────────────┘
                                                         │
                                                         │ cat([features, 0.75])
                                                         │
                                        ┌────────────────┴─────────────────────┐
                                        │  Stage Head 4 (z=1.00)              │
                                        │  Linear(129→64) → Tanh → Linear(4)  │──→ [x,y,tx,ty]₄
                                        └─────────────────────────────────────┘
```

### How the Heads Work

Each stage head receives:
- **Shared features** (128 dims) - extracted from initial state
- **z_fraction** (1 dim) - the normalized position this head is responsible for

```python
# For stage i with z_fraction = 0.25, 0.50, 0.75, or 1.0
z_input = torch.full((batch, 1), z_fraction)
stage_input = torch.cat([features, z_input], dim=1)  # [batch, 129]
stage_output = stage_head(stage_input)                # [batch, 4]
```

### Stage Combination (RK4-Style Weighting)

The final output is a **weighted sum** of stage predictions:

```python
# Initial weights (RK4 coefficients): [1, 2, 2, 1] / 6
weights = softmax(learnable_weights)  # Ensures sum = 1

final_output = w₁·stage₁ + w₂·stage₂ + w₃·stage₃ + w₄·stage₄
```

**Why RK4 initialization?**
In classical RK4 integration:
```
y_{n+1} = y_n + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```
The weights [1,2,2,1]/6 give more importance to midpoint evaluations.

**Why learnable?**
The optimal weights may differ from RK4 because:
- The network learns non-local information
- The physics loss already enforces correctness at each stage
- The data distribution may favor different weighting

### Loss Function
```
L = L_data + λ_IC · L_IC + λ_PDE · L_PDE
```

#### L_IC (Initial Condition Loss)
```python
# Query stage head 1 at z=0
z_zero = torch.zeros((batch, 1))
stage_input = torch.cat([features, z_zero], dim=1)
y_at_z0 = stage_heads[0](stage_input)

L_IC = MSE(y_at_z0, [x₀, y₀, tx₀, ty₀])
```

#### L_PDE (Physics Loss at Each Stage)
```python
total_residual = 0
for stage_head, z_frac in zip(heads, [0.25, 0.5, 0.75, 1.0]):
    # Forward with gradient tracking
    z_input = tensor([z_frac], requires_grad=True)
    y_stage = stage_head(cat([features, z_input]))
    
    # Autodiff: ∂y/∂z
    dy_dz = autograd.grad(y_stage, z_input)
    
    # Compare against Lorentz equations at this z
    dy_dz_expected = lorentz_equations(y_stage, B(z_frac * dz), κ)
    
    total_residual += ||dy_dz - dy_dz_expected||²

L_PDE = total_residual / n_stages
```

### Why Multiple Heads Instead of One Network?

| Single Network (PINN) | Multiple Heads (RK_PINN) |
|-----------------------|--------------------------|
| One network learns entire z → y mapping | Each head specializes for one z region |
| Must interpolate between all z | Direct prediction at fixed z positions |
| More flexible (any z query) | More structured (RK-like evaluation) |
| May have difficulty with sharp features | Better gradient flow through stages |

---

## Comparison Table

| Aspect | MLP | PINN | RK_PINN |
|--------|-----|------|---------|
| **Network** | Single feedforward | Single feedforward | Shared extractor + 4 heads |
| **Query z** | N/A (fixed dz) | Any z ∈ [0,1] | Fixed {0.25, 0.5, 0.75, 1.0} |
| **IC Loss** | ❌ | ✓ at z=0 | ✓ at z=0 |
| **PDE Loss** | ❌ | ✓ at 10 collocation pts | ✓ at 4 stage positions |
| **Output** | Direct prediction | Network output at z=1 | Weighted sum of stages |
| **Physics** | None | Full autodiff | Full autodiff |
| **Activation** | SiLU | Tanh | Tanh |

---

## Hyperparameters

### Common
- `hidden_dims`: Network layer sizes (default: [128, 128, 64])
- `activation`: Activation function (mlp: 'silu', pinn/rk_pinn: 'tanh')
- `batch_size`: Training batch size (default: 2048)
- `learning_rate`: AdamW learning rate (default: 1e-3)

### PINN/RK_PINN Specific
- `lambda_pde`: Weight for PDE residual loss (default: 1.0)
- `lambda_ic`: Weight for initial condition loss (default: 1.0)
- `n_collocation`: Number of interior collocation points (PINN only, default: 10)
- `n_stages`: Number of RK stages (RK_PINN only, default: 4)

---

## Usage Examples

```bash
# Train MLP baseline
python train.py --model mlp --preset medium --epochs 100

# Train PINN with physics
python train.py --model pinn --preset medium --lambda_pde 1.0 --lambda_ic 1.0 --epochs 100

# Train RK_PINN
python train.py --model rk_pinn --preset medium --lambda_pde 0.1 --lambda_ic 1.0 --epochs 100

# Sweep lambda_pde
for lambda in 0.01 0.1 1.0 10.0; do
    python train.py --model pinn --lambda_pde $lambda --name pinn_lambda_${lambda}
done
```

---

## Validation: Physics Equations Match LHCb

The Lorentz force equations were verified against `TrackRungeKuttaExtrapolator.cpp`:

```cpp
// LHCb C++ (lines 870-877)
const auto norm = std::sqrt(1 + tx2 + ty2);
const auto Ax = norm * (ty * (tx * Bx + Bz) - (1 + tx2) * By);
const auto Ay = norm * (-tx * (ty * By + Bz) + (1 + ty2) * Bx);
return { { tx, ty, qop * Ax, qop * Ay }, 0, 1 };  // dState/dz
```

This matches our Python implementation exactly. ✅
