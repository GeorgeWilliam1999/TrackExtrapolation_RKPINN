# Physics-Informed Neural Network (PINN) for Track Extrapolation

**An In-Depth Technical Explanation**

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Track Extrapolation Problem](#the-track-extrapolation-problem)
3. [Why Standard MLPs Aren't Enough](#why-standard-mlps-arent-enough)
4. [PINN Architecture Overview](#pinn-architecture-overview)
5. [The Residual Formulation](#the-residual-formulation)
6. [Supervised Collocation Points](#supervised-collocation-points)
7. [Loss Function Design](#loss-function-design)
8. [Training Data Structure](#training-data-structure)
9. [Example Trajectories](#example-trajectories)
10. [Mathematical Details](#mathematical-details)
11. [Comparison: MLP vs PINN](#comparison-mlp-vs-pinn)
12. [V3 Training Results](#v3-training-results)

---

## Introduction

This document explains the Physics-Informed Neural Network (PINN) architecture used in V3 
for track extrapolation in the LHCb detector. The key innovation is a **residual formulation** 
that guarantees the Initial Condition (IC) is exactly satisfied.

---

## The Track Extrapolation Problem

### What is Track Extrapolation?

When charged particles traverse the LHCb detector, they bend in the magnetic field. 
Track extrapolation predicts where a particle will be after traveling a certain distance.

```
                    Magnetic Field B(x,y,z)
                           ↓ ↓ ↓
Initial State              Curved Path                Final State
(x₀, y₀, tx₀, ty₀, q/p)   ~~~~~~~~~~~~~~~~~~~~~~~~>   (x₁, y₁, tx₁, ty₁)
at z = z_start                                        at z = z_end
```

### State Variables

| Variable | Description | Units | Typical Range |
|----------|-------------|-------|---------------|
| x | Horizontal position | mm | [-300, 300] |
| y | Vertical position | mm | [-250, 250] |
| tx | Horizontal slope (dx/dz) | - | [-0.3, 0.3] |
| ty | Vertical slope (dy/dz) | - | [-0.25, 0.25] |
| q/p | Charge over momentum | 1/MeV | [-4e-4, 4e-4] |
| dz | Propagation distance | mm | [500, 12000] |

### The Governing Physics

Track motion is governed by the Lorentz force:

$$\frac{d\vec{p}}{dt} = q(\vec{E} + \vec{v} \times \vec{B})$$

In LHCb, we have negligible electric field ($\vec{E} \approx 0$), giving:

$$\frac{d}{dz}\begin{pmatrix} x \\ y \\ t_x \\ t_y \end{pmatrix} = \begin{pmatrix} t_x \\ t_y \\ \kappa \cdot Q(t_x, t_y, q/p, \vec{B}) \\ \kappa \cdot R(t_x, t_y, q/p, \vec{B}) \end{pmatrix}$$

where:
- $\kappa = \sqrt{1 + t_x^2 + t_y^2}$ is a geometric factor
- $Q, R$ are the magnetic bending terms

This is a **system of ODEs** that the neural network must approximate.

---

## Why Standard MLPs Aren't Enough

### The Problem with Direct Prediction

A standard MLP learns a direct mapping:

```
Input: (x₀, y₀, tx₀, ty₀, q/p, dz)  →  MLP  →  Output: (x₁, y₁, tx₁, ty₁)
```

This works for endpoints but has a fundamental limitation: **no trajectory consistency**.

### What Does "Trajectory Consistency" Mean?

If we query the state at z_frac=0.3 and z_frac=0.7, those points should lie on the SAME 
physical trajectory. A standard MLP has no mechanism to enforce this.

```
Ideal (Physics-Consistent):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z=0    ●──────────●────────────────●──────────────● z=dz
     z_frac=0   z_frac=0.3      z_frac=0.7    z_frac=1

MLP (Independent Predictions - May Not Be Consistent):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z=0    ●          ○                  ○            ● z=dz
       ↑          ↑                  ↑            ↑
   IC (input)   query 1           query 2      output
                (may not be on same trajectory!)
```

### Initial Condition Violation

Another critical issue: at z_frac=0, the output should **exactly equal the input** 
(we haven't moved yet!). Standard networks must learn this constraint from data.

In V1 experiments, the PINN failed to learn this:
```
z_frac=0.0 (should = input state):
  Input:  x = 207 mm
  Output: x = 2768 mm  ← WRONG! Should be 207!
```

---

## PINN Architecture Overview

### The Key Insight: Residual Formulation

Instead of predicting states directly, we predict **corrections** to the initial state:

$$\text{Output} = \text{InitialCondition} + z_{frac} \times \text{Correction}$$

This guarantees:
- At $z_{frac}=0$: Output = IC (exactly, mathematically guaranteed!)
- At $z_{frac}=1$: Output = IC + Correction (learns like an MLP)
- Intermediate $z_{frac}$: Linear interpolation (smooth trajectory)

### Architecture Diagram

```
                        Input: (x₀, y₀, tx₀, ty₀, q/p, dz)
                                       │
                                       ▼
                     ┌─────────────────────────────────────┐
                     │           Core Network              │
                     │  [256] → SiLU → [256] → SiLU → [4] │
                     │                                     │
                     │  Learns: state → correction mapping │
                     └─────────────────────────────────────┘
                                       │
                                       ▼
                            Correction: (Δx, Δy, Δtx, Δty)
                                       │
                                       │ × z_frac
                                       ▼
                        ┌──────────────────────────────────┐
           IC ────────► │    Output = IC + z_frac × Corr  │
       (x₀,y₀,tx₀,ty₀) │                                  │
                        └──────────────────────────────────┘
                                       │
                                       ▼
                           (x, y, tx, ty) at z_frac
```

### Why This Works

1. **Guaranteed IC**: At z_frac=0, the correction term vanishes: Output = IC + 0 × Correction = IC ✓

2. **Same Inference Cost as MLP**: At z_frac=1, Output = IC + Correction. The network computation 
   is identical to an MLP - just one forward pass!

3. **Smooth Interpolation**: For any z_frac ∈ [0,1], we get a point on the line connecting IC 
   and the final state. The network learns to make this line follow the physical trajectory.

---

## The Residual Formulation

### Mathematical Formulation

Given:
- Input state $\mathbf{s}_0 = (x_0, y_0, t_{x,0}, t_{y,0})$
- Step size $dz$
- Trajectory parameter $z_{frac} \in [0, 1]$

The PINN computes:

$$\mathbf{s}(z_{frac}) = \mathbf{s}_0 + z_{frac} \cdot f_\theta(\mathbf{s}_0, q/p, dz)$$

Where $f_\theta$ is the neural network with parameters $\theta$.

### Properties

1. **Initial Condition (z_frac = 0)**:
   $$\mathbf{s}(0) = \mathbf{s}_0 + 0 \cdot f_\theta(\cdot) = \mathbf{s}_0 \quad \checkmark$$

2. **Final State (z_frac = 1)**:
   $$\mathbf{s}(1) = \mathbf{s}_0 + f_\theta(\mathbf{s}_0, q/p, dz)$$
   The network must learn $f_\theta = \mathbf{s}_1 - \mathbf{s}_0$ (the change in state).

3. **Intermediate States (0 < z_frac < 1)**:
   Linear interpolation between $\mathbf{s}_0$ and $\mathbf{s}_1$.

### Why Linear Interpolation is (Usually) Good Enough

For short-to-medium propagation distances (typical in LHCb), the trajectory is close to linear 
in (x, y) vs z. The main non-linearity is in the slopes (tx, ty) due to magnetic bending, but 
even this is smooth. Linear interpolation provides a good first approximation.

For higher precision, one could use a higher-order interpolation, but this adds complexity 
without significant benefit for our use case.

---

## Supervised Collocation Points

### The Traditional PINN Approach (Physics Residual)

Classical PINNs enforce physics by minimizing the ODE residual:

$$\mathcal{L}_{physics} = \left\| \frac{\partial \mathbf{s}}{\partial z} - \mathbf{F}(\mathbf{s}, z) \right\|^2$$

where $\mathbf{F}$ is the physics model (Lorentz force equations).

**Problem**: This requires:
1. Differentiating through the network (expensive)
2. Computing the magnetic field at collocation points
3. Trusting the physics model to be correct

### The V3 Approach: Supervised Collocation

Instead of enforcing physics via residuals, we **supervise the intermediate states directly** 
using ground-truth trajectory data!

```
Traditional PINN:
  Network → Predict state at z_frac → Compute physics residual → Loss

V3 PINN:
  Network → Predict state at z_frac → Compare to TRUE state (from data) → Loss
                                              ↑
                                   Extracted from RK4 trajectories!
```

### Why This Works Better

1. **No physics model needed at training time** - the physics is encoded in the trajectory data
2. **Faster convergence** - direct supervision is stronger than soft physics constraints
3. **More robust** - no issues with magnetic field interpolation errors

### Data Structure for Supervised Collocation

```python
# Training batch structure
{
    'X':      [B, 6]        # Input: [x, y, tx, ty, q/p, dz]
    'Y':      [B, 4]        # Endpoint: [x, y, tx, ty] at z_frac=1
    'z_frac': [B, N_col]    # Collocation z_frac values (e.g., [0.2, 0.4, 0.6, 0.8])
    'Y_col':  [B, N_col, 4] # TRUE states at collocation points!
}
```

### Example: 5 Collocation Points

For a trajectory segment from z=2000 to z=8000 (dz=6000mm):

```
z_frac values: [0.2, 0.4, 0.6, 0.8, 1.0]
Actual z:      [3200, 4400, 5600, 6800, 8000]

         z=2000                                           z=8000
            ●────●────●────●────●────●────●────●────●────●
            IC   │    │    │    │    │    │    │    │   End
                 │    │    │    │    │
              z_frac: 0.2  0.4  0.6  0.8
              
At each ●, we have the TRUE state from RK4 integration!
```

---

## Loss Function Design

### V3 PINN Loss Components

The total loss has three terms:

$$\mathcal{L}_{total} = \lambda_{IC} \mathcal{L}_{IC} + \lambda_{end} \mathcal{L}_{endpoint} + \lambda_{col} \mathcal{L}_{collocation}$$

### 1. Initial Condition Loss ($\mathcal{L}_{IC}$)

$$\mathcal{L}_{IC} = \frac{1}{B} \sum_{i=1}^{B} \| \text{PINN}(\mathbf{x}_i, z_{frac}=0) - \mathbf{s}_{0,i} \|^2$$

For the residual architecture, this should be **exactly zero** (mathematically guaranteed).
We still compute it to verify the architecture is working correctly.

### 2. Endpoint Loss ($\mathcal{L}_{endpoint}$)

$$\mathcal{L}_{endpoint} = \frac{1}{B} \sum_{i=1}^{B} \| \text{PINN}(\mathbf{x}_i, z_{frac}=1) - \mathbf{y}_i \|^2$$

This is the standard supervised loss matching the network output to the ground-truth final state.

### 3. Collocation Loss ($\mathcal{L}_{collocation}$)

$$\mathcal{L}_{collocation} = \frac{1}{B \cdot N_{col}} \sum_{i=1}^{B} \sum_{j=1}^{N_{col}} \| \text{PINN}(\mathbf{x}_i, z_{frac,j}) - \mathbf{Y}_{col,i,j} \|^2$$

This enforces that the network predictions at intermediate points match the true trajectory.

### Loss Weights (V3 Configuration)

```python
lambda_IC = 10.0    # High weight - IC must be satisfied
lambda_end = 1.0    # Standard supervised learning
lambda_col = 1.0    # Trajectory consistency
```

### Expected Loss Values (Residual Architecture)

| Loss Component | Expected Value | Reason |
|---------------|----------------|--------|
| $\mathcal{L}_{IC}$ | ≈ 0 | Guaranteed by residual formulation |
| $\mathcal{L}_{endpoint}$ | > 0 | Must be learned |
| $\mathcal{L}_{collocation}$ | > 0 | Must be learned |

---

## Training Data Structure

### How Training Data is Generated

```
Step 1: Generate Full Trajectories
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use high-precision RK4 integration
- 5mm step size (very fine)
- Full LHCb z-range: 0 → 15000mm
- 10,000 diverse trajectories

Step 2: Extract Segments with Collocation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
From each trajectory, sample random segments:

Full trajectory (3000 points):
•••••••••••••••••••••••••••••••••••••••••••••••••••

Select random segment [z_start, z_end]:
          ╔═══════════════════════════════════╗
          ║  z_start              z_end       ║
          ║    ↓                    ↓         ║
•••••••••••●━━━━●━━━━●━━━━●━━━━●━━━━●••••••••
            │    │    │    │    │   │
            IC  col1 col2 col3 col4 END
            
Extract: (IC, [col1-col4], END) as one training sample
```

### Data File Contents

```python
# PINN training data (training_pinn_v3_col10.npz)
{
    'X':      shape (10000000, 6)   # [x, y, tx, ty, q/p, dz]
    'Y':      shape (10000000, 4)   # [x, y, tx, ty] at endpoint
    'z_frac': shape (10000000, 10)  # 10 collocation fractions
    'Y_col':  shape (10000000, 10, 4)  # States at collocation points
}
```

### Collocation Studies (V3)

We train multiple models with different numbers of collocation points:

| Model | Collocation Points | Data Size | Purpose |
|-------|-------------------|-----------|---------|
| col5 | 5 | 913 MB | Fast training baseline |
| col10 | 10 | 1.4 GB | Standard |
| col20 | 20 | 2.4 GB | Higher trajectory resolution |
| col50 | 50 | 5.2 GB | Dense trajectory supervision |

---

## Example Trajectories

### High-Momentum Straight Track (p ≈ 50 GeV)

```
                    z (mm)
    0        3000       6000       9000      12000
    │         │          │          │          │
    ▼         ▼          ▼          ▼          ▼
    
    ●─────────●──────────●──────────●──────────●  x ≈ constant
    │                                            
    │  High momentum → minimal bending
    │  tx, ty change by < 0.001
    │
    └→ Almost a straight line!

Input:  (x=50, y=30, tx=0.05, ty=0.02, q/p=2e-5)
                                           ↑
                                      p ≈ 50 GeV
                                      
Output: (x=650, y=270, tx=0.051, ty=0.021)
                          ↑
                     Tiny change in slopes
```

### Low-Momentum Curved Track (p ≈ 2 GeV)

```
                    z (mm)
    0        3000       6000       9000      12000
    │         │          │          │          │
    ▼         ▼          ▼          ▼          ▼
    
    ●                                            
     ╲                                          
      ╲         Low momentum → strong bending    
       ╲        tx changes significantly!        
        ●                                        
         ╲                                       
          ╲                                      
           ●                                     
            ╲                                    
             ●                                   
              ╲                                  
               ●───→ x (curved path)             

Input:  (x=50, y=30, tx=0.05, ty=0.02, q/p=5e-4)
                                           ↑
                                      p ≈ 2 GeV
                                      
Output: (x=-2800, y=270, tx=-1.2, ty=0.025)
                   ↑        ↑
              Large change  tx now negative!
              in x          (track bent back)
```

### Charge Matters (Positive vs Negative)

```
Positive charge (q/p > 0):
    ●                                    
     ╲  Bends one way                   
      ╲                                 
       ╲                                
        ●                               

Negative charge (q/p < 0):
    ●                                    
     ╱  Bends the OTHER way              
    ╱                                    
   ╱                                     
  ●                                      

Same |p|, same initial direction,
opposite curvature due to Lorentz force!
```

---

## Mathematical Details

### The Forward Pass (Implementation)

```python
def forward(self, state_dz, z_frac=None):
    """
    state_dz: [B, 6] = [x, y, tx, ty, q/p, dz]
    z_frac:   [B, N] = trajectory parameter(s)
    """
    # Extract initial condition
    IC = state_dz[:, :4]  # [B, 4]
    
    # Compute correction (core network)
    correction = self.core(state_dz)  # [B, 4]
    
    # Residual formulation
    if z_frac is None:
        z_frac = 1.0  # Default: endpoint
    
    output = IC + z_frac * correction
    
    return output
```

### Network Architecture Details

```
Layer 0: Linear(6 → 256)
  └─ Input: [x, y, tx, ty, q/p, dz] (6 features)
  └─ Output: 256 features
  └─ Parameters: 6×256 + 256 = 1,792

Layer 1: SiLU activation
  └─ SiLU(x) = x × sigmoid(x)

Layer 2: Linear(256 → 256)
  └─ Parameters: 256×256 + 256 = 65,792

Layer 3: SiLU activation

Layer 4: Linear(256 → 4)
  └─ Output: [Δx, Δy, Δtx, Δty] (correction)
  └─ Parameters: 256×4 + 4 = 1,028

Total Parameters: 68,612
```

### Gradient Flow

```
Loss
  │
  ├─► ∂L/∂(IC + z_frac × correction)
  │              │
  │              ├─► ∂L/∂IC = 0 (no grad through IC, it's input)
  │              │
  │              └─► z_frac × ∂L/∂correction
  │                            │
  │                            └─► Backprop through core network
  │
  └─► Updates network weights θ
```

The key insight: gradients flow through the correction term, not through IC. 
The network learns the **change** in state, not the state itself.

---

## Comparison: MLP vs PINN

### Architecture Comparison

| Aspect | Standard MLP | PINN Residual |
|--------|-------------|---------------|
| Input | (state, dz) | (state, dz) |
| Output | (final state) | (final state) |
| IC Guarantee | ❌ Must learn | ✅ Mathematical |
| Intermediate states | ❌ Not available | ✅ Via z_frac |
| Parameters | Same | Same |
| Inference cost | Same | Same (at z_frac=1) |

### Loss Comparison

| Loss Term | MLP | PINN |
|-----------|-----|------|
| Endpoint | ✅ MSE(pred, target) | ✅ MSE(pred, target) |
| IC | ❌ Not enforced | ✅ Guaranteed (≈0) |
| Collocation | ❌ Not available | ✅ Supervised |

### When to Use Each

**Use MLP when:**
- You only need endpoint predictions
- Training data has no collocation points
- Simpler implementation is preferred

**Use PINN when:**
- Trajectory consistency matters
- You want intermediate state predictions
- You have trajectory data (not just endpoints)

---

## V3 Training Results

### Current Status (Feb 2026)

| Model | Status | Val Loss | Notes |
|-------|--------|----------|-------|
| pinn_v3_res_256_col5 | ✅ Complete | 0.0117 | Baseline |
| pinn_v3_res_256_col10 | ✅ Complete | 0.0110 | Standard |
| pinn_v3_res_256_col20 | ✅ Complete | 0.0107 | Better trajectory fit |
| pinn_v3_res_256_col50 | 🔄 Training | - | Dense collocation |

### Loss Breakdown (col10 model)

```
Epoch 50/50:
  IC Loss:          0.0000  ← Guaranteed by residual architecture!
  Endpoint Loss:    0.0008  ← Main prediction error
  Collocation Loss: 0.0102  ← Trajectory consistency error
  Total Loss:       0.0110
```

### Observation: More Collocation Points Help

| Collocation Points | Val Loss |
|-------------------|----------|
| 5 | 0.0117 |
| 10 | 0.0110 |
| 20 | 0.0107 |

More supervision points → better trajectory learning → lower loss.

---

## Summary

The V3 PINN architecture combines several key innovations:

1. **Residual Formulation**: Output = IC + z_frac × Correction
   - Guarantees IC satisfaction
   - Same inference cost as MLP at z_frac=1

2. **Supervised Collocation**: Use true trajectory states instead of physics residuals
   - Faster convergence
   - No physics model needed at training time

3. **Variable dz Training**: Train with random dz ∈ [500, 12000]
   - Enables deployment across all LHCb extrapolation distances
   - Proper normalization statistics

The result is a neural network that:
- ✅ Exactly satisfies initial conditions
- ✅ Provides smooth trajectory interpolation
- ✅ Works for any propagation distance in the trained range
- ✅ Has the same inference cost as a simple MLP

---

*Document generated: February 2026*
*Author: G. Scriven*
