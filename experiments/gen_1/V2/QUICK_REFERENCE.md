# Quick Reference: PINN Architecture Issue

## The Problem in One Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ V2 PINN Architecture (Current - BROKEN)                         │
└─────────────────────────────────────────────────────────────────┘

Input: [x₀, y₀, tx₀, ty₀, q/p] → Encoder → features → correction (c)
                                                           ↓
                            z_frac ────────────────→ multiply
                                                           ↓
Output: state(z) = baseline(z) + z_frac · c

❌ Single correction vector 'c' per track (position-independent!)
❌ Linear interpolation between start and end
❌ Cannot capture varying curvature in variable B field


┌─────────────────────────────────────────────────────────────────┐
│ MLP Architecture (Current - WORKS)                              │
└─────────────────────────────────────────────────────────────────┘

Input: [x₀, y₀, tx₀, ty₀, q/p, z_frac] → Network → Output
                              ↑
                      Position information!

✓ z_frac as explicit input → position-dependent predictions
✓ No architectural constraints
✓ Can learn arbitrary nonlinear relationships


┌─────────────────────────────────────────────────────────────────┐
│ Fixed PINN (Proposed - Option 1)                                │
└─────────────────────────────────────────────────────────────────┘

Input: [x₀, y₀, tx₀, ty₀, q/p, z_frac] → Encoder → correction(z)
                              ↑                          ↓
                      Position info!          (different at each z!)
                                                          ↓
Output: state(z) = baseline(z) + correction(z)

✓ Position-dependent corrections
✓ Maintains IC satisfaction
✓ Physics loss becomes meaningful
```

## The Physics: Why Position Matters

```
LHCb Magnetic Field (Dipole):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

z (mm)   |  By (T)  |  Curvature κ
─────────┼──────────┼──────────────
   0     |   1.1    |   High   ▓▓▓
 2000    |   0.9    |   ↓      ▓▓
 4000    |   0.7    |   ↓      ▓
 6000    |   0.5    |   ↓      ░
 8000    |   0.4    |   Low    ░

Field varies 3× → Curvature varies 3×
Linear correction CANNOT capture this!
```

## The Evidence

### Performance Comparison

| Model Type | Position Error (mm) | Has z Input? |
|:-----------|--------------------:|:------------:|
| MLP        | 0.08                | ✓            |
| PINN       | 0.15                | ❌           |
| Factor     | 2× worse           |              |

### Trajectory Analysis

At z_frac = 0%, 25%, 50%, 75%, 100%:

```
PINN predictions: ─────o─────o─────o─────o─────o  (perfectly linear!)
                  
RK4 ground truth: ─────╮                          (nonlinear curvature)
                        ╰────╮
                             ╰───╮
                                 ╰──o
                                 
Error grows systematically with |prediction - truth|
```

## The Fix: 4 Options

### Option 1: Add z_frac to encoder ⭐ **Simplest**
```python
features = encoder([x0, y0, tx0, ty0, qop, z_frac])  # 6 inputs instead of 5
```
- Minimal code change
- Network can learn position-dependent corrections
- Retrain from scratch

### Option 2: Polynomial residual
```python
state(z) = baseline(z) + z·c₁ + z²·c₂ + z³·c₃
```
- Captures acceleration and jerk
- Natural for trajectory physics
- V3/V4 use this approach

### Option 3: True PDE-residual PINN
```python
ds/dz = autograd.grad(s(z), z)  # Automatic differentiation
loss = ||ds/dz - lorentz_force(s, z, B)||²
```
- No architectural constraints
- True physics enforcement
- More expensive training

### Option 4: Compositional architecture  
```python
s₁ = model(s₀, Δz₁)   # Short steps
s₂ = model(s₁, Δz₂)   # Chain together
s₃ = model(s₂, Δz₃)
```
- Each step adapts to current state
- Error scales as (Δz)³ per step

## Key Insight

> **Physics-informed neural networks require careful co-design of physics constraints 
> and architectural expressivity. Too much constraint can be worse than too little.**

The V2 PINN:
- ✅ Perfectly enforces initial conditions
- ✅ Explicitly incorporates Lorentz force
- ❌ Cannot represent the solution (linear ansatz too restrictive)

**Result: Architecture bottleneck overrides physics knowledge.**

## Variable Fields Need Position Information

### Uniform Field (hypothetical)
- Constant curvature → circular arc
- Linear corrections might work
- Position less critical

### LHCb Variable Field (actual)
- Curvature κ(z) ∝ By(z) varies smoothly
- Spiral with varying pitch
- **Position information essential**

### Why MLPs Win
MLPs don't enforce physics explicitly, but their **flexibility** > PINN's **rigidity** 
when the architecture can't represent the physics anyway.

## References

- **Full Analysis:** [PINN_ARCHITECTURE_ISSUE.md](./PINN_ARCHITECTURE_ISSUE.md)
- **Jupyter Analysis:** [physics_exploration.ipynb](../physics_exploration.ipynb) - Section 8
- **Performance Data:** [v2_model_analysis.ipynb](./analysis/v2_model_analysis.ipynb)
- **V2 README:** [README.md](./README.md)

## Bottom Line

```
┌───────────────────────────────────────────────────────────────────┐
│  NOT a training problem                                           │
│  NOT a hyperparameter problem                                     │
│  NOT a collocation point problem                                  │
│                                                                    │
│  IT IS A FUNDAMENTAL CAPACITY PROBLEM                             │
│                                                                    │
│  The architecture must be changed.                                │
└───────────────────────────────────────────────────────────────────┘
```

Training longer or adding more collocation points won't help.
The linear ansatz cannot represent nonlinear physics.

**Action:** Implement one of the 4 fixes above and retrain.

---
*Created: February 20, 2026*  
*Analysis: V2 PINN failure investigation*
