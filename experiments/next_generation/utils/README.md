# Utilities

This directory contains Python utilities for track propagation and field calculations.

## Active Files

### `magnetic_field.py` - Unified Magnetic Field Module ⚠️ CRITICAL

**Status: ACTIVE AND IN USE**

This module provides the SINGLE source of truth for the magnetic field used in:
- Physics-informed neural networks (PINN, RK-PINN) via `InterpolatedFieldTorch`
- Training data generation via `InterpolatedFieldNumpy` / `InterpolatedField`
- Validation and testing

**Key Classes:**
- `InterpolatedFieldTorch` - PyTorch-compatible trilinear interpolation of real field map (twodip.rtf)
- `InterpolatedFieldNumpy` / `InterpolatedField` - NumPy versions for data generation
- `GaussianFieldTorch` / `GaussianFieldNumpy` - Analytical Gaussian approximation (for quick prototyping only)
- `get_field_numpy()` / `get_field_torch()` - Factory functions

**Critical Note:** PINN and RK_PINN models MUST use `InterpolatedFieldTorch` (the real field map)
for accurate physics loss computation. Using `GaussianFieldTorch` will cause systematic errors.

```python
# Usage in training (PyTorch)
from utils.magnetic_field import InterpolatedFieldTorch

field = InterpolatedFieldTorch('/path/to/twodip.rtf')
Bx, By, Bz = field(x, y, z)  # x, y, z are torch tensors

# Usage in data generation (NumPy)
from utils.magnetic_field import get_field_numpy

field = get_field_numpy(use_interpolated=True)
Bx, By, Bz = field(x, y, z)  # x, y, z are numpy arrays
```

---

### `rk4_propagator.py` - Python RK4 Integrator

**Status: ACTIVE AND IN USE**

Fourth-order Runge-Kutta integrator for track propagation in the LHCb magnetic field.
Used for training data generation.

**Key Classes:**
- `RK4Integrator` - Main integrator class with adaptive/fixed step integration

```python
from utils.rk4_propagator import RK4Integrator

# Create integrator with real field map
integrator = RK4Integrator(step_size=10.0, use_interpolated_field=True)

# Propagate a track
# state = [x, y, tx, ty, qop]
state_final = integrator.propagate(state_initial, z_start=3000, z_end=9000)

# Get intermediate trajectory
trajectory = integrator.propagate_with_trajectory(state_initial, z_start, z_end)
```

---

## File Summary

| File | Description | Status |
|------|-------------|--------|
| `magnetic_field.py` | Unified field module (PyTorch & NumPy) | ✅ Active |
| `rk4_propagator.py` | RK4 track propagator | ✅ Active |

---

**Status:** Active  
**Last Updated:** January 2026  
**Key Files:** Both `magnetic_field.py` and `rk4_propagator.py` are in active use
