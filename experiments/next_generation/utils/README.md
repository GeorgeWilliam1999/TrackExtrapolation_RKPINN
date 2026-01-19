# Utilities - Archived

This directory previously contained Python implementations of track propagation utilities.

## Status: Archived

All Python-based propagators have been **archived** in favor of using the battle-tested 
C++ LHCb extrapolators directly.

### Why Archived?

The Python RK4 propagator (`rk4_propagator.py`) experienced numerical instability:
- Track slopes exploded exponentially
- 100% failure rate for all tested configurations
- Root cause: Sign errors and/or unit mismatches in Lorentz force equations

Rather than debug and reimplement complex physics, we now use the proven C++ extrapolators 
from the main LHCb codebase.

### Archived Contents

Located in `archived/` subdirectory:

- **rk4_propagator.py** - Python RK4 integrator (broken, reference only)
  - Simplified LHCb magnetic field model
  - Fourth-order Runge-Kutta integration
  - Random track generation

**Do not use these files** - they are kept for reference only.

### Current Approach

For data generation, use C++ extrapolators via wrapper:

```python
# See: experiments/next_generation/data_generation/generate_cpp_data.py

from subprocess import run

# Call C++ extrapolator
result = run([
    './run', 'TrackExtrapolators.TrackExtrapolatorTesterSOA',
    '--extrapolator', 'BogackiShampine3',
    '--n-tracks', '10000'
])

# Parse output
# ... (see generate_cpp_data.py for full implementation)
```

### Future Work

If Python utilities are needed in the future, consider:

1. **pybind11 wrapper** - Directly expose C++ extrapolators to Python
2. **Validation suite** - Compare any new Python implementation against C++ reference
3. **Unit tests** - Test individual components with known analytical solutions

---

**Status:** Archived (2025-01-14)  
**Replaced by:** C++ extrapolators in `src/`  
**Reference:** See `data_generation/generate_cpp_data.py` for current approach
