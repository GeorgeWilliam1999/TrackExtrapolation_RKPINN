# V1 Utilities

Shared utility modules for V1 experiments.

## Modules

### `magnetic_field.py`
Magnetic field interpolation from real LHCb field map (twodip.rtf).

- `InterpolatedFieldTorch` - PyTorch-compatible trilinear interpolation
- `InterpolatedFieldNumpy` - NumPy version for data generation
- `GaussianFieldTorch` - Analytical Gaussian approximation

### `rk4_propagator.py`
Runge-Kutta 4th order track propagation.

### `physics_loss.py`
Physics loss functions for PINN training.

## Field Map

All field calculations use the real LHCb dipole map:
```
/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/field_maps/twodip.rtf
```

| Property | Value |
|----------|-------|
| Peak By | 1.03 T |
| z_center | 5007 mm |
| Grid size | 81×81×146 |
