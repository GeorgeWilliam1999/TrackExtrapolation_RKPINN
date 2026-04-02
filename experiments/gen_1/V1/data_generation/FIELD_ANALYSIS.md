# Magnetic Field Analysis — `twodip.rtf`

Summary of the LHCb dipole field map analysis from `field_analysis.ipynb`.

## Grid Specification

| Property | Value |
|----------|-------|
| File | `experiments/field_maps/twodip.rtf` |
| Format | ASCII, 6 columns (x, y, z, Bx, By, Bz) |
| Grid points | 81 × 81 × 146 = 957,906 |
| Spacing | 100 mm (uniform) |
| x range | −4000 to +4000 mm |
| y range | −4000 to +4000 mm |
| z range | −500 to +14,000 mm |

## Field Characteristics (On-Axis)

| Property | Value |
|----------|-------|
| Peak By | −1.032 T at z = 5000 mm |
| FWHM | 4000 mm (z = 3000 to 7000 mm) |
| Active region (|B| > 0.01 T) | z = −300 to 14,000 mm |
| Field integral 98% | z = 900 to 11,400 mm |
| Dominant component | By (dipole bending field) |

## Acceptance Region (|x| < 500, |y| < 500 mm)

| Property | Value |
|----------|-------|
| Max |B| | 1.13 T |
| Transverse variation | 14.3% within acceptance |
| By(x=0, y=0, z=5000) | −1.032 T |

## Grid-Edge Artifacts

Max |By| reaches 33,024 T at y = ±3900 mm near z = 5000 mm. These are
non-physical extrapolation artifacts at the grid boundaries and do **not**
affect the physics region (|x,y| < 500 mm).

## Gaussian Approximation

Fitted $B_y(z) = B_0 \exp\!\bigl(-(z - z_c)^2 / 2\sigma^2\bigr)$ to on-axis profile:

| Parameter | Value |
|-----------|-------|
| B₀ | −1.0153 T |
| z_center | 5000 mm |
| z_width (σ) | 1757 mm |
| RMS error | 12.6 mT (1.2%) |
| Max error | 19.5 mT |

**Conclusion**: Not suitable for precision training data. Use the real interpolated field map.

## Data Generation Parameters (Chosen)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| z range | [0, 14,000] mm | Covers full active region |
| dz range | [100, 10,000] mm | Variable — fixes old V1 normalization bug |
| x range | ±300 mm | LHCb VELO acceptance |
| y range | ±250 mm | LHCb VELO acceptance |
| tx range | ±0.3 | Typical track slopes |
| ty range | ±0.25 | Typical track slopes |
| Momentum | [1, 100] GeV (log-uniform) | Covers physics range |
| RK4 step | 5 mm | Sub-mm precision |
| Polarity | −1 (MagDown) | Standard running |
