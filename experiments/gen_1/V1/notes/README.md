# V1 Notes

Technical documentation and mathematical derivations.

## Documents

- `PINN_STABILITY_FIXES.md` - Fixes attempted for PINN training stability
- `experimental_protocol.pdf` - Experimental methodology
- `mathematical_derivations.pdf` - Physics equations and derivations

## Key Notes

### PINN IC Failure
V1 PINN models failed to satisfy Initial Condition constraints.
The physics loss remained constant while data loss decreased.

**Root cause**: Network ignored z_frac input.
**Solution**: Residual architecture in V2.
