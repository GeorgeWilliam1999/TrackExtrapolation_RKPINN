#!/usr/bin/env python3
"""
================================================================================
RK4 Track Propagator with Real Field Map
================================================================================

This module provides the RK4 integrator for track propagation using the
UNIFIED magnetic field module. This ensures consistency between:
- Training data generation
- Physics-informed neural network training  
- Validation

IMPORTANT: This uses the REAL field map (twodip.rtf) by default, NOT the
Gaussian approximation. This matches LHCb's C++ extrapolators.

Author: G. Scriven
Date: January 2026
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from magnetic_field import get_field_numpy, C_LIGHT, InterpolatedFieldNumpy, GaussianFieldNumpy


class RK4Integrator:
    """
    Fourth-order Runge-Kutta integrator for track propagation in magnetic field.
    
    Uses the unified field module to ensure consistency with PINN training.
    
    Equations of motion (Lorentz force in z-parameterization):
        dx/dz  = tx
        dy/dz  = ty
        dtx/dz = κ · N · [tx·ty·Bx - (1+tx²)·By + ty·Bz]
        dty/dz = κ · N · [(1+ty²)·Bx - tx·ty·By - tx·Bz]
        
    where κ = c × (q/p), N = √(1 + tx² + ty²), c = 2.99792458×10⁻⁴
    """
    
    def __init__(self, 
                 field=None,
                 step_size: float = 10.0,
                 use_interpolated_field: bool = True,
                 polarity: int = 1):
        """
        Initialize RK4 integrator.
        
        Args:
            field: Field model (if None, creates one from unified module)
            step_size: Integration step in mm (5-10mm recommended)
            use_interpolated_field: If True and field=None, use real field map
            polarity: +1 for MagUp, -1 for MagDown (only used if field=None)
        """
        if field is not None:
            self.field = field
        else:
            self.field = get_field_numpy(
                use_interpolated=use_interpolated_field,
                polarity=polarity
            )
        
        self.step_size = step_size
        self.c_light = C_LIGHT
        
        # Log field type for debugging
        field_type = type(self.field).__name__
        print(f"RK4Integrator initialized with {field_type}, step_size={step_size}mm")
    
    def derivatives(self, state: np.ndarray, z: float) -> np.ndarray:
        """
        Compute state derivatives at position z.
        
        Args:
            state: [x, y, tx, ty, qop]
            z: Longitudinal position (mm)
            
        Returns:
            [dx/dz, dy/dz, dtx/dz, dty/dz, d(qop)/dz]
        """
        x, y, tx, ty, qop = state
        
        # Get magnetic field at current position
        Bx, By, Bz = self.field(x, y, z)
        
        # Kinematic factor
        kappa = self.c_light * qop
        N = np.sqrt(1 + tx**2 + ty**2)
        
        # Lorentz force equations
        dx_dz = tx
        dy_dz = ty
        dtx_dz = kappa * N * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = kappa * N * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        dqop_dz = 0.0  # Momentum conserved (no material interactions)
        
        return np.array([dx_dz, dy_dz, dtx_dz, dty_dz, dqop_dz])
    
    def rk4_step(self, state: np.ndarray, z: float, dz: float) -> np.ndarray:
        """Single RK4 integration step."""
        k1 = self.derivatives(state, z)
        k2 = self.derivatives(state + 0.5 * dz * k1, z + 0.5 * dz)
        k3 = self.derivatives(state + 0.5 * dz * k2, z + 0.5 * dz)
        k4 = self.derivatives(state + dz * k3, z + dz)
        
        return state + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def propagate(self, state: np.ndarray, z_start: float, z_end: float,
                  save_trajectory: bool = False) -> np.ndarray:
        """
        Propagate track from z_start to z_end.
        
        Args:
            state: Initial state [x, y, tx, ty, qop]
            z_start: Starting z position (mm)
            z_end: Target z position (mm)
            save_trajectory: If True, return full trajectory
            
        Returns:
            Final state [x, y, tx, ty, qop], or trajectory array if save_trajectory=True
        """
        current_state = state.copy()
        current_z = z_start
        dz = z_end - z_start
        step = self.step_size if dz > 0 else -self.step_size
        
        if save_trajectory:
            trajectory = [[current_z, *current_state]]
        
        with np.errstate(over='ignore', invalid='ignore'):
            while abs(current_z - z_end) > abs(step):
                current_state = self.rk4_step(current_state, current_z, step)
                current_z += step
                
                if not np.all(np.isfinite(current_state)):
                    if save_trajectory:
                        return np.array(trajectory)
                    return current_state
                
                # Acceptance cuts
                if abs(current_state[0]) > 5000 or abs(current_state[1]) > 5000:
                    if save_trajectory:
                        return np.array(trajectory)
                    return np.full(5, np.nan)
                
                if abs(current_state[2]) > 2.0 or abs(current_state[3]) > 2.0:
                    if save_trajectory:
                        return np.array(trajectory)
                    return np.full(5, np.nan)
                
                if save_trajectory:
                    trajectory.append([current_z, *current_state])
            
            # Final fractional step
            remaining = z_end - current_z
            if abs(remaining) > 1e-6:
                current_state = self.rk4_step(current_state, current_z, remaining)
                if save_trajectory:
                    trajectory.append([z_end, *current_state])
        
        if save_trajectory:
            return np.array(trajectory)
        return current_state


def generate_random_track(p_range: Tuple[float, float] = (0.5, 100.0),
                         z_start: float = 4000.0,
                         charge: Optional[int] = None,
                         x_range: Tuple[float, float] = (-1500, 1500),
                         y_range: Tuple[float, float] = (-1200, 1200),
                         tx_range: Tuple[float, float] = (-0.4, 0.4),
                         ty_range: Tuple[float, float] = (-0.35, 0.35)) -> Tuple[np.ndarray, float]:
    """
    Generate random track state for training data.
    
    Args:
        p_range: Momentum range in GeV (min, max)
        z_start: Initial z position (mm)
        charge: +1 or -1 (random if None)
        x_range: x position range in mm (default covers full LHCb acceptance)
        y_range: y position range in mm (default covers full LHCb acceptance)
        tx_range: tx slope range (default matches LHCb test coverage)
        ty_range: ty slope range (default matches LHCb test coverage)
        
    Returns:
        state: [x, y, tx, ty, qop]
        momentum: Momentum in GeV
    """
    # Log-uniform momentum sampling
    log_p_min, log_p_max = np.log(p_range[0]), np.log(p_range[1])
    momentum = np.exp(np.random.uniform(log_p_min, log_p_max))
    
    charge = charge or np.random.choice([-1, +1])
    qop = charge / (momentum * 1000.0)  # 1/MeV
    
    # Random position in full detector acceptance
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    
    # Random slopes covering full LHCb test range
    tx = np.random.uniform(tx_range[0], tx_range[1])
    ty = np.random.uniform(ty_range[0], ty_range[1])
    
    return np.array([x, y, tx, ty, qop]), momentum


# =============================================================================
# Backward Compatibility: LHCbMagneticField class
# =============================================================================

class LHCbMagneticField:
    """
    DEPRECATED: Use get_field_numpy() from magnetic_field module instead.
    
    This class is kept for backward compatibility with existing data generation code.
    It now wraps the unified field module.
    """
    
    def __init__(self, use_interpolated: bool = True):
        import warnings
        warnings.warn(
            "LHCbMagneticField is deprecated. Use get_field_numpy() from magnetic_field.py",
            DeprecationWarning
        )
        self._field = get_field_numpy(use_interpolated=use_interpolated)
        
        # Compatibility shim
        from dataclasses import dataclass
        @dataclass
        class Params:
            polarity: int = 1
        self.params = Params()
    
    def get_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Get field at (x, y, z)."""
        return self._field(x, y, z)
    
    def get_field_vectorized(self, x: np.ndarray, y: np.ndarray, 
                            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized version."""
        return self._field(x, y, z)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RK4 Track Propagator Test (with real field map)")
    print("=" * 60)
    
    # Create integrator with real field map
    integrator = RK4Integrator(step_size=5.0, use_interpolated_field=True)
    
    # Generate test track
    state_initial, momentum = generate_random_track(p_range=(5.0, 50.0))
    z_start, z_end = 4000.0, 12000.0
    
    print(f"\nInitial state at z={z_start}mm:")
    print(f"  Position: ({state_initial[0]:.1f}, {state_initial[1]:.1f}) mm")
    print(f"  Slopes: (tx={state_initial[2]:.4f}, ty={state_initial[3]:.4f})")
    print(f"  Momentum: {momentum:.2f} GeV")
    
    # Propagate
    state_final = integrator.propagate(state_initial, z_start, z_end)
    
    print(f"\nFinal state at z={z_end}mm:")
    print(f"  Position: ({state_final[0]:.1f}, {state_final[1]:.1f}) mm")
    print(f"  Slopes: (tx={state_final[2]:.4f}, ty={state_final[3]:.4f})")
    
    # Compare with Gaussian field
    print("\n--- Comparison: Interpolated vs Gaussian ---")
    integrator_gauss = RK4Integrator(step_size=5.0, use_interpolated_field=False)
    state_final_gauss = integrator_gauss.propagate(state_initial, z_start, z_end)
    
    dx = state_final[0] - state_final_gauss[0]
    dy = state_final[1] - state_final_gauss[1]
    print(f"Position difference (interp - gauss): Δx={dx:.3f}mm, Δy={dy:.3f}mm")
    print(f"This difference shows why using the real field map matters!")
    
    print("\n" + "=" * 60)
