#!/usr/bin/env python3
"""
RK4 Track Propagator - Cannibalized from legacy code

This module provides a pure Python implementation of RK4 track propagation
with the LHCb magnetic field model. Used for generating training data.

Based on: legacy/old_python_scripts/ml_models/python/generate_training_data.py
Author: G. Scriven
Date: 2025-01-14 (cleaned up)
Updated: 2026-01-19 (field parameters from real field map fit)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Field Parameters (Fitted from real field map twodip.rtf)
# =============================================================================

# Default parameters from Gaussian fit to real field map at x=0, y=0
# Fit performed on z ∈ [2000, 10000] mm with RMS error = 0.0126 T
FITTED_B0 = -1.0182        # Tesla (negative = field points down)
FITTED_Z_CENTER = 5007.0   # mm
FITTED_Z_WIDTH = 1744.0    # mm


@dataclass
class FieldParameters:
    """
    LHCb magnetic field model parameters.
    
    Default values are fitted from the real field map (twodip.rtf).
    The fit has RMS error of 0.013 T (~1.3% of peak field).
    """
    B0: float = abs(FITTED_B0)     # Tesla, peak field strength (magnitude)
    z_center: float = FITTED_Z_CENTER  # mm, center of magnet
    z_halfwidth: float = FITTED_Z_WIDTH  # mm, characteristic width
    polarity: int = -1         # -1 for field down (default from fit), +1 for MagUp


class LHCbMagneticField:
    """
    Simplified analytical model of LHCb dipole magnet.
    
    The LHCb magnet is a warm dipole with main field component By (vertical).
    This model uses a Gaussian z-profile with transverse fringe corrections.
    
    Mathematical form:
        By(x,y,z) = polarity × B₀ × exp(-0.5×((z-zc)/σz)²) × [1 - 0.0001×(r⊥/1000)²]
        Bx(x,y,z) = -0.01 × By × (x/1000)
        Bz(x,y,z) = 0
    
    Note: This is an APPROXIMATION. Real field map is more complex but
    requires LHCb framework access. This model has been validated to produce
    ML models with 0.334mm mean error.
    """
    
    def __init__(self, params: Optional[FieldParameters] = None):
        self.params = params or FieldParameters()
    
    def get_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Get magnetic field components at position (x, y, z).
        
        Args:
            x, y, z: Position in mm
            
        Returns:
            (Bx, By, Bz) in Tesla
        """
        # Gaussian z-profile
        z_rel = (z - self.params.z_center) / self.params.z_halfwidth
        By_profile = np.exp(-0.5 * z_rel**2)
        
        # Transverse fringe field correction
        r_trans = np.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        
        # Main field component (vertical)
        By = self.params.polarity * self.params.B0 * By_profile * fringe_factor
        
        # Small horizontal component from non-uniformity
        Bx = -0.01 * By * (x / 1000.0)
        
        # Longitudinal component negligible
        Bz = 0.0
        
        return Bx, By, Bz
    
    def get_field_vectorized(self, x: np.ndarray, y: np.ndarray, 
                            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized version for batch processing."""
        z_rel = (z - self.params.z_center) / self.params.z_halfwidth
        By_profile = np.exp(-0.5 * z_rel**2)
        
        r_trans = np.sqrt(x**2 + y**2)
        fringe_factor = 1.0 - 0.0001 * (r_trans / 1000.0)**2
        
        By = self.params.polarity * self.params.B0 * By_profile * fringe_factor
        Bx = -0.01 * By * (x / 1000.0)
        Bz = np.zeros_like(x)
        
        return Bx, By, Bz


class RK4Integrator:
    """
    Fourth-order Runge-Kutta integrator for track propagation in magnetic field.
    
    Integrates the Lorentz force equations:
        dx/dz = tx
        dy/dz = ty
        dtx/dz = κ√(1+tx²+ty²) [tx·ty·Bx - (1+tx²)·By + ty·Bz]
        dty/dz = κ√(1+tx²+ty²) [(1+ty²)·Bx - tx·ty·By - tx·Bz]
        d(q/p)/dz = 0  (momentum constant in uniform field approximation)
    
    where κ = c · (q/p) with c = 2.99792458e-4 (units: mm·MeV/(ns·T) simplified)
    
    Unit analysis:
        - q/p in 1/MeV
        - B in Tesla  
        - Position in mm
        - c_light = 2.99792458e-4 gives dtx/dz in 1/mm (dimensionless slope change per mm)
        
    This matches LHCb C++ extrapolators (see TrackRungeKuttaExtrapolator.cpp)
    """
    
    def __init__(self, field: LHCbMagneticField, step_size: float = 10.0):
        """
        Initialize integrator.
        
        Args:
            field: Magnetic field model
            step_size: Integration step in mm (default 10mm, can be 5mm for accuracy)
        """
        self.field = field
        self.step_size = step_size
        # Speed of light factor for Lorentz force with q/p in 1/MeV, B in Tesla, z in mm
        # This matches the LHCb C++ extrapolators
        self.c_light = 2.99792458e-4
    
    def derivatives(self, state: np.ndarray, z: float) -> np.ndarray:
        """
        Compute derivatives of state vector at position z.
        
        Args:
            state: [x, y, tx, ty, qop] where:
                x, y: position (mm)
                tx, ty: slopes (dimensionless)
                qop: charge/momentum (1/MeV)
            z: longitudinal position (mm)
            
        Returns:
            [dx/dz, dy/dz, dtx/dz, dty/dz, d(qop)/dz]
        """
        x, y, tx, ty, qop = state
        
        # Get magnetic field at current position
        Bx, By, Bz = self.field.get_field(x, y, z)
        
        # Kinematic factor
        kappa = self.c_light * qop
        sqrt_term = np.sqrt(1 + tx**2 + ty**2)
        
        # Derivatives from Lorentz force
        dx_dz = tx
        dy_dz = ty
        dtx_dz = kappa * sqrt_term * (tx * ty * Bx - (1 + tx**2) * By + ty * Bz)
        dty_dz = kappa * sqrt_term * ((1 + ty**2) * Bx - tx * ty * By - tx * Bz)
        dqop_dz = 0.0  # Momentum conserved
        
        return np.array([dx_dz, dy_dz, dtx_dz, dty_dz, dqop_dz])
    
    def rk4_step(self, state: np.ndarray, z: float, dz: float) -> np.ndarray:
        """
        Single RK4 integration step.
        
        Args:
            state: Current state [x, y, tx, ty, qop]
            z: Current z position
            dz: Step size
            
        Returns:
            Updated state at z + dz
        """
        k1 = self.derivatives(state, z)
        k2 = self.derivatives(state + 0.5 * dz * k1, z + 0.5 * dz)
        k3 = self.derivatives(state + 0.5 * dz * k2, z + 0.5 * dz)
        k4 = self.derivatives(state + dz * k3, z + dz)
        
        return state + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def propagate(self, state: np.ndarray, z_start: float, z_end: float,
                  save_trajectory: bool = False) -> np.ndarray:
        """
        Propagate track from z_start to z_end using RK4.
        
        Args:
            state: Initial state [x, y, tx, ty, qop] at z_start
            z_start: Starting z position (mm)
            z_end: Target z position (mm)
            save_trajectory: If True, return full trajectory instead of just final state
            
        Returns:
            If save_trajectory=False: Final state [x, y, tx, ty, qop] at z_end
            If save_trajectory=True: Array of shape (n_steps+1, 6) where each row is
                                     [z, x, y, tx, ty, qop] at each integration step
        """
        current_state = state.copy()
        current_z = z_start
        dz = z_end - z_start
        step = self.step_size if dz > 0 else -self.step_size
        
        # Initialize trajectory storage if requested
        if save_trajectory:
            trajectory = [[current_z, *current_state]]
        
        # Suppress overflow warnings during integration
        with np.errstate(over='ignore', invalid='ignore'):
            # Integrate in steps
            while abs(current_z - z_end) > abs(step):
                current_state = self.rk4_step(current_state, current_z, step)
                current_z += step
                
                # Early exit if state becomes non-finite OR leaves acceptance
                if not np.all(np.isfinite(current_state)):
                    if save_trajectory:
                        return np.array(trajectory)  # Return partial trajectory
                    return current_state  # Will be caught and filtered out
                
                # Check if track leaves LHCb acceptance (±5m is generous)
                if abs(current_state[0]) > 5000 or abs(current_state[1]) > 5000:
                    if save_trajectory:
                        return np.array(trajectory)  # Return partial trajectory
                    return np.full(5, np.nan)  # Mark as escaped
                
                # Check if slopes become extreme (unphysical)
                if abs(current_state[2]) > 2.0 or abs(current_state[3]) > 2.0:
                    if save_trajectory:
                        return np.array(trajectory)  # Return partial trajectory
                    return np.full(5, np.nan)  # Mark as extreme curvature
                
                # Save trajectory point
                if save_trajectory:
                    trajectory.append([current_z, *current_state])
            
            # Final fractional step to exactly reach z_end
            remaining = z_end - current_z
            if abs(remaining) > 1e-6:  # Avoid tiny steps
                current_state = self.rk4_step(current_state, current_z, remaining)
                current_z = z_end
                if save_trajectory:
                    trajectory.append([current_z, *current_state])
        
        if save_trajectory:
            return np.array(trajectory)
        return current_state


def generate_random_track(p_range: Tuple[float, float] = (3.0, 100.0),
                         z_start: float = 4000.0,
                         charge: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Generate random track state for training data.
    
    Args:
        p_range: Momentum range in GeV (min, max). Default (3, 100) ensures
                 stable propagation over 8000mm without extreme curvature.
        z_start: Initial z position in mm
        charge: +1 or -1 (if None, random choice)
        
    Returns:
        state: [x, y, tx, ty, qop] at z_start
        momentum: Momentum in GeV
    """
    # Random momentum (log-uniform for better coverage)
    log_p_min, log_p_max = np.log(p_range[0]), np.log(p_range[1])
    momentum = np.exp(np.random.uniform(log_p_min, log_p_max))  # GeV
    
    # Random charge if not specified
    if charge is None:
        charge = np.random.choice([-1, +1])
    
    # q/p in 1/MeV
    qop = charge / (momentum * 1000.0)
    
    # Random position at z_start (LHCb acceptance, conservative)
    x = np.random.uniform(-300, 300)  # mm (reduced to avoid escapes)
    y = np.random.uniform(-250, 250)  # mm (reduced to avoid escapes)
    
    # Random slopes (conservative for long propagations)
    tx = np.random.uniform(-0.15, 0.15)  # Reduced from ±0.2
    ty = np.random.uniform(-0.15, 0.15)  # Reduced from ±0.2
    
    state = np.array([x, y, tx, ty, qop])
    return state, momentum


# Example usage
if __name__ == "__main__":
    print("RK4 Track Propagator - Test")
    print("=" * 60)
    
    # Setup
    field = LHCbMagneticField()
    integrator = RK4Integrator(field, step_size=5.0)  # 5mm steps
    
    # Generate random track
    state_initial, momentum = generate_random_track()
    z_start = 4000.0
    z_end = 12000.0
    
    print(f"\nInitial state at z={z_start}mm:")
    print(f"  Position: x={state_initial[0]:.1f}mm, y={state_initial[1]:.1f}mm")
    print(f"  Slopes: tx={state_initial[2]:.4f}, ty={state_initial[3]:.4f}")
    print(f"  Momentum: p={momentum:.2f} GeV, q/p={state_initial[4]:.6f} 1/MeV")
    
    # Propagate
    state_final = integrator.propagate(state_initial, z_start, z_end)
    
    print(f"\nFinal state at z={z_end}mm:")
    print(f"  Position: x={state_final[0]:.1f}mm, y={state_final[1]:.1f}mm")
    print(f"  Slopes: tx={state_final[2]:.4f}, ty={state_final[3]:.4f}")
    
    # Deflection
    deflection = np.sqrt((state_final[0] - state_initial[0])**2 + 
                        (state_final[1] - state_initial[1])**2)
    print(f"\nDeflection: {deflection:.1f}mm")
    print("=" * 60)
