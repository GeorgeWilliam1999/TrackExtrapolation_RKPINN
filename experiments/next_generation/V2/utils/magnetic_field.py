#!/usr/bin/env python3
"""
================================================================================
Unified Magnetic Field Module for LHCb Track Extrapolation
================================================================================

This module provides a SINGLE source of truth for the magnetic field used in:
- Training data generation (RK4 integrator)
- Physics-informed neural networks (PINN, RK-PINN)
- Validation and testing

Using different field models in data generation vs training will cause
systematic errors! Always use the same field model throughout.

Available Field Models
----------------------
1. InterpolatedField (RECOMMENDED for accuracy)
   - Trilinear interpolation of the real field map (twodip.rtf)
   - Interpolation error: O(h²) ~ 10⁻⁵ T for 100mm grid
   - Full 3D field: Bx, By, Bz all vary with (x, y, z)

2. GaussianField (for quick prototyping)
   - Analytical Gaussian approximation of By(z)
   - Fitted to real field map with RMS error ~0.013 T (~1.3%)
   - Only z-dependent, Bx ≈ 0, Bz = 0

Default Behavior
----------------
By default, this module uses the INTERPOLATED FIELD MAP if available.
This ensures consistency between data generation and model training.

Usage
-----
>>> from magnetic_field import get_field, get_field_numpy
>>> 
>>> # For PyTorch models (PINN training)
>>> field = get_field()  # Returns InterpolatedField or GaussianField
>>> Bx, By, Bz = field(x, y, z)  # x, y, z are torch tensors
>>> 
>>> # For numpy (data generation, RK4)
>>> field_np = get_field_numpy()
>>> Bx, By, Bz = field_np(x, y, z)  # x, y, z are floats or numpy arrays

Author: G. Scriven
Date: January 2026
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Try to import torch for GPU-accelerated interpolation
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Only numpy field models will work.")


# =============================================================================
# Physical Constants
# =============================================================================

# Speed of light factor for Lorentz force calculation
# Converts (q/p in 1/MeV) × (B in Tesla) → curvature in 1/mm
C_LIGHT = 2.99792458e-4


# =============================================================================
# Default Field Map Path
# =============================================================================

# Path relative to this file
_THIS_DIR = Path(__file__).parent
_DEFAULT_FIELD_MAP = _THIS_DIR.parent.parent.parent / 'field_maps' / 'twodip.rtf'


# =============================================================================
# Gaussian Approximation Parameters (fitted from real field map)
# =============================================================================

@dataclass
class GaussianFieldParams:
    """Parameters for Gaussian field approximation, fitted from twodip.rtf."""
    B0: float = -1.0182        # Peak field strength (Tesla), negative = pointing down
    z_center: float = 5007.0   # Center of dipole magnet (mm)
    z_width: float = 1744.0    # Gaussian width parameter (mm)
    
    # Fit quality metrics (at x=0, y=0)
    fit_rms_error: float = 0.0126   # Tesla
    fit_max_error: float = 0.035    # Tesla


# =============================================================================
# NumPy Field Models (for data generation)
# =============================================================================

class GaussianFieldNumpy:
    """
    Gaussian approximation of LHCb magnetic field (NumPy version).
    
    APPROXIMATION with ~1.3% error. Use InterpolatedFieldNumpy for accuracy.
    """
    
    def __init__(self, params: Optional[GaussianFieldParams] = None, polarity: int = 1):
        """
        Initialize Gaussian field model.
        
        Args:
            params: Field parameters (default: fitted from twodip.rtf)
            polarity: +1 for MagUp, -1 for MagDown (flips field direction)
        """
        self.params = params or GaussianFieldParams()
        self.polarity = polarity
    
    def __call__(self, x: Union[float, np.ndarray], 
                 y: Union[float, np.ndarray], 
                 z: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get field at position (x, y, z).
        
        Note: This simplified model only depends on z.
        """
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        
        z_rel = (z - self.params.z_center) / self.params.z_width
        By = self.polarity * self.params.B0 * np.exp(-0.5 * z_rel**2)
        Bx = np.zeros_like(By)
        Bz = np.zeros_like(By)
        
        # Return scalars if input was scalar
        if z.size == 1:
            return float(Bx[0]), float(By[0]), float(Bz[0])
        return Bx, By, Bz


class InterpolatedFieldNumpy:
    """
    Trilinear interpolation of the real LHCb field map (NumPy version).
    
    This is the ACCURATE field model for data generation.
    """
    
    def __init__(self, field_map_path: Optional[str] = None, polarity: int = 1):
        """
        Initialize interpolated field from field map file.
        
        Args:
            field_map_path: Path to field map (default: twodip.rtf)
            polarity: +1 for MagUp, -1 for MagDown (flips By and Bx signs)
        """
        self.polarity = polarity
        path = Path(field_map_path) if field_map_path else _DEFAULT_FIELD_MAP
        
        if not path.exists():
            raise FileNotFoundError(
                f"Field map not found: {path}\n"
                f"Please ensure twodip.rtf is in the field_maps directory."
            )
        
        self._load_field_map(path)
    
    def _load_field_map(self, path: Path) -> None:
        """Load field map and prepare interpolation grid."""
        print(f"Loading field map from {path}...")
        data = np.loadtxt(path)
        
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        Bx, By, Bz = data[:, 3], data[:, 4], data[:, 5]
        
        # Get unique coordinates
        self.x_grid = np.sort(np.unique(x))
        self.y_grid = np.sort(np.unique(y))
        self.z_grid = np.sort(np.unique(z))
        
        nx, ny, nz = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        print(f"  Grid size: {nx} × {ny} × {nz} = {nx*ny*nz} points")
        print(f"  x range: [{self.x_grid[0]}, {self.x_grid[-1]}] mm")
        print(f"  y range: [{self.y_grid[0]}, {self.y_grid[-1]}] mm")
        print(f"  z range: [{self.z_grid[0]}, {self.z_grid[-1]}] mm")
        
        # Grid spacing
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.dz = self.z_grid[1] - self.z_grid[0]
        print(f"  Grid spacing: dx={self.dx}mm, dy={self.dy}mm, dz={self.dz}mm")
        
        # File is ordered: y varies fastest, then x, then z
        # So reshape order is (nz, nx, ny) -> then transpose to (nx, ny, nz) for RegularGridInterpolator
        self.Bx_grid = Bx.reshape(nz, nx, ny).transpose(1, 2, 0)  # [nx, ny, nz]
        self.By_grid = By.reshape(nz, nx, ny).transpose(1, 2, 0)
        self.Bz_grid = Bz.reshape(nz, nx, ny).transpose(1, 2, 0)
        
        # Verify the loading by checking a known point
        ix0 = np.searchsorted(self.x_grid, 0)
        iy0 = np.searchsorted(self.y_grid, 0)
        iz5000 = np.searchsorted(self.z_grid, 5000)
        By_center = self.By_grid[ix0, iy0, iz5000]
        print(f"  By at (0, 0, {self.z_grid[iz5000]}): {By_center:.4f} T")
        print(f"  Peak |By|: {np.max(np.abs(self.By_grid)):.4f} T")
    
    def __call__(self, x: Union[float, np.ndarray], 
                 y: Union[float, np.ndarray], 
                 z: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get field at position (x, y, z) via trilinear interpolation.
        
        Args:
            x, y, z: Position(s) in mm
            
        Returns:
            (Bx, By, Bz) in Tesla
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolators (cached after first call via __call__)
        if not hasattr(self, '_interp_Bx'):
            self._interp_Bx = RegularGridInterpolator(
                (self.x_grid, self.y_grid, self.z_grid), self.Bx_grid,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._interp_By = RegularGridInterpolator(
                (self.x_grid, self.y_grid, self.z_grid), self.By_grid,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._interp_Bz = RegularGridInterpolator(
                (self.x_grid, self.y_grid, self.z_grid), self.Bz_grid,
                method='linear', bounds_error=False, fill_value=0.0
            )
        
        # Handle scalars
        scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        
        # Stack coordinates for interpolation
        points = np.stack([x, y, z], axis=-1)
        
        Bx = self._interp_Bx(points) * self.polarity
        By = self._interp_By(points) * self.polarity
        Bz = self._interp_Bz(points)  # Bz typically doesn't flip with polarity
        
        if scalar_input:
            return float(Bx[0]), float(By[0]), float(Bz[0])
        return Bx, By, Bz
    
    def interpolation_error_bound(self) -> float:
        """Estimate upper bound on interpolation error (O(h²) for trilinear)."""
        h_max = max(self.dx, self.dy, self.dz)
        # Conservative estimate: max second derivative ~10⁻⁶ T/mm²
        return 0.5 * h_max**2 * 1e-6


# =============================================================================
# PyTorch Field Models (for PINN training)
# =============================================================================

if TORCH_AVAILABLE:
    class GaussianFieldTorch(nn.Module):
        """
        Gaussian approximation of LHCb magnetic field (PyTorch version, differentiable).
        """
        
        def __init__(self, params: Optional[GaussianFieldParams] = None, polarity: int = 1):
            super().__init__()
            p = params or GaussianFieldParams()
            self.register_buffer('B0', torch.tensor(polarity * p.B0, dtype=torch.float32))
            self.register_buffer('z_center', torch.tensor(p.z_center, dtype=torch.float32))
            self.register_buffer('z_width', torch.tensor(p.z_width, dtype=torch.float32))
        
        def forward(self, x: torch.Tensor, y: torch.Tensor, 
                    z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get field at (x, y, z). Only z-dependent for Gaussian model."""
            z_rel = (z - self.z_center) / self.z_width
            By = self.B0 * torch.exp(-0.5 * z_rel**2)
            Bx = torch.zeros_like(By)
            Bz = torch.zeros_like(By)
            return Bx, By, Bz
    
    
    class InterpolatedFieldTorch(nn.Module):
        """
        Trilinear interpolation of real LHCb field map (PyTorch version, differentiable).
        
        Uses torch.nn.functional.grid_sample for GPU-accelerated interpolation.
        """
        
        def __init__(self, field_map_path: Optional[str] = None, 
                     polarity: int = 1, device: str = 'cpu'):
            super().__init__()
            self.polarity = polarity
            path = Path(field_map_path) if field_map_path else _DEFAULT_FIELD_MAP
            
            if not path.exists():
                raise FileNotFoundError(f"Field map not found: {path}")
            
            self._load_field_map(path, device)
        
        def _load_field_map(self, path: Path, device: str) -> None:
            """Load field map for GPU interpolation."""
            data = np.loadtxt(path)
            
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            Bx, By, Bz = data[:, 3], data[:, 4], data[:, 5]
            
            x_unique = np.sort(np.unique(x))
            y_unique = np.sort(np.unique(y))
            z_unique = np.sort(np.unique(z))
            
            nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
            
            # Store grid bounds
            self.register_buffer('x_min', torch.tensor(x_unique[0], dtype=torch.float32))
            self.register_buffer('x_max', torch.tensor(x_unique[-1], dtype=torch.float32))
            self.register_buffer('y_min', torch.tensor(y_unique[0], dtype=torch.float32))
            self.register_buffer('y_max', torch.tensor(y_unique[-1], dtype=torch.float32))
            self.register_buffer('z_min', torch.tensor(z_unique[0], dtype=torch.float32))
            self.register_buffer('z_max', torch.tensor(z_unique[-1], dtype=torch.float32))
            
            # File is ordered: y varies fastest, then x, then z
            # Reshape to [nz, nx, ny] then transpose to [nx, ny, nz]
            Bx_grid = Bx.reshape(nz, nx, ny).transpose(1, 2, 0)  # [nx, ny, nz]
            By_grid = By.reshape(nz, nx, ny).transpose(1, 2, 0)
            Bz_grid = Bz.reshape(nz, nx, ny).transpose(1, 2, 0)
            
            # For grid_sample: need [1, C, D, H, W] where D=z, H=y, W=x
            # Stack as [3, nx, ny, nz] then permute to [3, nz, ny, nx]
            B_grid = np.stack([Bx_grid, By_grid, Bz_grid], axis=0)  # [3, nx, ny, nz]
            B_grid = np.transpose(B_grid, (0, 3, 2, 1))  # [3, nz, ny, nx]
            B_grid = B_grid[np.newaxis, ...]  # [1, 3, nz, ny, nx]
            
            self.register_buffer('B_grid', torch.tensor(B_grid, dtype=torch.float32))
            self.grid_shape = (nx, ny, nz)
        
        def forward(self, x: torch.Tensor, y: torch.Tensor, 
                    z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get field at (x, y, z) via trilinear interpolation."""
            # Normalize to [-1, 1]
            x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
            y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
            z_norm = 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0
            
            # Clamp to valid range
            x_norm = torch.clamp(x_norm, -1.0, 1.0)
            y_norm = torch.clamp(y_norm, -1.0, 1.0)
            z_norm = torch.clamp(z_norm, -1.0, 1.0)
            
            # Store original shape and number of points
            original_shape = x.shape
            n_points = x.numel()
            
            # Prepare grid for grid_sample: [N, D, H, W, 3]
            grid = torch.stack([x_norm.flatten(), y_norm.flatten(), z_norm.flatten()], dim=-1)
            grid = grid.view(1, 1, 1, n_points, 3)
            
            # Interpolate: output is [1, 3, 1, 1, n_points]
            B_interp = torch.nn.functional.grid_sample(
                self.B_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            
            # Reshape output: [1, 3, 1, 1, n_points] -> [3, n_points] -> [n_points, 3]
            B_interp = B_interp.view(3, n_points).T  # [n_points, 3]
            
            Bx = B_interp[:, 0].view(original_shape) * self.polarity
            By = B_interp[:, 1].view(original_shape) * self.polarity
            Bz = B_interp[:, 2].view(original_shape)
            
            return Bx, By, Bz


# =============================================================================
# Factory Functions
# =============================================================================

def get_field_numpy(use_interpolated: bool = True, 
                    field_map_path: Optional[str] = None,
                    polarity: int = 1) -> Union[InterpolatedFieldNumpy, GaussianFieldNumpy]:
    """
    Get a NumPy magnetic field model (for data generation with RK4).
    
    Args:
        use_interpolated: If True, use real field map. If False, use Gaussian approximation.
        field_map_path: Path to field map file (default: twodip.rtf)
        polarity: +1 for MagUp, -1 for MagDown
        
    Returns:
        Field model with __call__(x, y, z) -> (Bx, By, Bz)
    """
    if use_interpolated:
        path = field_map_path or str(_DEFAULT_FIELD_MAP)
        if Path(path).exists():
            return InterpolatedFieldNumpy(path, polarity=polarity)
        else:
            warnings.warn(
                f"Field map not found at {path}. Falling back to Gaussian approximation. "
                "Data generated with Gaussian will have ~1.3% field error!"
            )
            return GaussianFieldNumpy(polarity=polarity)
    else:
        return GaussianFieldNumpy(polarity=polarity)


def get_field_torch(use_interpolated: bool = True,
                    field_map_path: Optional[str] = None,
                    polarity: int = 1,
                    device: str = 'cpu') -> Union['InterpolatedFieldTorch', 'GaussianFieldTorch']:
    """
    Get a PyTorch magnetic field model (for PINN training).
    
    Args:
        use_interpolated: If True, use real field map. If False, use Gaussian approximation.
        field_map_path: Path to field map file (default: twodip.rtf)
        polarity: +1 for MagUp, -1 for MagDown
        device: 'cpu' or 'cuda'
        
    Returns:
        Field model (nn.Module) with forward(x, y, z) -> (Bx, By, Bz)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for torch field models")
    
    if use_interpolated:
        path = field_map_path or str(_DEFAULT_FIELD_MAP)
        if Path(path).exists():
            return InterpolatedFieldTorch(path, polarity=polarity, device=device)
        else:
            warnings.warn(
                f"Field map not found at {path}. Falling back to Gaussian approximation."
            )
            return GaussianFieldTorch(polarity=polarity)
    else:
        return GaussianFieldTorch(polarity=polarity)


# Convenience aliases
get_field = get_field_numpy  # Default to numpy for data generation


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Magnetic Field Module Test")
    print("=" * 60)
    
    # Test NumPy fields
    print("\n--- NumPy Fields ---")
    
    gauss_np = GaussianFieldNumpy()
    Bx, By, Bz = gauss_np(0.0, 0.0, 5007.0)
    print(f"Gaussian at (0, 0, 5007): Bx={Bx:.4f}, By={By:.4f}, Bz={Bz:.4f} T")
    
    try:
        interp_np = InterpolatedFieldNumpy()
        Bx, By, Bz = interp_np(0.0, 0.0, 5007.0)
        print(f"Interpolated at (0, 0, 5007): Bx={Bx:.4f}, By={By:.4f}, Bz={Bz:.4f} T")
        
        # Compare fields along z-axis
        z_test = np.linspace(2000, 10000, 100)
        By_gauss = np.array([gauss_np(0, 0, zi)[1] for zi in z_test])
        By_interp = np.array([interp_np(0, 0, zi)[1] for zi in z_test])
        
        rms_error = np.sqrt(np.mean((By_gauss - By_interp)**2))
        max_error = np.max(np.abs(By_gauss - By_interp))
        print(f"\nGaussian vs Interpolated (at x=0, y=0):")
        print(f"  RMS error: {rms_error:.4f} T")
        print(f"  Max error: {max_error:.4f} T")
        print(f"  Relative error: {100*rms_error/np.max(np.abs(By_interp)):.2f}%")
        
    except FileNotFoundError as e:
        print(f"Field map not found: {e}")
    
    # Test PyTorch fields
    if TORCH_AVAILABLE:
        print("\n--- PyTorch Fields ---")
        
        gauss_torch = GaussianFieldTorch()
        z = torch.tensor([5007.0])
        Bx, By, Bz = gauss_torch(torch.zeros(1), torch.zeros(1), z)
        print(f"Gaussian at (0, 0, 5007): By={By.item():.4f} T")
        
        try:
            interp_torch = InterpolatedFieldTorch()
            Bx, By, Bz = interp_torch(torch.zeros(1), torch.zeros(1), z)
            print(f"Interpolated at (0, 0, 5007): By={By.item():.4f} T")
        except FileNotFoundError:
            print("Field map not found for PyTorch model")
    
    print("\n" + "=" * 60)
    print("Test complete!")
