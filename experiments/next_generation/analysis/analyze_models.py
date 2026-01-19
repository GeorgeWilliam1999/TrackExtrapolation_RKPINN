#!/usr/bin/env python3
"""
Comprehensive Model Analysis Suite for Track Extrapolator Neural Networks

This module provides advanced analysis tools for evaluating trained models,
with special focus on:
1. Trajectory visualization and comparison
2. Physics constraint validation (for PINNs)
3. Momentum-dependent performance analysis
4. Statistical error analysis with proper uncertainty quantification
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Add models directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from architectures import create_model


class TrackExtrapolatorAnalyzer:
    """
    Comprehensive analyzer for track extrapolator neural networks.
    
    Provides tools for:
    - Loading and comparing multiple models
    - Trajectory visualization
    - Physics constraint validation
    - Statistical performance analysis
    """
    
    def __init__(self, models_dir: Path, data_path: Path):
        """
        Initialize the analyzer.
        
        Args:
            models_dir: Directory containing trained models
            data_path: Path to the test data (training_50M.npz)
        """
        self.models_dir = Path(models_dir)
        self.data_path = Path(data_path)
        self.models: Dict[str, Dict] = {}
        self.data: Dict[str, np.ndarray] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Physical constants
        self.dz = 2300.0  # mm - extrapolation distance
        self.c_light = 299.792458  # mm/ns
        
    def load_data(self, n_samples: Optional[int] = None, seed: int = 42):
        """Load test data for analysis."""
        print(f"Loading data from {self.data_path}...")
        data = np.load(self.data_path)
        
        X = data['X'].astype(np.float32)  # [N, 5] - [x, y, tx, ty, qop]
        Y = data['Y'].astype(np.float32)  # [N, 5] - [x, y, tx, ty, qop_out]
        P = data['P'].astype(np.float32)
        
        # Add dz column to match training preprocessing
        # Training adds dz = 2300.0 as 6th column
        dz_col = np.full((len(X), 1), self.dz, dtype=np.float32)
        X = np.hstack([X, dz_col])  # [N, 6] - [x, y, tx, ty, qop, dz]
        
        # Extract only position and slopes from Y (not qop_out)
        Y = Y[:, :4]  # [N, 4] - [x, y, tx, ty]
        
        # Use last portion as test set (matching training split)
        n_total = len(X)
        n_test = n_total // 10  # 10% test
        
        X_test = X[-n_test:]
        Y_test = Y[-n_test:]
        P_test = P[-n_test:]
        
        if n_samples is not None and n_samples < len(X_test):
            np.random.seed(seed)
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_test = X_test[indices]
            Y_test = Y_test[indices]
            P_test = P_test[indices]
        
        self.data = {
            'X': X_test,
            'Y': Y_test,
            'P': P_test,
            'dz': self.dz
        }
        
        # Extract components for convenience
        # Input: [x, y, tx, ty, qop, dz_norm]
        # Output: [x_out, y_out, tx_out, ty_out]
        self.data['x_in'] = X_test[:, 0]
        self.data['y_in'] = X_test[:, 1]
        self.data['tx_in'] = X_test[:, 2]
        self.data['ty_in'] = X_test[:, 3]
        self.data['qop'] = X_test[:, 4]
        self.data['charge'] = np.sign(X_test[:, 4])
        
        self.data['x_out_true'] = Y_test[:, 0]
        self.data['y_out_true'] = Y_test[:, 1]
        self.data['tx_out_true'] = Y_test[:, 2]
        self.data['ty_out_true'] = Y_test[:, 3]
        
        print(f"  Loaded {len(X_test):,} test samples")
        print(f"  Momentum range: {self.data['P'].min():.1f} - {self.data['P'].max():.1f} GeV")
        print(f"  Charge distribution: {(self.data['charge'] > 0).sum():,} positive, {(self.data['charge'] < 0).sum():,} negative")
        
    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model directory
            
        Returns:
            True if model loaded successfully
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            print(f"  Model directory not found: {model_dir}")
            return False
            
        config_path = model_dir / 'config.json'
        model_path = model_dir / 'best_model.pt'
        results_path = model_dir / 'results.json'
        
        if not config_path.exists() or not model_path.exists():
            print(f"  Missing config or model file for {model_name}")
            return False
            
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Create model with correct signature per model type
            model_type = config.get('model_type', 'mlp')
            
            # Base kwargs accepted by all models
            model_kwargs = {
                'input_dim': 6,
                'output_dim': 4,
                'hidden_dims': config.get('hidden_dims', [256, 256, 128]),
                'activation': config.get('activation', 'silu'),
            }
            
            # Add model-specific parameters
            if model_type == 'mlp':
                model_kwargs['dropout'] = config.get('dropout', 0.0)
            elif model_type == 'pinn':
                model_kwargs['lambda_pde'] = config.get('lambda_pde', 1e-3)
            elif model_type == 'rk_pinn':
                model_kwargs['n_collocation'] = config.get('n_collocation', 10)
                model_kwargs['lambda_pde'] = config.get('lambda_pde', 1e-3)
            
            model = create_model(model_type, **model_kwargs)
            
            # Load weights (weights_only=False needed for registered buffers)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            # Load results if available
            results = {}
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
            
            self.models[model_name] = {
                'model': model,
                'config': config,
                'results': results,
                'predictions': None
            }
            
            print(f"  Loaded {model_name}: {config['parameters']:,} parameters")
            return True
            
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            return False
    
    def load_all_models(self, pattern: str = "*_v1"):
        """Load all models matching a pattern."""
        print("Loading models...")
        model_dirs = sorted(self.models_dir.glob(pattern))
        
        for model_dir in model_dirs:
            if model_dir.is_dir() and (model_dir / 'config.json').exists():
                self.load_model(model_dir.name)
        
        print(f"  Loaded {len(self.models)} models total")
    
    def predict(self, model_name: str) -> np.ndarray:
        """
        Run model inference on test data.
        
        Returns:
            Predictions array [n_samples, 4]
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        if self.models[model_name]['predictions'] is not None:
            return self.models[model_name]['predictions']
        
        model = self.models[model_name]['model']
        X = torch.tensor(self.data['X'], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            predictions = model(X).cpu().numpy()
        
        self.models[model_name]['predictions'] = predictions
        return predictions
    
    def compute_errors(self, model_name: str) -> Dict[str, np.ndarray]:
        """
        Compute detailed error metrics for a model.
        
        Returns:
            Dictionary with various error metrics
        """
        pred = self.predict(model_name)
        true = self.data['Y']
        
        # Raw errors
        dx = pred[:, 0] - true[:, 0]
        dy = pred[:, 1] - true[:, 1]
        dtx = pred[:, 2] - true[:, 2]
        dty = pred[:, 3] - true[:, 3]
        
        # Position error magnitude
        pos_err = np.sqrt(dx**2 + dy**2)
        
        # Slope error magnitude
        slope_err = np.sqrt(dtx**2 + dty**2)
        
        return {
            'dx': dx,
            'dy': dy,
            'dtx': dtx,
            'dty': dty,
            'pos_err': pos_err,
            'slope_err': slope_err,
            'x_pred': pred[:, 0],
            'y_pred': pred[:, 1],
            'tx_pred': pred[:, 2],
            'ty_pred': pred[:, 3]
        }
    
    # ==================== TRAJECTORY VISUALIZATION ====================
    
    def plot_trajectory_comparison(self, model_names: List[str], 
                                   n_tracks: int = 5,
                                   momentum_range: Optional[Tuple[float, float]] = None,
                                   charge: Optional[int] = None,
                                   save_path: Optional[Path] = None):
        """
        Plot predicted vs true trajectories for multiple models.
        
        Args:
            model_names: List of model names to compare
            n_tracks: Number of tracks to plot
            momentum_range: Optional (p_min, p_max) filter
            charge: Optional charge filter (+1 or -1)
            save_path: Optional path to save figure
        """
        # Select tracks based on criteria
        mask = np.ones(len(self.data['P']), dtype=bool)
        if momentum_range is not None:
            mask &= (self.data['P'] >= momentum_range[0]) & (self.data['P'] <= momentum_range[1])
        if charge is not None:
            mask &= self.data['charge'] == charge
        
        valid_indices = np.where(mask)[0]
        if len(valid_indices) < n_tracks:
            print(f"Warning: Only {len(valid_indices)} tracks match criteria")
            n_tracks = len(valid_indices)
        
        np.random.seed(42)
        selected = np.random.choice(valid_indices, n_tracks, replace=False)
        
        fig, axes = plt.subplots(2, n_tracks, figsize=(4*n_tracks, 8))
        if n_tracks == 1:
            axes = axes.reshape(2, 1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names) + 1))
        
        for i, idx in enumerate(selected):
            # Get track parameters
            x_in = self.data['x_in'][idx]
            y_in = self.data['y_in'][idx]
            tx_in = self.data['tx_in'][idx]
            ty_in = self.data['ty_in'][idx]
            p = self.data['P'][idx]
            q = self.data['charge'][idx]
            
            # True endpoint
            x_true = self.data['x_out_true'][idx]
            y_true = self.data['y_out_true'][idx]
            tx_true = self.data['tx_out_true'][idx]
            ty_true = self.data['ty_out_true'][idx]
            
            # Z coordinates
            z_in, z_out = 0, self.dz
            
            # Plot X-Z projection
            ax_xz = axes[0, i]
            
            # Starting point
            ax_xz.scatter([z_in], [x_in], c='black', s=100, marker='o', zorder=10, label='Start')
            
            # True trajectory (simplified as line for visualization)
            ax_xz.plot([z_in, z_out], [x_in, x_true], 'k-', lw=2, label='True', alpha=0.7)
            ax_xz.scatter([z_out], [x_true], c='black', s=100, marker='x', zorder=10)
            
            # Model predictions
            for j, model_name in enumerate(model_names):
                pred = self.predict(model_name)
                x_pred = pred[idx, 0]
                
                short_name = model_name.replace('_v1', '').replace('_', ' ')
                ax_xz.plot([z_in, z_out], [x_in, x_pred], '--', 
                          color=colors[j+1], lw=1.5, label=short_name, alpha=0.8)
                ax_xz.scatter([z_out], [x_pred], c=colors[j+1], s=50, marker='s', zorder=9)
            
            ax_xz.set_xlabel('z [mm]')
            ax_xz.set_ylabel('x [mm]')
            ax_xz.set_title(f'Track {i+1}: P={p:.1f} GeV, q={int(q):+d}')
            if i == 0:
                ax_xz.legend(fontsize=8, loc='best')
            ax_xz.grid(True, alpha=0.3)
            
            # Plot Y-Z projection
            ax_yz = axes[1, i]
            
            ax_yz.scatter([z_in], [y_in], c='black', s=100, marker='o', zorder=10)
            ax_yz.plot([z_in, z_out], [y_in, y_true], 'k-', lw=2, alpha=0.7)
            ax_yz.scatter([z_out], [y_true], c='black', s=100, marker='x', zorder=10)
            
            for j, model_name in enumerate(model_names):
                pred = self.predict(model_name)
                y_pred = pred[idx, 1]
                ax_yz.plot([z_in, z_out], [y_in, y_pred], '--', 
                          color=colors[j+1], lw=1.5, alpha=0.8)
                ax_yz.scatter([z_out], [y_pred], c=colors[j+1], s=50, marker='s', zorder=9)
            
            ax_yz.set_xlabel('z [mm]')
            ax_yz.set_ylabel('y [mm]')
            ax_yz.grid(True, alpha=0.3)
        
        fig.suptitle('Trajectory Comparison: X-Z (top) and Y-Z (bottom) Projections', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trajectory plot to {save_path}")
        
        return fig
    
    def plot_trajectory_residuals(self, model_names: List[str],
                                  n_tracks: int = 100,
                                  save_path: Optional[Path] = None):
        """
        Plot trajectory residuals (errors) for multiple models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for j, model_name in enumerate(model_names):
            errors = self.compute_errors(model_name)
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            
            # X residual distribution
            axes[0, 0].hist(errors['dx'], bins=100, alpha=0.5, 
                           label=f"{short_name} (σ={np.std(errors['dx']):.3f}mm)",
                           color=colors[j], density=True)
            
            # Y residual distribution
            axes[0, 1].hist(errors['dy'], bins=100, alpha=0.5,
                           label=f"{short_name} (σ={np.std(errors['dy']):.3f}mm)",
                           color=colors[j], density=True)
            
            # tx residual distribution
            axes[1, 0].hist(errors['dtx'] * 1000, bins=100, alpha=0.5,
                           label=f"{short_name} (σ={np.std(errors['dtx'])*1000:.3f}mrad)",
                           color=colors[j], density=True)
            
            # ty residual distribution
            axes[1, 1].hist(errors['dty'] * 1000, bins=100, alpha=0.5,
                           label=f"{short_name} (σ={np.std(errors['dty'])*1000:.3f}mrad)",
                           color=colors[j], density=True)
        
        axes[0, 0].set_xlabel('Δx [mm]')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('X Position Residual')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Δy [mm]')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Y Position Residual')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Δtx [mrad]')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('X Slope Residual')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Δty [mrad]')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Y Slope Residual')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Residual Distributions', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    # ==================== PHYSICS CONSTRAINT ANALYSIS ====================
    
    def analyze_ty_conservation(self, model_names: List[str],
                                save_path: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Analyze ty (y-slope) conservation.
        
        In regions without magnetic field, ty should be conserved.
        This tests whether PINN models better preserve this physics constraint.
        
        Returns:
            Dictionary with ty conservation metrics for each model
        """
        results = {}
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        ty_in = self.data['ty_in']
        ty_true = self.data['ty_out_true']
        
        # True ty change (should be ~0 if no field in y)
        dty_true = ty_true - ty_in
        
        for j, model_name in enumerate(model_names):
            errors = self.compute_errors(model_name)
            ty_pred = errors['ty_pred']
            dty_pred = ty_pred - ty_in
            
            # Compute ty conservation violation
            ty_violation = np.abs(dty_pred - dty_true)
            
            results[model_name] = {
                'mean_dty': float(np.mean(dty_pred)),
                'std_dty': float(np.std(dty_pred)),
                'mean_violation': float(np.mean(ty_violation)),
                'max_violation': float(np.max(ty_violation)),
                'rmse_dty': float(np.sqrt(np.mean((dty_pred - dty_true)**2)))
            }
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            
            # Plot dty distribution
            axes[0].hist(dty_pred * 1000, bins=100, alpha=0.5,
                        label=f"{short_name} (μ={np.mean(dty_pred)*1000:.3f})",
                        color=colors[j], density=True)
            
            # Plot ty_pred vs ty_in
            sample_idx = np.random.choice(len(ty_in), min(5000, len(ty_in)), replace=False)
            axes[1].scatter(ty_in[sample_idx], ty_pred[sample_idx], 
                           alpha=0.1, s=1, color=colors[j], label=short_name)
        
        # True distribution
        axes[0].hist(dty_true * 1000, bins=100, alpha=0.7,
                    label=f"True (μ={np.mean(dty_true)*1000:.3f})",
                    color='black', density=True, histtype='step', lw=2)
        
        # Perfect conservation line
        ty_range = [ty_in.min(), ty_in.max()]
        axes[1].plot(ty_range, ty_range, 'k--', lw=2, label='Perfect conservation')
        
        axes[0].set_xlabel('Δty (ty_out - ty_in) [mrad]')
        axes[0].set_ylabel('Density')
        axes[0].set_title('ty Change Distribution')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('ty_in')
        axes[1].set_ylabel('ty_pred')
        axes[1].set_title('ty Conservation Check')
        axes[1].legend(fontsize=8, markerscale=5)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')
        
        fig.suptitle('Physics Constraint: ty Conservation Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return results
    
    def analyze_charge_consistency(self, model_names: List[str],
                                   save_path: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Analyze charge consistency in model predictions.
        
        For opposite charges, the bending direction should be opposite.
        This tests whether the model correctly handles charge-dependent behavior.
        
        Returns:
            Dictionary with charge consistency metrics
        """
        results = {}
        
        # Separate positive and negative charges
        pos_mask = self.data['charge'] > 0
        neg_mask = self.data['charge'] < 0
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for j, model_name in enumerate(model_names):
            errors = self.compute_errors(model_name)
            
            # Compute bending (change in tx)
            tx_in = self.data['tx_in']
            tx_pred = errors['tx_pred']
            bending = tx_pred - tx_in
            
            pos_bending = bending[pos_mask]
            neg_bending = bending[neg_mask]
            
            # For correct physics: pos_bending and neg_bending should have opposite signs
            pos_mean = np.mean(pos_bending)
            neg_mean = np.mean(neg_bending)
            
            # Charge asymmetry metric (should be ~2 for perfect antisymmetry)
            if pos_mean != 0:
                asymmetry = -neg_mean / pos_mean
            else:
                asymmetry = 0
            
            results[model_name] = {
                'pos_mean_bending': float(pos_mean),
                'neg_mean_bending': float(neg_mean),
                'asymmetry': float(asymmetry),
                'pos_std': float(np.std(pos_bending)),
                'neg_std': float(np.std(neg_bending)),
                'pos_err_mean': float(np.mean(errors['pos_err'][pos_mask])),
                'neg_err_mean': float(np.mean(errors['pos_err'][neg_mask]))
            }
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            
            # Bending distribution for positive charges
            axes[0, 0].hist(pos_bending * 1000, bins=100, alpha=0.5,
                           label=short_name, color=colors[j], density=True)
            
            # Bending distribution for negative charges  
            axes[0, 1].hist(neg_bending * 1000, bins=100, alpha=0.5,
                           label=short_name, color=colors[j], density=True)
            
            # Error by charge
            axes[1, 0].scatter([j], [np.mean(errors['pos_err'][pos_mask])], 
                              c=[colors[j]], s=100, marker='o', label=f'{short_name} q+')
            axes[1, 0].scatter([j], [np.mean(errors['pos_err'][neg_mask])], 
                              c=[colors[j]], s=100, marker='x')
        
        # True bending distributions
        true_tx_change = self.data['tx_out_true'] - self.data['tx_in']
        axes[0, 0].hist(true_tx_change[pos_mask] * 1000, bins=100, alpha=0.7,
                       label='True', color='black', density=True, histtype='step', lw=2)
        axes[0, 1].hist(true_tx_change[neg_mask] * 1000, bins=100, alpha=0.7,
                       label='True', color='black', density=True, histtype='step', lw=2)
        
        axes[0, 0].set_xlabel('Δtx (bending) [mrad]')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Bending: Positive Charges (q > 0)')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Δtx (bending) [mrad]')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Bending: Negative Charges (q < 0)')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Charge asymmetry comparison
        names = [m.replace('_v1', '').replace('_', '\n') for m in model_names]
        asymmetries = [results[m]['asymmetry'] for m in model_names]
        axes[1, 1].bar(range(len(model_names)), asymmetries, color=colors)
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Perfect asymmetry')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(names, fontsize=8, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Charge Asymmetry (-neg/pos)')
        axes[1, 1].set_title('Charge Asymmetry Metric')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Clean up position error plot
        axes[1, 0].set_ylabel('Position Error [mm]')
        axes[1, 0].set_title('Error by Charge Sign (o=pos, x=neg)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(names, fontsize=8, rotation=45, ha='right')
        
        fig.suptitle('Physics Constraint: Charge Consistency Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return results
    
    # ==================== MOMENTUM-DEPENDENT ANALYSIS ====================
    
    def analyze_momentum_dependence(self, model_names: List[str],
                                    n_bins: int = 20,
                                    save_path: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Analyze model performance as a function of momentum.
        
        Low momentum tracks bend more in the magnetic field and are harder to extrapolate.
        This analysis reveals momentum-dependent systematic effects.
        
        Returns:
            Dictionary with momentum-binned error statistics
        """
        results = {}
        
        P = self.data['P']
        p_bins = np.logspace(np.log10(P.min()), np.log10(P.max()), n_bins + 1)
        p_centers = np.sqrt(p_bins[:-1] * p_bins[1:])  # Geometric mean
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for j, model_name in enumerate(model_names):
            errors = self.compute_errors(model_name)
            
            pos_err_means = []
            pos_err_stds = []
            slope_err_means = []
            slope_err_stds = []
            
            for i in range(len(p_bins) - 1):
                mask = (P >= p_bins[i]) & (P < p_bins[i+1])
                
                if mask.sum() > 0:
                    pos_err_means.append(np.mean(errors['pos_err'][mask]))
                    pos_err_stds.append(np.std(errors['pos_err'][mask]))
                    slope_err_means.append(np.mean(errors['slope_err'][mask]))
                    slope_err_stds.append(np.std(errors['slope_err'][mask]))
                else:
                    pos_err_means.append(np.nan)
                    pos_err_stds.append(np.nan)
                    slope_err_means.append(np.nan)
                    slope_err_stds.append(np.nan)
            
            results[model_name] = {
                'p_centers': p_centers.tolist(),
                'pos_err_means': pos_err_means,
                'pos_err_stds': pos_err_stds,
                'slope_err_means': slope_err_means,
                'slope_err_stds': slope_err_stds
            }
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            
            # Position error vs momentum
            axes[0, 0].errorbar(p_centers, pos_err_means, yerr=pos_err_stds,
                               label=short_name, color=colors[j], marker='o', 
                               markersize=4, capsize=2, alpha=0.7)
            
            # Slope error vs momentum
            axes[0, 1].errorbar(p_centers, [s*1000 for s in slope_err_means],
                               yerr=[s*1000 for s in slope_err_stds],
                               label=short_name, color=colors[j], marker='o',
                               markersize=4, capsize=2, alpha=0.7)
            
            # Error distribution in low-p regime
            low_p_mask = P < 2.0
            if low_p_mask.sum() > 0:
                axes[1, 0].hist(errors['pos_err'][low_p_mask], bins=50, alpha=0.5,
                               label=f'{short_name} (μ={np.mean(errors["pos_err"][low_p_mask]):.2f}mm)',
                               color=colors[j], density=True)
            
            # Error distribution in high-p regime
            high_p_mask = P > 10.0
            if high_p_mask.sum() > 0:
                axes[1, 1].hist(errors['pos_err'][high_p_mask], bins=50, alpha=0.5,
                               label=f'{short_name} (μ={np.mean(errors["pos_err"][high_p_mask]):.2f}mm)',
                               color=colors[j], density=True)
        
        axes[0, 0].set_xlabel('Momentum [GeV]')
        axes[0, 0].set_ylabel('Position Error [mm]')
        axes[0, 0].set_title('Position Error vs Momentum')
        axes[0, 0].set_xscale('log')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Momentum [GeV]')
        axes[0, 1].set_ylabel('Slope Error [mrad]')
        axes[0, 1].set_title('Slope Error vs Momentum')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Position Error [mm]')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Error Distribution: Low Momentum (P < 2 GeV)')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Position Error [mm]')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution: High Momentum (P > 10 GeV)')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Momentum-Dependent Performance Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return results
    
    # ==================== STATISTICAL ANALYSIS ====================
    
    def compute_statistical_summary(self, model_names: List[str]) -> Dict[str, Dict]:
        """
        Compute comprehensive statistical summary for each model.
        
        Returns:
            Dictionary with detailed statistics for each model
        """
        results = {}
        
        for model_name in model_names:
            errors = self.compute_errors(model_name)
            config = self.models[model_name]['config']
            model = self.models[model_name]['model']
            
            # Compute percentiles
            pos_percentiles = np.percentile(errors['pos_err'], [50, 68, 90, 95, 99])
            slope_percentiles = np.percentile(errors['slope_err'] * 1000, [50, 68, 90, 95, 99])
            
            results[model_name] = {
                'model_type': config.get('model_type', 'unknown'),
                'parameters': model.count_parameters(),
                'hidden_dims': config.get('hidden_dims', []),
                
                # Position errors
                'pos_mean': float(np.mean(errors['pos_err'])),
                'pos_std': float(np.std(errors['pos_err'])),
                'pos_median': float(pos_percentiles[0]),
                'pos_68': float(pos_percentiles[1]),
                'pos_90': float(pos_percentiles[2]),
                'pos_95': float(pos_percentiles[3]),
                'pos_99': float(pos_percentiles[4]),
                
                # Slope errors (in mrad)
                'slope_mean_mrad': float(np.mean(errors['slope_err']) * 1000),
                'slope_std_mrad': float(np.std(errors['slope_err']) * 1000),
                'slope_median_mrad': float(slope_percentiles[0]),
                'slope_68_mrad': float(slope_percentiles[1]),
                'slope_90_mrad': float(slope_percentiles[2]),
                
                # Individual components
                'dx_mean': float(np.mean(errors['dx'])),
                'dx_std': float(np.std(errors['dx'])),
                'dy_mean': float(np.mean(errors['dy'])),
                'dy_std': float(np.std(errors['dy'])),
                'dtx_mean_mrad': float(np.mean(errors['dtx']) * 1000),
                'dtx_std_mrad': float(np.std(errors['dtx']) * 1000),
                'dty_mean_mrad': float(np.mean(errors['dty']) * 1000),
                'dty_std_mrad': float(np.std(errors['dty']) * 1000),
            }
        
        return results
    
    def compare_pinn_vs_mlp(self, save_path: Optional[Path] = None) -> Dict:
        """
        Specialized comparison between PINN and MLP models.
        
        This analysis specifically examines whether physics-informed
        constraints improve model performance and generalization.
        """
        # Categorize models
        pinn_models = [m for m in self.models if 'pinn' in m.lower()]
        mlp_models = [m for m in self.models if 'mlp' in m.lower() and 'pinn' not in m.lower()]
        resmlp_models = [m for m in self.models if 'resmlp' in m.lower()]
        
        results = {
            'pinn': [],
            'mlp': [],
            'resmlp': []
        }
        
        for model_name in pinn_models:
            errors = self.compute_errors(model_name)
            results['pinn'].append({
                'name': model_name,
                'pos_err': np.mean(errors['pos_err']),
                'slope_err': np.mean(errors['slope_err']) * 1000,
                'params': self.models[model_name]['config']['parameters']
            })
        
        for model_name in mlp_models:
            errors = self.compute_errors(model_name)
            results['mlp'].append({
                'name': model_name,
                'pos_err': np.mean(errors['pos_err']),
                'slope_err': np.mean(errors['slope_err']) * 1000,
                'params': self.models[model_name]['config']['parameters']
            })
        
        for model_name in resmlp_models:
            errors = self.compute_errors(model_name)
            results['resmlp'].append({
                'name': model_name,
                'pos_err': np.mean(errors['pos_err']),
                'slope_err': np.mean(errors['slope_err']) * 1000,
                'params': self.models[model_name]['config']['parameters']
            })
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Position error comparison
        for category, color, marker in [('pinn', 'red', 'o'), ('mlp', 'blue', 's'), ('resmlp', 'green', '^')]:
            if results[category]:
                params = [r['params'] for r in results[category]]
                pos_errs = [r['pos_err'] for r in results[category]]
                axes[0].scatter(params, pos_errs, c=color, marker=marker, 
                               s=100, label=category.upper(), alpha=0.7)
        
        axes[0].set_xlabel('Parameters')
        axes[0].set_ylabel('Mean Position Error [mm]')
        axes[0].set_title('Position Error vs Model Size')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Slope error comparison
        for category, color, marker in [('pinn', 'red', 'o'), ('mlp', 'blue', 's'), ('resmlp', 'green', '^')]:
            if results[category]:
                params = [r['params'] for r in results[category]]
                slope_errs = [r['slope_err'] for r in results[category]]
                axes[1].scatter(params, slope_errs, c=color, marker=marker,
                               s=100, label=category.upper(), alpha=0.7)
        
        axes[1].set_xlabel('Parameters')
        axes[1].set_ylabel('Mean Slope Error [mrad]')
        axes[1].set_title('Slope Error vs Model Size')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Bar chart of best models per category
        categories = ['MLP', 'PINN', 'RK-PINN']
        best_pos = []
        best_names = []
        for cat in ['mlp', 'pinn', 'rk_pinn']:
            if results[cat]:
                best = min(results[cat], key=lambda x: x['pos_err'])
                best_pos.append(best['pos_err'])
                best_names.append(best['name'].replace('_v1', ''))
            else:
                best_pos.append(np.nan)
                best_names.append('')
        
        bars = axes[2].bar(categories, best_pos, color=['red', 'blue', 'green'], alpha=0.7)
        axes[2].set_ylabel('Best Position Error [mm]')
        axes[2].set_title('Best Model per Category')
        axes[2].grid(True, alpha=0.3)
        
        # Add model names as labels
        for bar, name in zip(bars, best_names):
            if not np.isnan(bar.get_height()):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{name}\n{bar.get_height():.3f}mm',
                            ha='center', va='bottom', fontsize=8)
        
        fig.suptitle('PINN vs MLP Comparison', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return results
    
    def generate_comprehensive_report(self, output_dir: Path) -> str:
        """
        Generate a comprehensive analysis report with all visualizations.
        
        Args:
            output_dir: Directory to save all plots and report
            
        Returns:
            Path to the generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Generating Comprehensive Model Analysis Report")
        print("=" * 60)
        
        model_names = list(self.models.keys())
        
        if len(model_names) == 0:
            print("No models loaded. Please load models first.")
            return ""
        
        print(f"\nAnalyzing {len(model_names)} models...")
        
        # Select representative models for detailed plots
        # (to avoid overcrowded plots)
        if len(model_names) > 8:
            # Select best from each category
            stats = self.compute_statistical_summary(model_names)
            sorted_models = sorted(stats.items(), key=lambda x: x[1]['pos_mean'])
            representative = [m[0] for m in sorted_models[:8]]
        else:
            representative = model_names
        
        print(f"  Representative models for plots: {len(representative)}")
        
        # 1. Trajectory visualization
        print("\n1. Generating trajectory comparison...")
        self.plot_trajectory_comparison(
            representative[:4] if len(representative) > 4 else representative,
            n_tracks=4,
            save_path=output_dir / 'trajectories.png'
        )
        plt.close()
        
        # 2. Residual distributions
        print("2. Generating residual distributions...")
        self.plot_trajectory_residuals(
            representative,
            save_path=output_dir / 'residuals.png'
        )
        plt.close()
        
        # 3. ty conservation analysis
        print("3. Analyzing ty conservation...")
        ty_results = self.analyze_ty_conservation(
            representative,
            save_path=output_dir / 'ty_conservation.png'
        )
        plt.close()
        
        # 4. Charge consistency analysis
        print("4. Analyzing charge consistency...")
        charge_results = self.analyze_charge_consistency(
            representative,
            save_path=output_dir / 'charge_consistency.png'
        )
        plt.close()
        
        # 5. Momentum dependence
        print("5. Analyzing momentum dependence...")
        momentum_results = self.analyze_momentum_dependence(
            representative,
            save_path=output_dir / 'momentum_dependence.png'
        )
        plt.close()
        
        # 6. PINN vs MLP comparison
        print("6. Comparing PINN vs MLP...")
        comparison = self.compare_pinn_vs_mlp(
            save_path=output_dir / 'pinn_vs_mlp.png'
        )
        plt.close()
        
        # 7. Statistical summary
        print("7. Computing statistical summary...")
        all_stats = self.compute_statistical_summary(model_names)
        
        # Save statistics to JSON
        with open(output_dir / 'model_statistics.json', 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        # Save analysis results
        analysis_results = {
            'ty_conservation': ty_results,
            'charge_consistency': charge_results,
            'model_comparison': comparison
        }
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Generate summary table
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Model':<30} {'Type':<12} {'Params':>10} {'Pos Err':>10} {'Slope Err':>12}")
        print("-" * 80)
        
        sorted_stats = sorted(all_stats.items(), key=lambda x: x[1]['pos_mean'])
        for name, stats in sorted_stats:
            print(f"{name:<30} {stats['model_type']:<12} {stats['parameters']:>10,} "
                  f"{stats['pos_mean']:>9.4f}mm {stats['slope_mean_mrad']:>10.4f}mrad")
        
        print("=" * 80)
        
        # Best model
        best_name, best_stats = sorted_stats[0]
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   Position Error: {best_stats['pos_mean']:.4f} ± {best_stats['pos_std']:.4f} mm")
        print(f"   Slope Error: {best_stats['slope_mean_mrad']:.4f} ± {best_stats['slope_std_mrad']:.4f} mrad")
        print(f"   Parameters: {best_stats['parameters']:,}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print("=" * 60)
        
        return str(output_dir)


def main():
    """Run analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trained track extrapolator models')
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (training_50M.npz)')
    parser.add_argument('--output', type=str, default='analysis_output',
                       help='Output directory for analysis results')
    parser.add_argument('--n-samples', type=int, default=100000,
                       help='Number of test samples to use')
    parser.add_argument('--pattern', type=str, default='*_v1',
                       help='Pattern to match model directories')
    
    args = parser.parse_args()
    
    analyzer = TrackExtrapolatorAnalyzer(
        models_dir=args.models_dir,
        data_path=args.data
    )
    
    analyzer.load_data(n_samples=args.n_samples)
    analyzer.load_all_models(pattern=args.pattern)
    analyzer.generate_comprehensive_report(Path(args.output))


if __name__ == '__main__':
    main()
