#!/usr/bin/env python3
"""
Advanced Physics Analysis for Track Extrapolator Models

This module provides specialized physics-aware analysis tools that go beyond
standard ML metrics. It focuses on understanding:

1. Magnetic field effects and track bending
2. Energy/momentum conservation
3. Lorentz force consistency
4. Phase space coverage and extrapolation accuracy
5. Systematic vs statistical errors
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from architectures import create_model


class PhysicsAnalyzer:
    """
    Physics-focused analyzer for track extrapolator models.
    
    This analyzer examines whether models learn correct physical behavior
    rather than just minimizing loss on training data.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.data = {}
        
        # LHCb dipole magnet properties (approximate)
        self.B_field = 1.0  # Tesla (average integrated field)
        self.z_magnet_center = 5000  # mm - approximate center of magnet
        self.dz = 2300.0  # mm - extrapolation distance
        
    def load_data(self, data_path: Path, n_samples: int = 100000):
        """Load test data."""
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        
        X = data['X'].astype(np.float32)  # [N, 5] - [x, y, tx, ty, qop]
        Y = data['Y'].astype(np.float32)  # [N, 5] - [x, y, tx, ty, qop_out]
        P = data['P'].astype(np.float32)
        
        # Add dz column to match training preprocessing
        dz_col = np.full((len(X), 1), self.dz, dtype=np.float32)
        X = np.hstack([X, dz_col])  # [N, 6] - [x, y, tx, ty, qop, dz]
        Y = Y[:, :4]  # Extract only [x, y, tx, ty]
        
        # Use test portion
        n_test = len(X) // 10
        X_test = X[-n_test:]
        Y_test = Y[-n_test:]
        P_test = P[-n_test:]
        
        if n_samples < len(X_test):
            np.random.seed(42)
            idx = np.random.choice(len(X_test), n_samples, replace=False)
            X_test, Y_test, P_test = X_test[idx], Y_test[idx], P_test[idx]
        
        self.data = {
            'X': X_test, 'Y': Y_test, 'P': P_test,
            'x_in': X_test[:, 0], 'y_in': X_test[:, 1],
            'tx_in': X_test[:, 2], 'ty_in': X_test[:, 3],
            'qop': X_test[:, 4],
            'x_out': Y_test[:, 0], 'y_out': Y_test[:, 1],
            'tx_out': Y_test[:, 2], 'ty_out': Y_test[:, 3],
            'charge': np.sign(X_test[:, 4]),
            'momentum': P_test
        }
        
        print(f"  Loaded {len(X_test):,} samples")
        
    def load_model(self, model_dir: Path) -> bool:
        """Load a trained model."""
        config_path = model_dir / 'config.json'
        model_path = model_dir / 'best_model.pt'
        
        if not config_path.exists() or not model_path.exists():
            return False
            
        with open(config_path) as f:
            config = json.load(f)
        
        # Create model with correct signature per model type
        model_type = config.get('model_type', 'mlp')
        
        model_kwargs = {
            'input_dim': 6,
            'output_dim': 4,
            'hidden_dims': config.get('hidden_dims', [256, 256, 128]),
            'activation': config.get('activation', 'silu'),
        }
        
        if model_type in ['mlp', 'residual_mlp']:
            model_kwargs['dropout'] = config.get('dropout', 0.0)
        elif model_type == 'pinn':
            model_kwargs['physics_weight'] = config.get('physics_weight', 0.1)
        elif model_type == 'rk_pinn':
            model_kwargs['n_stages'] = config.get('n_stages', 4)
            
        model = create_model(model_type, **model_kwargs)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device).eval()
        
        self.models[model_dir.name] = {
            'model': model,
            'config': config
        }
        return True
    
    def predict(self, model_name: str) -> np.ndarray:
        """Get model predictions."""
        model = self.models[model_name]['model']
        X = torch.tensor(self.data['X'], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return model(X).cpu().numpy()
    
    # ==================== LORENTZ FORCE ANALYSIS ====================
    
    def analyze_lorentz_force(self, model_names: List[str], save_path: Optional[Path] = None):
        """
        Analyze whether models correctly capture Lorentz force effects.
        
        The Lorentz force F = q(v × B) causes:
        - Bending in x-z plane (for vertical B field)
        - No bending in y-z plane (ty should be conserved)
        - Bending proportional to q/p
        
        For a vertical B field (By):
        - d(tx)/dz ∝ q * By / p
        - d(ty)/dz ≈ 0
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        results = {}
        
        for j, model_name in enumerate(model_names):
            pred = self.predict(model_name)
            
            # Compute bending
            dtx = pred[:, 2] - self.data['tx_in']  # Change in tx (x-slope)
            dty = pred[:, 3] - self.data['ty_in']  # Change in ty (y-slope)
            
            dtx_true = self.data['tx_out'] - self.data['tx_in']
            dty_true = self.data['ty_out'] - self.data['ty_in']
            
            qop = self.data['qop']
            
            # Lorentz force predicts: dtx ∝ qop (for constant B)
            # Fit linear relationship
            slope, intercept, r_value, _, _ = stats.linregress(qop, dtx)
            slope_true, intercept_true, r_true, _, _ = stats.linregress(qop, dtx_true)
            
            results[model_name] = {
                'dtx_vs_qop_slope': float(slope),
                'dtx_vs_qop_intercept': float(intercept),
                'dtx_vs_qop_r2': float(r_value**2),
                'true_slope': float(slope_true),
                'true_r2': float(r_true**2),
                'slope_ratio': float(slope / slope_true) if slope_true != 0 else 0,
                'dty_mean': float(np.mean(dty)),
                'dty_std': float(np.std(dty)),
                'dty_true_mean': float(np.mean(dty_true)),
            }
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            
            # Plot dtx vs q/p
            sample_idx = np.random.choice(len(qop), min(5000, len(qop)), replace=False)
            axes[0, 0].scatter(qop[sample_idx], dtx[sample_idx] * 1000, 
                              alpha=0.1, s=1, color=colors[j], label=short_name)
            
            # Plot dty vs q/p (should be flat/zero)
            axes[0, 1].scatter(qop[sample_idx], dty[sample_idx] * 1000,
                              alpha=0.1, s=1, color=colors[j], label=short_name)
            
            # dtx error vs qop
            dtx_err = (dtx - dtx_true) * 1000
            axes[0, 2].scatter(qop[sample_idx], dtx_err[sample_idx],
                              alpha=0.1, s=1, color=colors[j], label=short_name)
        
        # True relationship overlay
        qop_range = np.array([self.data['qop'].min(), self.data['qop'].max()])
        axes[0, 0].plot(qop_range, (slope_true * qop_range + intercept_true) * 1000,
                       'k--', lw=2, label='True (linear fit)')
        axes[0, 1].axhline(y=np.mean(dty_true) * 1000, color='k', linestyle='--', 
                          lw=2, label='True mean')
        axes[0, 2].axhline(y=0, color='k', linestyle='--', lw=2)
        
        axes[0, 0].set_xlabel('q/p [1/GeV]')
        axes[0, 0].set_ylabel('Δtx [mrad]')
        axes[0, 0].set_title('X-Slope Change vs q/p (Lorentz Force)')
        axes[0, 0].legend(fontsize=7, markerscale=5)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('q/p [1/GeV]')
        axes[0, 1].set_ylabel('Δty [mrad]')
        axes[0, 1].set_title('Y-Slope Change vs q/p (Should be ~0)')
        axes[0, 1].legend(fontsize=7, markerscale=5)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_xlabel('q/p [1/GeV]')
        axes[0, 2].set_ylabel('Δtx Error [mrad]')
        axes[0, 2].set_title('X-Slope Prediction Error vs q/p')
        axes[0, 2].legend(fontsize=7, markerscale=5)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bottom row: Summary statistics
        names = [m.replace('_v1', '').replace('_', '\n') for m in model_names]
        
        # Slope ratio (should be 1.0 for perfect Lorentz)
        slope_ratios = [results[m]['slope_ratio'] for m in model_names]
        axes[1, 0].bar(range(len(model_names)), slope_ratios, color=colors)
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Slope Ratio (pred/true)')
        axes[1, 0].set_title('Lorentz Force Learning (1.0 = perfect)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² for dtx vs qop correlation
        r2_values = [results[m]['dtx_vs_qop_r2'] for m in model_names]
        axes[1, 1].bar(range(len(model_names)), r2_values, color=colors)
        axes[1, 1].axhline(y=results[model_names[0]]['true_r2'], 
                          color='red', linestyle='--', lw=2, label=f'True R²={results[model_names[0]]["true_r2"]:.3f}')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        axes[1, 1].set_ylabel('R² (dtx vs q/p)')
        axes[1, 1].set_title('Linear Correlation Quality')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # dty std (should be low - y-slope is conserved)
        dty_stds = [results[m]['dty_std'] * 1000 for m in model_names]
        axes[1, 2].bar(range(len(model_names)), dty_stds, color=colors)
        axes[1, 2].set_xticks(range(len(model_names)))
        axes[1, 2].set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        axes[1, 2].set_ylabel('σ(Δty) [mrad]')
        axes[1, 2].set_title('Y-Slope Conservation (lower = better)')
        axes[1, 2].grid(True, alpha=0.3)
        
        fig.suptitle('Lorentz Force Analysis: Does the Model Learn Correct Physics?', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return results
    
    # ==================== PHASE SPACE COVERAGE ====================
    
    def analyze_phase_space(self, model_names: List[str], save_path: Optional[Path] = None):
        """
        Analyze model performance across the input phase space.
        
        Examines errors as a function of:
        - Initial position (x, y)
        - Initial slopes (tx, ty)
        - Momentum
        - Charge
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Use best model for detailed analysis
        best_model = model_names[0]  # Assume sorted by performance
        pred = self.predict(best_model)
        
        dx = pred[:, 0] - self.data['x_out']
        dy = pred[:, 1] - self.data['y_out']
        pos_err = np.sqrt(dx**2 + dy**2)
        
        # 1. Error vs initial x position
        h1 = axes[0, 0].hexbin(self.data['x_in'], pos_err, 
                               gridsize=50, cmap='YlOrRd', mincnt=1)
        axes[0, 0].set_xlabel('x_in [mm]')
        axes[0, 0].set_ylabel('Position Error [mm]')
        axes[0, 0].set_title('Error vs Initial X')
        plt.colorbar(h1, ax=axes[0, 0], label='Count')
        
        # 2. Error vs initial y position
        h2 = axes[0, 1].hexbin(self.data['y_in'], pos_err,
                               gridsize=50, cmap='YlOrRd', mincnt=1)
        axes[0, 1].set_xlabel('y_in [mm]')
        axes[0, 1].set_ylabel('Position Error [mm]')
        axes[0, 1].set_title('Error vs Initial Y')
        plt.colorbar(h2, ax=axes[0, 1], label='Count')
        
        # 3. Error in x-tx phase space
        h3 = axes[0, 2].hexbin(self.data['tx_in'], dx,
                               gridsize=50, cmap='RdBu_r', mincnt=1)
        axes[0, 2].set_xlabel('tx_in')
        axes[0, 2].set_ylabel('Δx [mm]')
        axes[0, 2].set_title('X Error vs Initial X-Slope')
        plt.colorbar(h3, ax=axes[0, 2], label='Count')
        
        # 4. 2D error map: x_in vs tx_in
        # Bin the data
        x_bins = np.linspace(self.data['x_in'].min(), self.data['x_in'].max(), 30)
        tx_bins = np.linspace(self.data['tx_in'].min(), self.data['tx_in'].max(), 30)
        
        err_map = np.zeros((len(x_bins)-1, len(tx_bins)-1))
        for i in range(len(x_bins)-1):
            for j in range(len(tx_bins)-1):
                mask = ((self.data['x_in'] >= x_bins[i]) & 
                       (self.data['x_in'] < x_bins[i+1]) &
                       (self.data['tx_in'] >= tx_bins[j]) &
                       (self.data['tx_in'] < tx_bins[j+1]))
                if mask.sum() > 0:
                    err_map[i, j] = np.mean(pos_err[mask])
                else:
                    err_map[i, j] = np.nan
        
        im = axes[1, 0].imshow(err_map.T, origin='lower', aspect='auto',
                               extent=[x_bins[0], x_bins[-1], tx_bins[0], tx_bins[-1]],
                               cmap='YlOrRd')
        axes[1, 0].set_xlabel('x_in [mm]')
        axes[1, 0].set_ylabel('tx_in')
        axes[1, 0].set_title(f'Error Map: {best_model.replace("_v1", "")}')
        plt.colorbar(im, ax=axes[1, 0], label='Mean Error [mm]')
        
        # 5. Error vs momentum (log scale)
        h5 = axes[1, 1].hexbin(self.data['momentum'], pos_err,
                               gridsize=50, cmap='YlOrRd', mincnt=1,
                               xscale='log')
        axes[1, 1].set_xlabel('Momentum [GeV]')
        axes[1, 1].set_ylabel('Position Error [mm]')
        axes[1, 1].set_title('Error vs Momentum')
        plt.colorbar(h5, ax=axes[1, 1], label='Count')
        
        # 6. Error distribution by momentum bin
        p_bins = [0.5, 1, 2, 5, 10, 50, 100]
        colors = plt.cm.viridis(np.linspace(0, 1, len(p_bins)-1))
        
        for i in range(len(p_bins)-1):
            mask = (self.data['momentum'] >= p_bins[i]) & (self.data['momentum'] < p_bins[i+1])
            if mask.sum() > 0:
                axes[1, 2].hist(pos_err[mask], bins=50, alpha=0.5, density=True,
                               label=f'{p_bins[i]}-{p_bins[i+1]} GeV', color=colors[i])
        
        axes[1, 2].set_xlabel('Position Error [mm]')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_title('Error Distribution by Momentum Bin')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        fig.suptitle(f'Phase Space Analysis: {best_model.replace("_v1", "")}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    # ==================== SYSTEMATIC ERROR ANALYSIS ====================
    
    def analyze_systematic_errors(self, model_names: List[str], save_path: Optional[Path] = None):
        """
        Decompose errors into systematic (bias) and statistical (random) components.
        
        For a good model:
        - Mean error (bias) should be ~0
        - Error should be uncorrelated with input variables
        - No visible patterns in residuals
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        results = {}
        
        for j, model_name in enumerate(model_names):
            pred = self.predict(model_name)
            
            dx = pred[:, 0] - self.data['x_out']
            dy = pred[:, 1] - self.data['y_out']
            
            # Compute systematic (mean) and statistical (std) errors
            dx_mean, dx_std = np.mean(dx), np.std(dx)
            dy_mean, dy_std = np.mean(dy), np.std(dy)
            
            # Correlation with input variables
            corr_dx_xin = np.corrcoef(dx, self.data['x_in'])[0, 1]
            corr_dx_txin = np.corrcoef(dx, self.data['tx_in'])[0, 1]
            corr_dx_qop = np.corrcoef(dx, self.data['qop'])[0, 1]
            
            results[model_name] = {
                'dx_bias': float(dx_mean),
                'dx_random': float(dx_std),
                'dy_bias': float(dy_mean),
                'dy_random': float(dy_std),
                'bias_to_random_x': float(abs(dx_mean) / dx_std) if dx_std > 0 else 0,
                'bias_to_random_y': float(abs(dy_mean) / dy_std) if dy_std > 0 else 0,
                'corr_dx_xin': float(corr_dx_xin),
                'corr_dx_txin': float(corr_dx_txin),
                'corr_dx_qop': float(corr_dx_qop),
            }
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            
            # Residual distribution
            axes[0, 0].hist(dx, bins=100, alpha=0.5, label=short_name,
                           color=colors[j], density=True)
        
        axes[0, 0].axvline(x=0, color='red', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Δx [mm]')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('X Residual Distribution')
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Bias comparison
        names = [m.replace('_v1', '').replace('_', '\n') for m in model_names]
        biases = [results[m]['dx_bias'] for m in model_names]
        axes[0, 1].bar(range(len(model_names)), biases, color=colors)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Mean Δx (Bias) [mm]')
        axes[0, 1].set_title('Systematic Bias')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bias-to-random ratio
        b2r = [results[m]['bias_to_random_x'] for m in model_names]
        axes[0, 2].bar(range(len(model_names)), b2r, color=colors)
        axes[0, 2].axhline(y=0.1, color='green', linestyle='--', lw=2, label='Good (<0.1)')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        axes[0, 2].set_ylabel('|Bias| / Random')
        axes[0, 2].set_title('Bias-to-Random Ratio (lower is better)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Correlation analysis
        corr_xin = [results[m]['corr_dx_xin'] for m in model_names]
        corr_txin = [results[m]['corr_dx_txin'] for m in model_names]
        corr_qop = [results[m]['corr_dx_qop'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        axes[1, 0].bar(x - width, corr_xin, width, label='corr(Δx, x_in)', color='blue')
        axes[1, 0].bar(x, corr_txin, width, label='corr(Δx, tx_in)', color='orange')
        axes[1, 0].bar(x + width, corr_qop, width, label='corr(Δx, q/p)', color='green')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Error-Input Correlation (0 = unbiased)')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residual vs predicted (for best model)
        best_model = model_names[0]
        pred = self.predict(best_model)
        dx = pred[:, 0] - self.data['x_out']
        
        sample_idx = np.random.choice(len(dx), min(10000, len(dx)), replace=False)
        axes[1, 1].scatter(pred[sample_idx, 0], dx[sample_idx], alpha=0.1, s=1)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted x [mm]')
        axes[1, 1].set_ylabel('Residual Δx [mm]')
        axes[1, 1].set_title(f'Residual vs Predicted: {best_model.replace("_v1", "")}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # QQ plot for normality
        from scipy import stats as scipy_stats
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(dx, dist='norm')
        axes[1, 2].plot(osm, osr, 'b.', markersize=1, alpha=0.3)
        axes[1, 2].plot(osm, slope * osm + intercept, 'r-', lw=2)
        axes[1, 2].set_xlabel('Theoretical Quantiles')
        axes[1, 2].set_ylabel('Sample Quantiles')
        axes[1, 2].set_title(f'Q-Q Plot: {best_model.replace("_v1", "")} (R²={r**2:.3f})')
        axes[1, 2].grid(True, alpha=0.3)
        
        fig.suptitle('Systematic Error Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return results
    
    # ==================== PINN CONSTRAINT EFFECTIVENESS ====================
    
    def analyze_pinn_constraints(self, model_names: List[str], save_path: Optional[Path] = None):
        """
        Analyze effectiveness of physics-informed constraints.
        
        For PINN models, we examine:
        1. ty conservation (lambda_ty constraint)
        2. Charge-consistent bending (lambda_charge constraint)
        3. Comparison with non-PINN models
        """
        # Separate PINN and non-PINN models
        pinn_models = [m for m in model_names if 'pinn' in m.lower()]
        mlp_models = [m for m in model_names if 'pinn' not in m.lower()]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        all_results = {}
        
        # Analyze ty conservation
        ty_in = self.data['ty_in']
        ty_true = self.data['ty_out']
        dty_true = ty_true - ty_in
        
        colors_pinn = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(pinn_models), 1)))
        colors_mlp = plt.cm.Blues(np.linspace(0.3, 0.9, max(len(mlp_models), 1)))
        
        for j, model_name in enumerate(pinn_models):
            pred = self.predict(model_name)
            dty_pred = pred[:, 3] - ty_in
            
            rmse_dty = np.sqrt(np.mean((dty_pred - dty_true)**2))
            all_results[model_name] = {'rmse_dty': float(rmse_dty), 'type': 'PINN'}
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            axes[0, 0].hist(dty_pred * 1000, bins=100, alpha=0.5,
                           label=f'{short_name} (RMSE={rmse_dty*1000:.2f}mrad)',
                           color=colors_pinn[j], density=True)
        
        for j, model_name in enumerate(mlp_models):
            pred = self.predict(model_name)
            dty_pred = pred[:, 3] - ty_in
            
            rmse_dty = np.sqrt(np.mean((dty_pred - dty_true)**2))
            all_results[model_name] = {'rmse_dty': float(rmse_dty), 'type': 'MLP'}
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            axes[0, 0].hist(dty_pred * 1000, bins=100, alpha=0.5,
                           label=f'{short_name} (RMSE={rmse_dty*1000:.2f}mrad)',
                           color=colors_mlp[j], density=True, linestyle='--')
        
        axes[0, 0].hist(dty_true * 1000, bins=100, alpha=0.7, label='True',
                       color='black', density=True, histtype='step', lw=2)
        axes[0, 0].set_xlabel('Δty [mrad]')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('ty Conservation: PINN vs MLP')
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison bar chart
        all_names = pinn_models + mlp_models
        all_rmse = [all_results[m]['rmse_dty'] * 1000 for m in all_names]
        all_colors = list(colors_pinn[:len(pinn_models)]) + list(colors_mlp[:len(mlp_models)])
        
        axes[0, 1].bar(range(len(all_names)), all_rmse, color=all_colors)
        axes[0, 1].set_xticks(range(len(all_names)))
        axes[0, 1].set_xticklabels([m.replace('_v1', '').replace('_', '\n') for m in all_names],
                                   fontsize=7, rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE(Δty) [mrad]')
        axes[0, 1].set_title('ty Conservation Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Position error comparison
        for j, model_name in enumerate(pinn_models):
            pred = self.predict(model_name)
            pos_err = np.sqrt((pred[:, 0] - self.data['x_out'])**2 + 
                             (pred[:, 1] - self.data['y_out'])**2)
            all_results[model_name]['pos_err_mean'] = float(np.mean(pos_err))
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            axes[0, 2].hist(pos_err, bins=100, alpha=0.5, label=short_name,
                           color=colors_pinn[j], density=True)
        
        for j, model_name in enumerate(mlp_models):
            pred = self.predict(model_name)
            pos_err = np.sqrt((pred[:, 0] - self.data['x_out'])**2 + 
                             (pred[:, 1] - self.data['y_out'])**2)
            all_results[model_name]['pos_err_mean'] = float(np.mean(pos_err))
            
            short_name = model_name.replace('_v1', '').replace('_', ' ')
            axes[0, 2].hist(pos_err, bins=100, alpha=0.5, label=short_name,
                           color=colors_mlp[j], density=True, linestyle='--')
        
        axes[0, 2].set_xlabel('Position Error [mm]')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Position Error Distribution')
        axes[0, 2].legend(fontsize=7)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Trade-off plot: ty conservation vs position error
        for j, model_name in enumerate(pinn_models):
            axes[1, 0].scatter(all_results[model_name]['rmse_dty'] * 1000,
                              all_results[model_name]['pos_err_mean'],
                              s=100, marker='o', color=colors_pinn[j],
                              label=model_name.replace('_v1', ''))
        
        for j, model_name in enumerate(mlp_models):
            axes[1, 0].scatter(all_results[model_name]['rmse_dty'] * 1000,
                              all_results[model_name]['pos_err_mean'],
                              s=100, marker='s', color=colors_mlp[j],
                              label=model_name.replace('_v1', ''))
        
        axes[1, 0].set_xlabel('RMSE(Δty) [mrad] - ty Conservation')
        axes[1, 0].set_ylabel('Mean Position Error [mm]')
        axes[1, 0].set_title('Physics Constraint vs Accuracy Trade-off')
        axes[1, 0].legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot comparison by model type
        pinn_errors = [all_results[m]['pos_err_mean'] for m in pinn_models] if pinn_models else [0]
        mlp_errors = [all_results[m]['pos_err_mean'] for m in mlp_models] if mlp_models else [0]
        
        bp = axes[1, 1].boxplot([pinn_errors, mlp_errors], labels=['PINN', 'MLP'],
                                patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.5)
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('blue')
            bp['boxes'][1].set_alpha(0.5)
        axes[1, 1].set_ylabel('Mean Position Error [mm]')
        axes[1, 1].set_title('Error Distribution by Model Type')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        summary_text = "Summary Statistics:\n"
        summary_text += "-" * 40 + "\n"
        
        if pinn_models:
            pinn_pos_mean = np.mean([all_results[m]['pos_err_mean'] for m in pinn_models])
            pinn_dty_mean = np.mean([all_results[m]['rmse_dty'] for m in pinn_models]) * 1000
            summary_text += f"PINN Models:\n"
            summary_text += f"  Avg Pos Error: {pinn_pos_mean:.4f} mm\n"
            summary_text += f"  Avg RMSE(Δty): {pinn_dty_mean:.4f} mrad\n"
        
        if mlp_models:
            mlp_pos_mean = np.mean([all_results[m]['pos_err_mean'] for m in mlp_models])
            mlp_dty_mean = np.mean([all_results[m]['rmse_dty'] for m in mlp_models]) * 1000
            summary_text += f"\nMLP Models:\n"
            summary_text += f"  Avg Pos Error: {mlp_pos_mean:.4f} mm\n"
            summary_text += f"  Avg RMSE(Δty): {mlp_dty_mean:.4f} mrad\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       transform=axes[1, 2].transAxes, verticalalignment='center')
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary')
        
        fig.suptitle('PINN Constraint Effectiveness Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return all_results


def main():
    """Run physics analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics analysis for track extrapolator models')
    parser.add_argument('--models-dir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='physics_analysis')
    parser.add_argument('--n-samples', type=int, default=100000)
    
    args = parser.parse_args()
    
    analyzer = PhysicsAnalyzer()
    analyzer.load_data(Path(args.data), args.n_samples)
    
    # Load all models
    models_dir = Path(args.models_dir)
    model_names = []
    for d in sorted(models_dir.glob('*_v1')):
        if (d / 'config.json').exists() and (d / 'best_model.pt').exists():
            if analyzer.load_model(d):
                model_names.append(d.name)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(model_names)} models...")
    
    # Run analyses
    print("1. Lorentz force analysis...")
    lorentz_results = analyzer.analyze_lorentz_force(model_names, 
                                                      save_path=output_dir / 'lorentz_force.png')
    
    print("2. Phase space analysis...")
    analyzer.analyze_phase_space(model_names, save_path=output_dir / 'phase_space.png')
    
    print("3. Systematic error analysis...")
    systematic_results = analyzer.analyze_systematic_errors(model_names,
                                                            save_path=output_dir / 'systematic_errors.png')
    
    print("4. PINN constraint analysis...")
    pinn_results = analyzer.analyze_pinn_constraints(model_names,
                                                      save_path=output_dir / 'pinn_constraints.png')
    
    # Save results
    all_results = {
        'lorentz': lorentz_results,
        'systematic': systematic_results,
        'pinn': pinn_results
    }
    with open(output_dir / 'physics_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
