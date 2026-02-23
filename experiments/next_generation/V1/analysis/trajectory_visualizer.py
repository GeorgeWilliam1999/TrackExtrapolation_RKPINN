#!/usr/bin/env python3
"""
Interactive Trajectory Visualizer for Track Extrapolator Models

This module provides rich trajectory visualization tools including:
1. 3D trajectory visualization
2. Animated track evolution
3. Error heatmaps and residual plots
4. Comparative trajectory analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from architectures import create_model


class Arrow3D(FancyArrowPatch):
    """3D arrow for trajectory visualization."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class TrajectoryVisualizer:
    """
    Advanced trajectory visualization for track extrapolation models.
    """
    
    def __init__(self, models_dir: Path, data_path: Path):
        self.models_dir = Path(models_dir)
        self.data_path = Path(data_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.data = {}
        
        # Geometry
        self.dz = 2300.0  # mm
        self.z_start = 0.0
        self.z_end = self.dz
        
    def load_data(self, n_samples: int = 10000):
        """Load test data."""
        data = np.load(self.data_path)
        X = data['X'].astype(np.float32)  # [N, 5] - [x, y, tx, ty, qop]
        Y = data['Y'].astype(np.float32)  # [N, 5] - [x, y, tx, ty, qop_out]
        P = data['P'].astype(np.float32)
        
        # Add dz column to match training preprocessing
        dz_col = np.full((len(X), 1), self.dz, dtype=np.float32)
        X = np.hstack([X, dz_col])  # [N, 6] - [x, y, tx, ty, qop, dz]
        Y = Y[:, :4]  # Extract only [x, y, tx, ty]
        
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
        
        print(f"Loaded {len(X_test):,} samples")
        
    def load_model(self, model_dir: Path) -> bool:
        """Load a model."""
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
        self.models[model_dir.name] = {'model': model, 'config': config}
        return True
    
    def predict(self, model_name: str) -> np.ndarray:
        """Get predictions."""
        model = self.models[model_name]['model']
        X = torch.tensor(self.data['X'], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return model(X).cpu().numpy()
    
    def interpolate_trajectory(self, x_start: float, y_start: float,
                                tx_start: float, ty_start: float,
                                x_end: float, y_end: float,
                                tx_end: float, ty_end: float,
                                n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate a track trajectory between start and end points.
        
        Uses a simple parabolic interpolation for curved tracks in magnetic field.
        """
        z = np.linspace(self.z_start, self.z_end, n_points)
        t = (z - self.z_start) / (self.z_end - self.z_start)  # Normalized parameter
        
        # Hermite interpolation for smooth trajectory
        # Position: cubic Hermite spline
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        
        x = h00 * x_start + h10 * tx_start * self.dz + h01 * x_end + h11 * tx_end * self.dz
        y = h00 * y_start + h10 * ty_start * self.dz + h01 * y_end + h11 * ty_end * self.dz
        
        return x, y, z
    
    def plot_3d_trajectories(self, model_names: List[str],
                             track_indices: Optional[List[int]] = None,
                             n_tracks: int = 5,
                             save_path: Optional[Path] = None):
        """
        Create 3D visualization of track trajectories.
        """
        if track_indices is None:
            np.random.seed(42)
            track_indices = np.random.choice(len(self.data['X']), n_tracks, replace=False)
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names) + 1))
        
        for i, idx in enumerate(track_indices):
            # Start position
            x_in = self.data['x_in'][idx]
            y_in = self.data['y_in'][idx]
            tx_in = self.data['tx_in'][idx]
            ty_in = self.data['ty_in'][idx]
            p = self.data['P'][idx]
            q = self.data['charge'][idx]
            
            # True end position
            x_true = self.data['x_out'][idx]
            y_true = self.data['y_out'][idx]
            tx_true = self.data['tx_out'][idx]
            ty_true = self.data['ty_out'][idx]
            
            # Plot true trajectory
            x_traj, y_traj, z_traj = self.interpolate_trajectory(
                x_in, y_in, tx_in, ty_in, x_true, y_true, tx_true, ty_true
            )
            ax.plot(z_traj, x_traj, y_traj, 'k-', lw=2, alpha=0.7,
                   label='True' if i == 0 else '')
            
            # Start point
            ax.scatter([self.z_start], [x_in], [y_in], c='green', s=100, 
                      marker='o', zorder=10, label='Start' if i == 0 else '')
            
            # True end point
            ax.scatter([self.z_end], [x_true], [y_true], c='black', s=100,
                      marker='x', zorder=10, label='True End' if i == 0 else '')
            
            # Model predictions
            for j, model_name in enumerate(model_names):
                pred = self.predict(model_name)
                x_pred = pred[idx, 0]
                y_pred = pred[idx, 1]
                tx_pred = pred[idx, 2]
                ty_pred = pred[idx, 3]
                
                x_traj, y_traj, z_traj = self.interpolate_trajectory(
                    x_in, y_in, tx_in, ty_in, x_pred, y_pred, tx_pred, ty_pred
                )
                
                short_name = model_name.replace('_v1', '')
                ax.plot(z_traj, x_traj, y_traj, '--', color=colors[j+1], lw=1.5, alpha=0.7,
                       label=short_name if i == 0 else '')
                
                # Predicted end point
                ax.scatter([self.z_end], [x_pred], [y_pred], c=colors[j+1], s=50,
                          marker='s', zorder=9)
            
            # Add track label
            ax.text(self.z_start - 100, x_in, y_in, f'#{i+1}\nP={p:.1f}GeV\nq={int(q):+d}',
                   fontsize=8, ha='right')
        
        ax.set_xlabel('Z [mm]', fontsize=12)
        ax.set_ylabel('X [mm]', fontsize=12)
        ax.set_zlabel('Y [mm]', fontsize=12)
        ax.set_title('3D Track Trajectories', fontsize=14)
        ax.legend(loc='upper left', fontsize=8)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_trajectory_gallery(self, model_names: List[str],
                                n_tracks: int = 12,
                                momentum_bins: List[Tuple[float, float]] = None,
                                save_path: Optional[Path] = None):
        """
        Create a gallery of trajectories organized by momentum.
        """
        if momentum_bins is None:
            momentum_bins = [(0.5, 2), (2, 5), (5, 10), (10, 100)]
        
        n_cols = 4
        n_rows = len(momentum_bins)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names) + 1))
        
        for row, (p_min, p_max) in enumerate(momentum_bins):
            # Select tracks in this momentum bin
            mask = (self.data['P'] >= p_min) & (self.data['P'] < p_max)
            valid_idx = np.where(mask)[0]
            
            if len(valid_idx) < n_cols:
                selected = valid_idx
            else:
                np.random.seed(row)
                selected = np.random.choice(valid_idx, n_cols, replace=False)
            
            for col, idx in enumerate(selected):
                ax = axes[row, col]
                
                x_in = self.data['x_in'][idx]
                y_in = self.data['y_in'][idx]
                x_true = self.data['x_out'][idx]
                y_true = self.data['y_out'][idx]
                p = self.data['P'][idx]
                q = self.data['charge'][idx]
                
                # Plot X-Z projection
                ax.plot([self.z_start, self.z_end], [x_in, x_true], 'k-', lw=2,
                       label='True' if col == 0 and row == 0 else '')
                ax.scatter([self.z_start], [x_in], c='green', s=80, marker='o', zorder=10)
                ax.scatter([self.z_end], [x_true], c='black', s=80, marker='x', zorder=10)
                
                for j, model_name in enumerate(model_names):
                    pred = self.predict(model_name)
                    x_pred = pred[idx, 0]
                    
                    short_name = model_name.replace('_v1', '')
                    ax.plot([self.z_start, self.z_end], [x_in, x_pred], '--',
                           color=colors[j+1], lw=1.5,
                           label=short_name if col == 0 and row == 0 else '')
                    ax.scatter([self.z_end], [x_pred], c=colors[j+1], s=40, marker='s', zorder=9)
                
                ax.set_xlabel('Z [mm]' if row == n_rows - 1 else '')
                ax.set_ylabel('X [mm]' if col == 0 else '')
                ax.set_title(f'P={p:.1f} GeV, q={int(q):+d}', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                if col == 0 and row == 0:
                    ax.legend(fontsize=7, loc='best')
            
            # Row label
            axes[row, 0].annotate(f'{p_min}-{p_max} GeV', xy=(-0.3, 0.5),
                                  xycoords='axes fraction', fontsize=12,
                                  fontweight='bold', rotation=90, va='center')
        
        fig.suptitle('Trajectory Gallery by Momentum Range (X-Z Projection)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_error_heatmap(self, model_name: str,
                           save_path: Optional[Path] = None):
        """
        Create error heatmaps showing spatial distribution of prediction errors.
        """
        pred = self.predict(model_name)
        
        dx = pred[:, 0] - self.data['x_out']
        dy = pred[:, 1] - self.data['y_out']
        pos_err = np.sqrt(dx**2 + dy**2)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Error vs input position
        h1 = axes[0, 0].hexbin(self.data['x_in'], self.data['y_in'], C=pos_err,
                               gridsize=40, cmap='YlOrRd', reduce_C_function=np.mean)
        axes[0, 0].set_xlabel('x_in [mm]')
        axes[0, 0].set_ylabel('y_in [mm]')
        axes[0, 0].set_title('Mean Error vs Initial Position')
        plt.colorbar(h1, ax=axes[0, 0], label='Error [mm]')
        
        # Error vs input slopes
        h2 = axes[0, 1].hexbin(self.data['tx_in'], self.data['ty_in'], C=pos_err,
                               gridsize=40, cmap='YlOrRd', reduce_C_function=np.mean)
        axes[0, 1].set_xlabel('tx_in')
        axes[0, 1].set_ylabel('ty_in')
        axes[0, 1].set_title('Mean Error vs Initial Slopes')
        plt.colorbar(h2, ax=axes[0, 1], label='Error [mm]')
        
        # Error vs qop and momentum
        h3 = axes[0, 2].hexbin(self.data['qop'], self.data['momentum'], C=pos_err,
                               gridsize=40, cmap='YlOrRd', reduce_C_function=np.mean)
        axes[0, 2].set_xlabel('q/p [1/GeV]')
        axes[0, 2].set_ylabel('Momentum [GeV]')
        axes[0, 2].set_title('Mean Error vs Charge/Momentum')
        plt.colorbar(h3, ax=axes[0, 2], label='Error [mm]')
        
        # Predicted vs true position
        sample_idx = np.random.choice(len(pos_err), min(10000, len(pos_err)), replace=False)
        
        h4 = axes[1, 0].hexbin(self.data['x_out'][sample_idx], pred[sample_idx, 0],
                               gridsize=40, cmap='Blues', mincnt=1)
        axes[1, 0].plot([self.data['x_out'].min(), self.data['x_out'].max()],
                       [self.data['x_out'].min(), self.data['x_out'].max()],
                       'r--', lw=2)
        axes[1, 0].set_xlabel('True x_out [mm]')
        axes[1, 0].set_ylabel('Predicted x_out [mm]')
        axes[1, 0].set_title('X Prediction vs Truth')
        plt.colorbar(h4, ax=axes[1, 0], label='Count')
        
        h5 = axes[1, 1].hexbin(self.data['y_out'][sample_idx], pred[sample_idx, 1],
                               gridsize=40, cmap='Blues', mincnt=1)
        axes[1, 1].plot([self.data['y_out'].min(), self.data['y_out'].max()],
                       [self.data['y_out'].min(), self.data['y_out'].max()],
                       'r--', lw=2)
        axes[1, 1].set_xlabel('True y_out [mm]')
        axes[1, 1].set_ylabel('Predicted y_out [mm]')
        axes[1, 1].set_title('Y Prediction vs Truth')
        plt.colorbar(h5, ax=axes[1, 1], label='Count')
        
        # Error distribution
        axes[1, 2].hist(pos_err, bins=100, edgecolor='black', alpha=0.7)
        axes[1, 2].axvline(np.mean(pos_err), color='red', linestyle='--', lw=2,
                          label=f'Mean: {np.mean(pos_err):.4f} mm')
        axes[1, 2].axvline(np.median(pos_err), color='green', linestyle='--', lw=2,
                          label=f'Median: {np.median(pos_err):.4f} mm')
        axes[1, 2].set_xlabel('Position Error [mm]')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        fig.suptitle(f'Error Analysis: {model_name.replace("_v1", "")}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_charge_separated_trajectories(self, model_names: List[str],
                                           n_tracks_per_charge: int = 5,
                                           save_path: Optional[Path] = None):
        """
        Plot trajectories separated by charge to visualize opposite bending.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names) + 1))
        
        for charge, (ax_xz, ax_yz) in zip([1, -1], [(axes[0, 0], axes[0, 1]), 
                                                      (axes[1, 0], axes[1, 1])]):
            # Select tracks with this charge
            mask = self.data['charge'] == charge
            valid_idx = np.where(mask)[0]
            
            # Prefer high momentum tracks for clearer visualization
            p_sorted = np.argsort(self.data['P'][valid_idx])[::-1]
            selected = valid_idx[p_sorted[:n_tracks_per_charge]]
            
            for i, idx in enumerate(selected):
                x_in = self.data['x_in'][idx]
                y_in = self.data['y_in'][idx]
                x_true = self.data['x_out'][idx]
                y_true = self.data['y_out'][idx]
                p = self.data['P'][idx]
                
                # True trajectory (X-Z)
                ax_xz.plot([0, self.dz], [x_in, x_true], 'k-', lw=2, alpha=0.7,
                          label='True' if i == 0 else '')
                ax_xz.scatter([0], [x_in], c='green', s=60, marker='o', zorder=10)
                ax_xz.scatter([self.dz], [x_true], c='black', s=60, marker='x', zorder=10)
                
                # True trajectory (Y-Z)
                ax_yz.plot([0, self.dz], [y_in, y_true], 'k-', lw=2, alpha=0.7)
                ax_yz.scatter([0], [y_in], c='green', s=60, marker='o', zorder=10)
                ax_yz.scatter([self.dz], [y_true], c='black', s=60, marker='x', zorder=10)
                
                # Model predictions
                for j, model_name in enumerate(model_names):
                    pred = self.predict(model_name)
                    x_pred = pred[idx, 0]
                    y_pred = pred[idx, 1]
                    
                    short_name = model_name.replace('_v1', '')
                    ax_xz.plot([0, self.dz], [x_in, x_pred], '--', color=colors[j+1],
                              lw=1.5, alpha=0.7, label=short_name if i == 0 else '')
                    ax_xz.scatter([self.dz], [x_pred], c=colors[j+1], s=30, marker='s', zorder=9)
                    
                    ax_yz.plot([0, self.dz], [y_in, y_pred], '--', color=colors[j+1],
                              lw=1.5, alpha=0.7)
                    ax_yz.scatter([self.dz], [y_pred], c=colors[j+1], s=30, marker='s', zorder=9)
            
            charge_sign = '+' if charge > 0 else '-'
            ax_xz.set_xlabel('Z [mm]')
            ax_xz.set_ylabel('X [mm]')
            ax_xz.set_title(f'X-Z Projection (q={charge_sign})')
            ax_xz.legend(fontsize=8, loc='best')
            ax_xz.grid(True, alpha=0.3)
            
            ax_yz.set_xlabel('Z [mm]')
            ax_yz.set_ylabel('Y [mm]')
            ax_yz.set_title(f'Y-Z Projection (q={charge_sign})')
            ax_yz.grid(True, alpha=0.3)
        
        fig.suptitle('Charge-Separated Trajectories: Opposite Bending in Magnetic Field', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Trajectory visualization for track extrapolator models')
    parser.add_argument('--models-dir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='trajectory_plots')
    parser.add_argument('--n-samples', type=int, default=10000)
    
    args = parser.parse_args()
    
    viz = TrajectoryVisualizer(
        models_dir=Path(args.models_dir),
        data_path=Path(args.data)
    )
    
    viz.load_data(n_samples=args.n_samples)
    
    # Load models
    model_names = []
    for d in sorted(Path(args.models_dir).glob('*_v1')):
        if (d / 'config.json').exists() and (d / 'best_model.pt').exists():
            if viz.load_model(d):
                model_names.append(d.name)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations for {len(model_names)} models...")
    
    # Select top models for visualization
    top_models = model_names[:4] if len(model_names) > 4 else model_names
    
    print("1. 3D trajectories...")
    viz.plot_3d_trajectories(top_models, n_tracks=5,
                             save_path=output_dir / 'trajectories_3d.png')
    plt.close()
    
    print("2. Trajectory gallery...")
    viz.plot_trajectory_gallery(top_models, n_tracks=4,
                                save_path=output_dir / 'trajectory_gallery.png')
    plt.close()
    
    print("3. Error heatmaps...")
    if model_names:
        viz.plot_error_heatmap(model_names[0],
                              save_path=output_dir / 'error_heatmap.png')
        plt.close()
    
    print("4. Charge-separated trajectories...")
    viz.plot_charge_separated_trajectories(top_models,
                                           save_path=output_dir / 'charge_separated.png')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


if __name__ == '__main__':
    main()
