#!/usr/bin/env python3
"""
Compare MLP (data-driven) vs True PINN performance.

This script:
1. Loads both trained models
2. Evaluates them on the same test data
3. Generates comparison plots
4. Reports accuracy and timing metrics

Author: G. Scriven
Date: 2025-12-20
"""

import numpy as np
import struct
import time
import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Load and run MLP model for comparison."""
    
    def __init__(self, model_path: str):
        super().__init__()
        self.load_from_binary(model_path)
    
    def load_from_binary(self, filepath: str):
        """Load model from binary format."""
        with open(filepath, 'rb') as f:
            # Number of layers
            n_layers = struct.unpack('i', f.read(4))[0]
            
            # Load layers
            layers = []
            for i in range(n_layers):
                rows, cols = struct.unpack('ii', f.read(8))
                W = np.frombuffer(f.read(rows * cols * 8), dtype=np.float64).reshape(rows, cols)
                b = np.frombuffer(f.read(rows * 8), dtype=np.float64)
                
                linear = nn.Linear(cols, rows)
                linear.weight.data = torch.FloatTensor(W)
                linear.bias.data = torch.FloatTensor(b)
                layers.append(linear)
                
                if i < n_layers - 1:
                    layers.append(nn.Tanh())
            
            self.network = nn.Sequential(*layers)
            
            # Input normalization
            input_size = struct.unpack('i', f.read(4))[0]
            self.input_mean = torch.FloatTensor(
                np.frombuffer(f.read(input_size * 8), dtype=np.float64))
            self.input_std = torch.FloatTensor(
                np.frombuffer(f.read(input_size * 8), dtype=np.float64))
            
            # Output normalization
            output_size = struct.unpack('i', f.read(4))[0]
            self.output_mean = torch.FloatTensor(
                np.frombuffer(f.read(output_size * 8), dtype=np.float64))
            self.output_std = torch.FloatTensor(
                np.frombuffer(f.read(output_size * 8), dtype=np.float64))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / self.input_std
        out = self.network(x_norm)
        return out * self.output_std + self.output_mean


def generate_test_data(n_samples: int = 1000, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test data using the Python RK integrator."""
    from train_pinn import HighPrecisionRKIntegrator, LHCbMagneticField
    
    np.random.seed(seed)
    
    field = LHCbMagneticField(polarity=1)
    integrator = HighPrecisionRKIntegrator(field, step_size=1.0)
    
    z_in, z_out = 3000.0, 7000.0
    dz = z_out - z_in
    
    X_list = []
    Y_list = []
    
    for i in range(n_samples):
        x0 = np.random.uniform(-900, 900)
        y0 = np.random.uniform(-750, 750)
        tx0 = np.random.uniform(-0.3, 0.3)
        ty0 = np.random.uniform(-0.25, 0.25)
        
        # Both positive and negative charges at 2.5 GeV
        charge = np.random.choice([-1, 1])
        qop = charge / 2500.0  # 2.5 GeV in MeV
        
        state_in = np.array([x0, y0, tx0, ty0, qop])
        state_out = integrator.propagate(state_in, z_in, z_out)
        
        X_list.append([x0, y0, tx0, ty0, qop, dz])
        Y_list.append([state_out[0], state_out[1], state_out[2], state_out[3]])
    
    return np.array(X_list), np.array(Y_list)


def evaluate_model(model: nn.Module, X: np.ndarray, Y: np.ndarray) -> Dict:
    """Evaluate model and compute metrics."""
    model.eval()
    
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        pred = model(X_t).numpy()
    
    errors = pred - Y
    radial_errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
    
    return {
        'predictions': pred,
        'errors': errors,
        'radial_errors': radial_errors,
        'mean_radial': np.mean(radial_errors),
        'std_radial': np.std(radial_errors),
        'max_radial': np.max(radial_errors),
        'p95_radial': np.percentile(radial_errors, 95),
        'mean_x': np.mean(np.abs(errors[:, 0])),
        'mean_y': np.mean(np.abs(errors[:, 1])),
        'mean_tx': np.mean(np.abs(errors[:, 2])),
        'mean_ty': np.mean(np.abs(errors[:, 3]))
    }


def time_model(model: nn.Module, n_trials: int = 1000) -> float:
    """Time model inference."""
    model.eval()
    
    x_test = torch.FloatTensor([[100.0, 50.0, 0.1, 0.05, 1e-4, 4000.0]])
    
    # Warmup
    for _ in range(100):
        with torch.no_grad():
            _ = model(x_test)
    
    # Timing
    start = time.time()
    for _ in range(n_trials):
        with torch.no_grad():
            _ = model(x_test)
    elapsed = time.time() - start
    
    return (elapsed / n_trials) * 1e6  # microseconds


def plot_comparison(mlp_results: Dict, pinn_results: Dict, output_path: str):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Error histograms
    ax = axes[0, 0]
    ax.hist(mlp_results['radial_errors'], bins=50, alpha=0.7, label='MLP', color='blue', density=True)
    ax.hist(pinn_results['radial_errors'], bins=50, alpha=0.7, label='PINN', color='red', density=True)
    ax.axvline(mlp_results['mean_radial'], color='blue', linestyle='--', linewidth=2)
    ax.axvline(pinn_results['mean_radial'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Radial Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Radial Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X error comparison
    ax = axes[0, 1]
    ax.hist(mlp_results['errors'][:, 0], bins=50, alpha=0.7, label='MLP', color='blue', density=True)
    ax.hist(pinn_results['errors'][:, 0], bins=50, alpha=0.7, label='PINN', color='red', density=True)
    ax.set_xlabel('X Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title('X Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y error comparison
    ax = axes[0, 2]
    ax.hist(mlp_results['errors'][:, 1], bins=50, alpha=0.7, label='MLP', color='blue', density=True)
    ax.hist(pinn_results['errors'][:, 1], bins=50, alpha=0.7, label='PINN', color='red', density=True)
    ax.set_xlabel('Y Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Y Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar chart comparison
    ax = axes[1, 0]
    metrics = ['Mean', 'Std', '95th %ile', 'Max']
    mlp_vals = [mlp_results['mean_radial'], mlp_results['std_radial'], 
                mlp_results['p95_radial'], mlp_results['max_radial']]
    pinn_vals = [pinn_results['mean_radial'], pinn_results['std_radial'],
                 pinn_results['p95_radial'], pinn_results['max_radial']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, mlp_vals, width, label='MLP', color='blue', alpha=0.7)
    ax.bar(x + width/2, pinn_vals, width, label='PINN', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Radial Error (mm)')
    ax.set_title('Error Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error scatter plot
    ax = axes[1, 1]
    ax.scatter(mlp_results['errors'][:, 0], mlp_results['errors'][:, 1], 
               alpha=0.3, s=5, label='MLP', color='blue')
    ax.scatter(pinn_results['errors'][:, 0], pinn_results['errors'][:, 1],
               alpha=0.3, s=5, label='PINN', color='red')
    ax.set_xlabel('X Error (mm)')
    ax.set_ylabel('Y Error (mm)')
    ax.set_title('Error Distribution in X-Y Plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Model Comparison Summary
    ========================
    
    MLP (Data-Driven):
    • Mean radial error: {mlp_results['mean_radial']:.2f} mm
    • Std radial error:  {mlp_results['std_radial']:.2f} mm
    • 95th percentile:   {mlp_results['p95_radial']:.2f} mm
    • Max error:         {mlp_results['max_radial']:.2f} mm
    
    PINN (Physics-Informed):
    • Mean radial error: {pinn_results['mean_radial']:.2f} mm
    • Std radial error:  {pinn_results['std_radial']:.2f} mm
    • 95th percentile:   {pinn_results['p95_radial']:.2f} mm
    • Max error:         {pinn_results['max_radial']:.2f} mm
    
    Improvement: {((mlp_results['mean_radial'] - pinn_results['mean_radial']) / mlp_results['mean_radial'] * 100):.1f}%
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("MLP vs True PINN Comparison")
    print("=" * 70)
    print()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "ml_models" / "models"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load models
    mlp_path = models_dir / "mlp_model_cpp_v2.bin"
    pinn_path = models_dir / "pinn_model_true.bin"
    
    if not mlp_path.exists():
        print(f"MLP model not found at {mlp_path}")
        return
    
    print(f"Loading MLP model from: {mlp_path}")
    mlp_model = MLPModel(str(mlp_path))
    
    if pinn_path.exists():
        print(f"Loading PINN model from: {pinn_path}")
        pinn_model = MLPModel(str(pinn_path))
        has_pinn = True
    else:
        print(f"PINN model not found at {pinn_path}")
        print("Run train_true_pinn.py first to train the PINN model")
        has_pinn = False
    
    # Generate test data
    print("\nGenerating test data...")
    X_test, Y_test = generate_test_data(n_samples=500, seed=456)
    print(f"Generated {len(X_test)} test samples")
    
    # Evaluate MLP
    print("\nEvaluating MLP model...")
    mlp_results = evaluate_model(mlp_model, X_test, Y_test)
    mlp_time = time_model(mlp_model)
    
    print(f"  Mean radial error: {mlp_results['mean_radial']:.2f} mm")
    print(f"  Std radial error:  {mlp_results['std_radial']:.2f} mm")
    print(f"  Inference time:    {mlp_time:.2f} μs/track")
    
    if has_pinn:
        # Evaluate PINN
        print("\nEvaluating PINN model...")
        pinn_results = evaluate_model(pinn_model, X_test, Y_test)
        pinn_time = time_model(pinn_model)
        
        print(f"  Mean radial error: {pinn_results['mean_radial']:.2f} mm")
        print(f"  Std radial error:  {pinn_results['std_radial']:.2f} mm")
        print(f"  Inference time:    {pinn_time:.2f} μs/track")
        
        # Generate comparison plot
        print("\nGenerating comparison plots...")
        plot_comparison(mlp_results, pinn_results, str(plots_dir / "mlp_vs_pinn_comparison.png"))
        
        # Print summary
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<25} {'MLP':<15} {'PINN':<15} {'Improvement':<15}")
        print("-" * 70)
        
        improvement = (mlp_results['mean_radial'] - pinn_results['mean_radial']) / mlp_results['mean_radial'] * 100
        print(f"{'Mean radial error (mm)':<25} {mlp_results['mean_radial']:<15.2f} {pinn_results['mean_radial']:<15.2f} {improvement:+.1f}%")
        
        improvement = (mlp_results['max_radial'] - pinn_results['max_radial']) / mlp_results['max_radial'] * 100
        print(f"{'Max radial error (mm)':<25} {mlp_results['max_radial']:<15.2f} {pinn_results['max_radial']:<15.2f} {improvement:+.1f}%")
        
        print(f"{'Inference time (μs)':<25} {mlp_time:<15.2f} {pinn_time:<15.2f}")
        
        # Save results
        results = {
            'mlp': {
                'mean_radial_mm': mlp_results['mean_radial'],
                'std_radial_mm': mlp_results['std_radial'],
                'max_radial_mm': mlp_results['max_radial'],
                'inference_time_us': mlp_time
            },
            'pinn': {
                'mean_radial_mm': pinn_results['mean_radial'],
                'std_radial_mm': pinn_results['std_radial'],
                'max_radial_mm': pinn_results['max_radial'],
                'inference_time_us': pinn_time
            },
            'test_samples': len(X_test)
        }
        
        with open(models_dir / "comparison_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {models_dir / 'comparison_results.json'}")
    
    else:
        print("\nTo compare with PINN, first train it with:")
        print("  python train_true_pinn.py --epochs 2000 --lambda-physics 0.1")


if __name__ == "__main__":
    main()
