#!/usr/bin/env python3
"""
================================================================================
Comprehensive Model Evaluation for Track Extrapolation
================================================================================

This script provides detailed evaluation of trained models including:
- Overall metrics (MSE, MAE, position/slope errors)
- Momentum-binned performance analysis
- Comparison against RK baseline
- Publication-quality plots
- JSON results export for CI/CD integration

Usage Examples:
    # Basic evaluation
    python evaluate.py --model-path checkpoints/mlp_medium/best_model.pt

    # Detailed analysis with plots
    python evaluate.py --model-path checkpoints/pinn_medium/ --plots --momentum-bins

    # Compare multiple models
    python evaluate.py --compare checkpoints/mlp_medium checkpoints/pinn_medium

    # Export for deployment validation
    python evaluate.py --model-path checkpoints/best/ --export-json results.json

Author: G. Scriven
Date: January 2026
LHCb Track Extrapolation Project
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from architectures import MODEL_REGISTRY, create_model


# =============================================================================
# Constants
# =============================================================================

# Default momentum bins for analysis (GeV)
DEFAULT_MOMENTUM_BINS = [
    (0.5, 2.0, "low"),      # Low momentum: most bending
    (2.0, 10.0, "mid"),     # Medium momentum
    (10.0, 100.0, "high"),  # High momentum: nearly straight
]

# Target metrics for LHCb production
TARGET_METRICS = {
    'pos_mean_mm': 0.5,      # Position accuracy target
    'pos_95_mm': 2.0,        # 95th percentile target
    'slope_mean': 1e-4,      # Slope accuracy target
}


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    model_path: Path,
    device: torch.device
) -> Tuple[nn.Module, dict]:
    """
    Load a trained model from checkpoint directory.
    
    Args:
        model_path: Path to model directory or checkpoint file
        device: Target device (cuda/cpu)
        
    Returns:
        (model, config) tuple
    """
    # Handle both file and directory paths
    if model_path.is_file():
        checkpoint_path = model_path
        model_dir = model_path.parent
    else:
        model_dir = model_path
        checkpoint_path = model_dir / 'best_model.pt'
        if not checkpoint_path.exists():
            # Try finding any .pt file
            pt_files = list(model_dir.glob('*.pt'))
            if pt_files:
                checkpoint_path = pt_files[0]
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    # Load config
    config = {}
    config_paths = [
        model_dir / 'config.json',
        model_dir / 'model_config.json',
    ]
    for cfg_path in config_paths:
        if cfg_path.exists():
            with open(cfg_path) as f:
                config.update(json.load(f))
            break
    
    # Extract model parameters
    model_type = config.get('model_type', 'mlp').lower()
    hidden_dims = config.get('hidden_dims', [128, 128, 64])
    activation = config.get('activation', 'silu')
    
    # Build model kwargs based on type
    model_kwargs = {
        'hidden_dims': hidden_dims,
        'activation': activation,
    }
    
    if model_type in ['pinn', 'rk_pinn']:
        model_kwargs['lambda_pde'] = config.get('lambda_pde', 1.0)
        model_kwargs['lambda_ic'] = config.get('lambda_ic', 1.0)
        if model_type == 'pinn':
            model_kwargs['n_collocation'] = config.get('n_collocation', 10)
    
    if model_type != 'rk_pinn':
        model_kwargs['dropout'] = config.get('dropout', 0.0)
    
    # Create and load model
    model = create_model(model_type, **model_kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load normalization if available
    norm_path = model_dir / 'normalization.json'
    if norm_path.exists():
        model.load_normalization(str(norm_path))
    
    model.to(device)
    model.eval()
    
    return model, config


# =============================================================================
# Data Loading
# =============================================================================

def load_test_data(
    data_path: str,
    max_samples: Optional[int] = None,
    test_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test data from NPZ file.
    
    Args:
        data_path: Path to data NPZ file
        max_samples: Maximum samples to use (None = all)
        test_fraction: Fraction of data to use as test set
        seed: Random seed for reproducible splits
        
    Returns:
        (X, Y, P) arrays: inputs, outputs, momentum
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    X = data['X'].astype(np.float32)  # [N, 5]
    Y = data['Y'][:, :4].astype(np.float32)  # [N, 4] (exclude qop_out)
    P = data['P'].astype(np.float32)  # [N]
    
    # Get dz from metadata or use default
    dz_mean = float(data.get('dz_mean', 2300.0))
    
    # Add dz to input features
    N = X.shape[0]
    dz = np.full((N, 1), dz_mean, dtype=np.float32)
    X = np.hstack([X, dz])  # [N, 6]
    
    # Use last test_fraction of data (after shuffling with fixed seed)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    
    test_start = int(N * (1 - test_fraction))
    test_indices = indices[test_start:]
    
    X = X[test_indices]
    Y = Y[test_indices]
    P = P[test_indices]
    
    # Limit samples if requested
    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        Y = Y[:max_samples]
        P = P[:max_samples]
    
    print(f"  Test samples: {len(X):,}")
    print(f"  dz: {dz_mean:.1f} mm")
    print(f"  P range: [{P.min():.1f}, {P.max():.1f}] GeV")
    
    return X, Y, P


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096
) -> np.ndarray:
    """
    Run model inference on data.
    
    Args:
        model: Trained model
        X: Input features [N, 6]
        device: Compute device
        batch_size: Batch size for inference
        
    Returns:
        Y_pred: Predictions [N, 4]
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    predictions = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]
        pred = model(batch)
        predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def compute_errors(
    Y_pred: np.ndarray,
    Y_true: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute per-sample errors.
    
    Args:
        Y_pred: Predictions [N, 4]
        Y_true: Ground truth [N, 4]
        
    Returns:
        Dictionary of error arrays
    """
    # Individual component errors
    x_err = Y_pred[:, 0] - Y_true[:, 0]
    y_err = Y_pred[:, 1] - Y_true[:, 1]
    tx_err = Y_pred[:, 2] - Y_true[:, 2]
    ty_err = Y_pred[:, 3] - Y_true[:, 3]
    
    # Combined errors
    pos_err = np.sqrt(x_err**2 + y_err**2)
    slope_err = np.sqrt(tx_err**2 + ty_err**2)
    
    return {
        'x_err': x_err,
        'y_err': y_err,
        'tx_err': tx_err,
        'ty_err': ty_err,
        'pos_err': pos_err,
        'slope_err': slope_err,
    }


def compute_metrics(errors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute summary statistics from error arrays.
    
    Args:
        errors: Dictionary of error arrays from compute_errors()
        
    Returns:
        Dictionary of scalar metrics
    """
    pos = errors['pos_err']
    slope = errors['slope_err']
    
    return {
        # Position metrics (mm)
        'pos_mean_mm': float(np.mean(pos)),
        'pos_std_mm': float(np.std(pos)),
        'pos_median_mm': float(np.median(pos)),
        'pos_50_mm': float(np.percentile(pos, 50)),
        'pos_68_mm': float(np.percentile(pos, 68)),  # ~1 sigma
        'pos_90_mm': float(np.percentile(pos, 90)),
        'pos_95_mm': float(np.percentile(pos, 95)),
        'pos_99_mm': float(np.percentile(pos, 99)),
        'pos_max_mm': float(np.max(pos)),
        
        # Slope metrics (dimensionless)
        'slope_mean': float(np.mean(slope)),
        'slope_std': float(np.std(slope)),
        'slope_median': float(np.median(slope)),
        'slope_95': float(np.percentile(slope, 95)),
        'slope_99': float(np.percentile(slope, 99)),
        
        # Individual component metrics
        'x_mean_mm': float(np.mean(np.abs(errors['x_err']))),
        'y_mean_mm': float(np.mean(np.abs(errors['y_err']))),
        'tx_mean': float(np.mean(np.abs(errors['tx_err']))),
        'ty_mean': float(np.mean(np.abs(errors['ty_err']))),
        
        # Bias (should be ~0 for good models)
        'x_bias_mm': float(np.mean(errors['x_err'])),
        'y_bias_mm': float(np.mean(errors['y_err'])),
        'tx_bias': float(np.mean(errors['tx_err'])),
        'ty_bias': float(np.mean(errors['ty_err'])),
    }


def evaluate_by_momentum(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    P: np.ndarray,
    bins: List[Tuple[float, float, str]] = DEFAULT_MOMENTUM_BINS
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics binned by momentum.
    
    Args:
        Y_pred: Predictions [N, 4]
        Y_true: Ground truth [N, 4]
        P: Momentum array [N] in GeV
        bins: List of (p_min, p_max, label) tuples
        
    Returns:
        Dictionary mapping bin labels to metrics
    """
    results = {}
    
    for p_min, p_max, label in bins:
        mask = (P >= p_min) & (P < p_max)
        n_samples = np.sum(mask)
        
        if n_samples == 0:
            continue
        
        errors = compute_errors(Y_pred[mask], Y_true[mask])
        metrics = compute_metrics(errors)
        metrics['n_samples'] = int(n_samples)
        metrics['p_min_gev'] = p_min
        metrics['p_max_gev'] = p_max
        
        results[label] = metrics
    
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def create_plots(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    P: np.ndarray,
    output_dir: Path,
    model_name: str = "model"
) -> None:
    """
    Create evaluation plots.
    
    Args:
        Y_pred: Predictions [N, 4]
        Y_true: Ground truth [N, 4]
        P: Momentum [N]
        output_dir: Directory to save plots
        model_name: Name for plot titles
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = compute_errors(Y_pred, Y_true)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Position error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.hist(errors['pos_err'], bins=100, range=(0, 5), density=True, 
            alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(errors['pos_err']), color='red', linestyle='--', 
               label=f'Mean: {np.mean(errors["pos_err"]):.3f} mm')
    ax.axvline(np.percentile(errors['pos_err'], 95), color='orange', linestyle='--',
               label=f'95%: {np.percentile(errors["pos_err"], 95):.3f} mm')
    ax.set_xlabel('Position Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name}: Position Error Distribution')
    ax.legend()
    
    ax = axes[1]
    ax.hist(errors['slope_err'], bins=100, range=(0, 0.001), density=True,
            alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(errors['slope_err']), color='red', linestyle='--',
               label=f'Mean: {np.mean(errors["slope_err"]):.2e}')
    ax.set_xlabel('Slope Error')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name}: Slope Error Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error vs momentum
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bin by momentum for cleaner plot
    p_bins = np.logspace(np.log10(0.5), np.log10(100), 20)
    p_centers = np.sqrt(p_bins[:-1] * p_bins[1:])
    
    pos_means = []
    slope_means = []
    for i in range(len(p_bins) - 1):
        mask = (P >= p_bins[i]) & (P < p_bins[i+1])
        if np.sum(mask) > 10:
            pos_means.append(np.mean(errors['pos_err'][mask]))
            slope_means.append(np.mean(errors['slope_err'][mask]))
        else:
            pos_means.append(np.nan)
            slope_means.append(np.nan)
    
    ax = axes[0]
    ax.plot(p_centers, pos_means, 'o-', color='steelblue', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Momentum (GeV)')
    ax.set_ylabel('Mean Position Error (mm)')
    ax.set_title(f'{model_name}: Position Error vs Momentum')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(p_centers, slope_means, 's-', color='forestgreen', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Momentum (GeV)')
    ax.set_ylabel('Mean Slope Error')
    ax.set_title(f'{model_name}: Slope Error vs Momentum')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 2D error correlation
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for idx, (name, err) in enumerate([('x', errors['x_err']), 
                                        ('y', errors['y_err']),
                                        ('tx', errors['tx_err']),
                                        ('ty', errors['ty_err'])]):
        ax = axes[idx // 2, idx % 2]
        
        # Sample for speed
        n_plot = min(50000, len(err))
        indices = np.random.choice(len(err), n_plot, replace=False)
        
        if name in ['x', 'y']:
            ax.hist2d(Y_true[indices, idx], err[indices], bins=50, 
                     range=[[-200, 200], [-2, 2]], cmap='Blues', norm=mcolors.LogNorm())
            ax.set_xlabel(f'True {name} (mm)')
            ax.set_ylabel(f'{name} Error (mm)')
        else:
            ax.hist2d(Y_true[indices, idx], err[indices], bins=50,
                     range=[[-0.3, 0.3], [-0.0005, 0.0005]], cmap='Greens', norm=mcolors.LogNorm())
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'{name} Error')
        
        ax.set_title(f'{name} Error vs True Value')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_models(
    model_paths: List[Path],
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test set.
    
    Args:
        model_paths: List of model directory paths
        X, Y, P: Test data
        device: Compute device
        
    Returns:
        Dictionary mapping model names to metrics
    """
    results = {}
    
    for path in model_paths:
        path = Path(path)
        model_name = path.name
        
        print(f"\nEvaluating: {model_name}")
        
        try:
            model, config = load_model(path, device)
            Y_pred = predict(model, X, device)
            errors = compute_errors(Y_pred, Y)
            metrics = compute_metrics(errors)
            metrics['model_type'] = config.get('model_type', 'unknown')
            metrics['parameters'] = model.count_parameters()
            results[model_name] = metrics
            
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            continue
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print formatted comparison table."""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"{'Model':<25} {'Type':<10} {'Params':>10} {'Pos Mean':>12} {'Pos 95%':>12} {'Slope':>12}")
    print("-" * 80)
    
    # Sort by position error
    sorted_models = sorted(results.items(), key=lambda x: x[1]['pos_mean_mm'])
    
    for name, m in sorted_models:
        print(f"{name:<25} {m.get('model_type', '?'):<10} "
              f"{m.get('parameters', 0):>10,} "
              f"{m['pos_mean_mm']:>10.4f}mm "
              f"{m['pos_95_mm']:>10.4f}mm "
              f"{m['slope_mean']:>12.2e}")
    
    print("=" * 80)


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_model(
    model_path: Path,
    data_path: str,
    device: torch.device,
    max_samples: Optional[int] = None,
    create_plots_flag: bool = False,
    momentum_bins_flag: bool = False,
    export_json: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model
        data_path: Path to test data
        device: Compute device
        max_samples: Limit test samples
        create_plots_flag: Generate evaluation plots
        momentum_bins_flag: Compute momentum-binned metrics
        export_json: Path to export JSON results
        verbose: Print detailed output
        
    Returns:
        Dictionary containing all evaluation results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'data_path': data_path,
    }
    
    # Load model
    if verbose:
        print(f"\n{'='*60}")
        print("LOADING MODEL")
        print(f"{'='*60}")
    
    model, config = load_model(model_path, device)
    results['config'] = config
    results['parameters'] = model.count_parameters()
    
    if verbose:
        print(f"  Model type: {config.get('model_type', 'unknown')}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Hidden dims: {config.get('hidden_dims', [])}")
    
    # Load data
    if verbose:
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
    
    X, Y, P = load_test_data(data_path, max_samples=max_samples)
    results['n_test_samples'] = len(X)
    
    # Run inference
    if verbose:
        print(f"\n{'='*60}")
        print("RUNNING INFERENCE")
        print(f"{'='*60}")
    
    start_time = time.time()
    Y_pred = predict(model, X, device)
    inference_time = time.time() - start_time
    
    results['inference_time_s'] = inference_time
    results['throughput_samples_per_sec'] = len(X) / inference_time
    
    if verbose:
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Throughput: {len(X)/inference_time:,.0f} samples/sec")
    
    # Compute errors and metrics
    errors = compute_errors(Y_pred, Y)
    metrics = compute_metrics(errors)
    results['metrics'] = metrics
    
    # Check against targets
    results['meets_targets'] = {
        name: metrics.get(name, float('inf')) <= target
        for name, target in TARGET_METRICS.items()
    }
    
    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\nPosition Error:")
        print(f"  Mean:   {metrics['pos_mean_mm']:.4f} mm")
        print(f"  Std:    {metrics['pos_std_mm']:.4f} mm")
        print(f"  Median: {metrics['pos_median_mm']:.4f} mm")
        print(f"  68%:    {metrics['pos_68_mm']:.4f} mm")
        print(f"  95%:    {metrics['pos_95_mm']:.4f} mm")
        print(f"  99%:    {metrics['pos_99_mm']:.4f} mm")
        print(f"  Max:    {metrics['pos_max_mm']:.4f} mm")
        
        print(f"\nSlope Error:")
        print(f"  Mean:   {metrics['slope_mean']:.2e}")
        print(f"  95%:    {metrics['slope_95']:.2e}")
        
        print(f"\nComponent Errors:")
        print(f"  |x|:  {metrics['x_mean_mm']:.4f} mm   (bias: {metrics['x_bias_mm']:+.4f})")
        print(f"  |y|:  {metrics['y_mean_mm']:.4f} mm   (bias: {metrics['y_bias_mm']:+.4f})")
        print(f"  |tx|: {metrics['tx_mean']:.2e}  (bias: {metrics['tx_bias']:+.2e})")
        print(f"  |ty|: {metrics['ty_mean']:.2e}  (bias: {metrics['ty_bias']:+.2e})")
        
        print(f"\nTarget Compliance:")
        for name, passed in results['meets_targets'].items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name}: {status}")
    
    # Momentum-binned analysis
    if momentum_bins_flag:
        if verbose:
            print(f"\n{'='*60}")
            print("MOMENTUM-BINNED ANALYSIS")
            print(f"{'='*60}")
        
        momentum_results = evaluate_by_momentum(Y_pred, Y, P)
        results['momentum_bins'] = momentum_results
        
        if verbose:
            print(f"\n{'Bin':<12} {'N samples':>12} {'Pos Mean':>12} {'Pos 95%':>12} {'Slope':>12}")
            print("-" * 60)
            for label, m in momentum_results.items():
                print(f"{label:<12} {m['n_samples']:>12,} "
                      f"{m['pos_mean_mm']:>10.4f}mm "
                      f"{m['pos_95_mm']:>10.4f}mm "
                      f"{m['slope_mean']:>12.2e}")
    
    # Create plots
    if create_plots_flag:
        if verbose:
            print(f"\n{'='*60}")
            print("GENERATING PLOTS")
            print(f"{'='*60}")
        
        model_dir = model_path if model_path.is_dir() else model_path.parent
        plot_dir = model_dir / 'plots'
        model_name = config.get('model_type', 'model').upper()
        create_plots(Y_pred, Y, P, plot_dir, model_name)
    
    # Export JSON
    if export_json:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json_results = convert(results)
        with open(export_json, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        if verbose:
            print(f"\nResults exported to: {export_json}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained track extrapolation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate.py --model-path checkpoints/mlp_medium/

  # With plots and momentum analysis
  python evaluate.py --model-path checkpoints/pinn_medium/ --plots --momentum-bins

  # Compare multiple models
  python evaluate.py --compare checkpoints/mlp_medium checkpoints/pinn_medium

  # Export results for CI/CD
  python evaluate.py --model-path checkpoints/best/ --export-json results.json
"""
    )
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-path', type=str,
                       help='Path to trained model directory or checkpoint')
    group.add_argument('--compare', type=str, nargs='+',
                       help='Compare multiple models')
    
    # Data
    parser.add_argument('--data-path', type=str,
                        default='../data_generation/data/training_50M.npz',
                        help='Path to test data NPZ file')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum test samples (default: all)')
    
    # Output options
    parser.add_argument('--plots', action='store_true',
                        help='Generate evaluation plots')
    parser.add_argument('--momentum-bins', action='store_true',
                        help='Compute momentum-binned metrics')
    parser.add_argument('--export-json', type=str, default=None,
                        help='Export results to JSON file')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Single model evaluation
    if args.model_path:
        model_path = Path(args.model_path)
        
        evaluate_model(
            model_path=model_path,
            data_path=args.data_path,
            device=device,
            max_samples=args.max_samples,
            create_plots_flag=args.plots,
            momentum_bins_flag=args.momentum_bins,
            export_json=args.export_json,
            verbose=True
        )
    
    # Multi-model comparison
    elif args.compare:
        X, Y, P = load_test_data(args.data_path, max_samples=args.max_samples)
        
        results = compare_models(
            model_paths=[Path(p) for p in args.compare],
            X=X, Y=Y, P=P,
            device=device
        )
        
        print_comparison_table(results)
        
        if args.export_json:
            with open(args.export_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nComparison exported to: {args.export_json}")


if __name__ == '__main__':
    main()
