#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

Evaluates ALL trained models regardless of whether they have history.json.
Loads models directly and runs inference on test data.

Author: G. Scriven
Date: January 2026

Usage:
    python evaluate_all_models.py --output_dir ../analysis
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from architectures import MODEL_REGISTRY


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ModelResult:
    """Results from evaluating a single model."""
    name: str
    model_type: str
    hidden_dims: List[int]
    n_parameters: int
    
    # Position errors
    pos_mean_mm: float
    pos_std_mm: float
    pos_median_mm: float
    pos_95_mm: float
    pos_99_mm: float
    
    # Slope errors
    tx_mean: float
    ty_mean: float
    slope_mean: float
    
    # Timing
    inference_time_ms: float  # per batch of 1000
    
    # Status
    success: bool
    error_msg: str = ""


# =============================================================================
# Data Loading
# =============================================================================

def load_test_data(data_path: str, max_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test data for evaluation."""
    print(f"Loading test data from {data_path}...")
    
    data = np.load(data_path, mmap_mode='r')
    
    # Take last `max_samples` as test set (different from training)
    total = data['X'].shape[0]
    start_idx = max(0, total - max_samples)
    
    X = np.array(data['X'][start_idx:])
    Y = np.array(data['Y'][start_idx:])
    P = np.array(data['P'][start_idx:])  # This is momentum, not dz!
    
    # Add constant dz=2300 to X (as done in training)
    dz = np.full((X.shape[0], 1), 2300.0, dtype=np.float32)
    X_with_dz = np.hstack([X.astype(np.float32), dz])  # [N, 6]
    
    # Output is just [x, y, tx, ty]
    Y_out = Y[:, :4].astype(np.float32)
    
    print(f"  Loaded {len(X)} test samples")
    print(f"  X_with_dz shape: {X_with_dz.shape}")
    print(f"  Y_out shape: {Y_out.shape}")
    
    return X_with_dz, Y_out, P.astype(np.float32)


# =============================================================================
# Model Loading and Evaluation
# =============================================================================

def load_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, dict, dict]:
    """Load a trained model and its configuration."""
    
    # Try different config file names
    config_path = model_dir / 'config.json'
    model_config_path = model_dir / 'model_config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif model_config_path.exists():
        with open(model_config_path, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"No config file found in {model_dir}")
    
    # Load normalization
    norm_path = model_dir / 'normalization.json'
    if norm_path.exists():
        with open(norm_path, 'r') as f:
            norm = json.load(f)
    else:
        norm = None
    
    # Get model class
    model_type = config.get('model_type', 'mlp').lower()
    hidden_dims = config.get('hidden_dims', [64, 32])
    activation = config.get('activation', 'silu')
    
    # Map to model class (handle lowercase variants)
    if model_type in MODEL_REGISTRY:
        model_class = MODEL_REGISTRY[model_type]
    else:
        print(f"  Warning: Unknown model type '{model_type}', using MLP")
        model_class = MODEL_REGISTRY['mlp']
    
    # Create model
    model = model_class(hidden_dims=hidden_dims, activation=activation)
    
    # Load weights
    checkpoint_path = model_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config, norm


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    norm: dict,
    device: torch.device,
    batch_size: int = 1024
) -> Dict:
    """Evaluate model on test data.
    
    Args:
        X: Input features [N, 6] - [x, y, tx, ty, q/p, dz]
        Y: Output targets [N, 4] - [x_out, y_out, tx_out, ty_out]
        P: Momentum for analysis [N]
        norm: Normalization dict (not used - model handles normalization internally)
    """
    
    # Prepare input - X already contains dz
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    
    # Run inference in batches
    # Note: Model handles normalization internally via forward()
    all_preds = []
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_inputs = X_tensor[i:i+batch_size]
            outputs = model(batch_inputs)  # Model normalizes internally
            all_preds.append(outputs)
    
    inference_time = time.time() - start_time
    
    # Concatenate predictions - already denormalized by model
    Y_pred = torch.cat(all_preds, dim=0)
    
    # Compute errors
    pos_errors = torch.sqrt(
        (Y_pred[:, 0] - Y_tensor[:, 0])**2 + 
        (Y_pred[:, 1] - Y_tensor[:, 1])**2
    ).cpu().numpy()
    
    tx_errors = torch.abs(Y_pred[:, 2] - Y_tensor[:, 2]).cpu().numpy()
    ty_errors = torch.abs(Y_pred[:, 3] - Y_tensor[:, 3]).cpu().numpy()
    
    return {
        'pos_mean_mm': float(np.mean(pos_errors)),
        'pos_std_mm': float(np.std(pos_errors)),
        'pos_median_mm': float(np.median(pos_errors)),
        'pos_95_mm': float(np.percentile(pos_errors, 95)),
        'pos_99_mm': float(np.percentile(pos_errors, 99)),
        'tx_mean': float(np.mean(tx_errors)),
        'ty_mean': float(np.mean(ty_errors)),
        'slope_mean': float(np.mean(tx_errors) + np.mean(ty_errors)) / 2,
        'inference_time_ms': float(inference_time * 1000 / (len(X) / 1000)),
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_all_models(
    trained_dir: Path,
    data_path: str,
    device: torch.device,
    max_test_samples: int = 100000
) -> List[ModelResult]:
    """Evaluate all trained models."""
    
    results = []
    
    # Load test data
    X, Y, P = load_test_data(data_path, max_test_samples)
    
    # Find all model directories
    model_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir()])
    
    print(f"\nEvaluating {len(model_dirs)} models...")
    
    for model_dir in tqdm(model_dirs, desc="Evaluating"):
        name = model_dir.name
        
        # Skip if no best_model.pt
        if not (model_dir / 'best_model.pt').exists():
            print(f"  Skipping {name} (no best_model.pt)")
            continue
        
        try:
            # Load model
            model, config, norm = load_model(model_dir, device)
            
            # Get model info
            model_type = config.get('model_type', 'unknown')
            hidden_dims = config.get('hidden_dims', [])
            n_params = sum(p.numel() for p in model.parameters())
            
            # Evaluate
            metrics = evaluate_model(model, X, Y, P, norm, device)
            
            result = ModelResult(
                name=name,
                model_type=model_type,
                hidden_dims=hidden_dims,
                n_parameters=n_params,
                pos_mean_mm=metrics['pos_mean_mm'],
                pos_std_mm=metrics['pos_std_mm'],
                pos_median_mm=metrics['pos_median_mm'],
                pos_95_mm=metrics['pos_95_mm'],
                pos_99_mm=metrics['pos_99_mm'],
                tx_mean=metrics['tx_mean'],
                ty_mean=metrics['ty_mean'],
                slope_mean=metrics['slope_mean'],
                inference_time_ms=metrics['inference_time_ms'],
                success=True,
            )
            
        except Exception as e:
            print(f"  Error evaluating {name}: {e}")
            result = ModelResult(
                name=name,
                model_type='unknown',
                hidden_dims=[],
                n_parameters=0,
                pos_mean_mm=float('inf'),
                pos_std_mm=0,
                pos_median_mm=float('inf'),
                pos_95_mm=float('inf'),
                pos_99_mm=float('inf'),
                tx_mean=float('inf'),
                ty_mean=float('inf'),
                slope_mean=float('inf'),
                inference_time_ms=0,
                success=False,
                error_msg=str(e),
            )
        
        results.append(result)
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(results: List[ModelResult], output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful evaluations
    valid = [r for r in results if r.success and r.pos_mean_mm < float('inf')]
    
    if not valid:
        print("No valid results to plot")
        return
    
    # Color mapping
    type_colors = {
        'mlp': '#1f77b4',
        'pinn': '#2ca02c',
        'rk_pinn': '#d62728',
        'unknown': '#7f7f7f',
    }
    
    # Sort by position error
    valid_sorted = sorted(valid, key=lambda x: x.pos_mean_mm)
    
    # Plot 1: Bar chart of all models
    fig, ax = plt.subplots(figsize=(14, max(6, len(valid_sorted) * 0.3)))
    
    names = [r.name for r in valid_sorted]
    errors = [r.pos_mean_mm for r in valid_sorted]
    colors = [type_colors.get(r.model_type, '#7f7f7f') for r in valid_sorted]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, errors, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Position Error (mm)')
    ax.set_title('Model Comparison: Position Error (Mean)')
    
    # Reference lines
    ax.axvline(0.76, color='red', linestyle='--', linewidth=2, label='Herab (0.76 mm)')
    ax.axvline(0.10, color='green', linestyle='--', linewidth=2, label='BS3 (0.10 mm)')
    ax.axvline(0.01, color='blue', linestyle=':', linewidth=2, label='10 μm target')
    
    # Legend for model types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items() if any(r.model_type == t for r in valid)]
    legend_elements.extend([
        plt.Line2D([0], [0], color='red', linestyle='--', label='Herab'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='BS3'),
        plt.Line2D([0], [0], color='blue', linestyle=':', label='10 μm'),
    ])
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(0, max(1.0, max(errors) * 1.1))
    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_comparison.png', dpi=150)
    plt.savefig(output_dir / 'all_models_comparison.pdf')
    plt.close()
    
    # Plot 2: Parameters vs Accuracy
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_type, color in type_colors.items():
        type_results = [r for r in valid if r.model_type == model_type]
        if not type_results:
            continue
        
        x = [r.n_parameters for r in type_results]
        y = [r.pos_mean_mm for r in type_results]
        
        ax.scatter(x, y, c=color, s=100, label=model_type, alpha=0.7)
        
        for r in type_results:
            ax.annotate(r.name.replace('_v1', ''), 
                       (r.n_parameters, r.pos_mean_mm),
                       fontsize=6, alpha=0.7)
    
    ax.axhline(0.76, color='red', linestyle='--', label='Herab')
    ax.axhline(0.10, color='green', linestyle='--', label='BS3')
    ax.axhline(0.01, color='blue', linestyle=':', label='10 μm')
    
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Model Complexity vs Accuracy')
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_vs_accuracy.png', dpi=150)
    plt.savefig(output_dir / 'complexity_vs_accuracy.pdf')
    plt.close()
    
    # Plot 3: Model type boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    type_groups = {}
    for r in valid:
        if r.model_type not in type_groups:
            type_groups[r.model_type] = []
        type_groups[r.model_type].append(r.pos_mean_mm)
    
    types = list(type_groups.keys())
    
    ax = axes[0]
    bp = ax.boxplot([type_groups[t] for t in types], tick_labels=types, patch_artist=True)
    for patch, t in zip(bp['boxes'], types):
        patch.set_facecolor(type_colors.get(t, '#7f7f7f'))
    ax.axhline(0.76, color='red', linestyle='--', label='Herab')
    ax.axhline(0.10, color='green', linestyle='--', label='BS3')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Error by Model Type')
    ax.set_yscale('log')
    ax.legend()
    
    # Inference speed
    ax = axes[1]
    speed_groups = {}
    for r in valid:
        if r.model_type not in speed_groups:
            speed_groups[r.model_type] = []
        speed_groups[r.model_type].append(r.inference_time_ms)
    
    bp = ax.boxplot([speed_groups[t] for t in types], tick_labels=types, patch_artist=True)
    for patch, t in zip(bp['boxes'], types):
        patch.set_facecolor(type_colors.get(t, '#7f7f7f'))
    ax.set_ylabel('Inference Time (ms per 1k samples)')
    ax.set_title('Inference Speed by Model Type')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_type_boxplots.png', dpi=150)
    plt.savefig(output_dir / 'model_type_boxplots.pdf')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def save_results(results: List[ModelResult], output_dir: Path):
    """Save results to JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid results
    valid = [r for r in results if r.success]
    
    # Sort by position error
    valid_sorted = sorted(valid, key=lambda x: x.pos_mean_mm)
    
    # JSON results
    json_results = []
    for r in valid_sorted:
        json_results.append({
            'name': r.name,
            'model_type': r.model_type,
            'hidden_dims': r.hidden_dims,
            'n_parameters': r.n_parameters,
            'pos_mean_mm': r.pos_mean_mm,
            'pos_mean_um': r.pos_mean_mm * 1000,
            'pos_std_mm': r.pos_std_mm,
            'pos_median_mm': r.pos_median_mm,
            'pos_95_mm': r.pos_95_mm,
            'pos_99_mm': r.pos_99_mm,
            'tx_mean': r.tx_mean,
            'ty_mean': r.ty_mean,
            'slope_mean': r.slope_mean,
            'inference_time_ms': r.inference_time_ms,
        })
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Markdown report
    lines = [
        "# Model Evaluation Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Total models evaluated: {len(valid)}\n",
        "",
        "## All Models Ranked by Position Error\n",
        "| Rank | Model | Type | Params | Pos (μm) | Pos 95% (μm) | Speed (ms/1k) |",
        "|------|-------|------|--------|----------|--------------|---------------|",
    ]
    
    for i, r in enumerate(valid_sorted, 1):
        lines.append(
            f"| {i} | {r.name} | {r.model_type} | {r.n_parameters:,} | "
            f"{r.pos_mean_mm*1000:.1f} | {r.pos_95_mm*1000:.1f} | {r.inference_time_ms:.2f} |"
        )
    
    # Best by type
    lines.extend([
        "\n## Best Model Per Type\n",
        "| Type | Best Model | Position Error (μm) | Parameters |",
        "|------|------------|---------------------|------------|",
    ])
    
    best_per_type = {}
    for r in valid_sorted:
        if r.model_type not in best_per_type:
            best_per_type[r.model_type] = r
    
    for model_type, r in best_per_type.items():
        lines.append(f"| {model_type} | {r.name} | {r.pos_mean_mm*1000:.1f} | {r.n_parameters:,} |")
    
    # Baseline comparison
    lines.extend([
        "\n## Baseline Comparison\n",
        "| Model | Position (μm) | vs Herab (760 μm) | vs BS3 (100 μm) | vs Target (10 μm) |",
        "|-------|---------------|-------------------|-----------------|-------------------|",
    ])
    
    for r in valid_sorted[:10]:
        pos_um = r.pos_mean_mm * 1000
        vs_herab = f"✓ {760/pos_um:.0f}x better" if pos_um < 760 else "✗"
        vs_bs3 = f"✓ {100/pos_um:.0f}x better" if pos_um < 100 else "✗"
        vs_target = f"✓ {10/pos_um:.1f}x better" if pos_um < 10 else f"✗ {pos_um/10:.1f}x worse"
        
        lines.append(f"| {r.name} | {pos_um:.1f} | {vs_herab} | {vs_bs3} | {vs_target} |")
    
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Results saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained models')
    parser.add_argument('--trained_dir', type=str, 
                       default='/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/trained_models',
                       help='Directory with trained models')
    parser.add_argument('--data_path', type=str,
                       default='/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/data_generation/data/training_50M.npz',
                       help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='../analysis',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=100000,
                       help='Maximum test samples')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluate all models
    results = evaluate_all_models(
        trained_dir=Path(args.trained_dir),
        data_path=args.data_path,
        device=device,
        max_test_samples=args.max_samples,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    
    # Generate plots
    plot_comparison(results, output_dir / 'plots')
    
    # Print summary
    valid = [r for r in results if r.success and r.pos_mean_mm < float('inf')]
    valid_sorted = sorted(valid, key=lambda x: x.pos_mean_mm)
    
    print("\n" + "=" * 60)
    print("TOP 10 MODELS")
    print("=" * 60)
    
    for i, r in enumerate(valid_sorted[:10], 1):
        print(f"{i:2}. {r.name:30} | {r.model_type:12} | {r.pos_mean_mm*1000:7.1f} μm | {r.n_parameters:7,} params")
    
    print("\n" + "=" * 60)
    print("BEST BY MODEL TYPE")
    print("=" * 60)
    
    best_per_type = {}
    for r in valid_sorted:
        if r.model_type not in best_per_type:
            best_per_type[r.model_type] = r
            print(f"  {r.model_type:12}: {r.name:30} | {r.pos_mean_mm*1000:.1f} μm")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
