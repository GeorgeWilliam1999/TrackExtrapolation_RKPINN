#!/usr/bin/env python3
"""
Post-Training Analysis Script

After training completes, this script:
1. Scans all completed experiments
2. Registers models in the registry
3. Compares performance across all models
4. Generates comparison plots
5. Identifies best models for deployment

Author: G. Scriven
Date: January 2026

Usage:
    python analyze_results.py --trained_dir ../trained_models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))
from model_registry import ModelRegistry


# =============================================================================
# Result Collection
# =============================================================================

def scan_experiments(trained_dir: str) -> List[dict]:
    """
    Scan trained models directory for completed experiments.
    
    Returns list of experiment info dictionaries.
    """
    trained_dir = Path(trained_dir)
    experiments = []
    
    for exp_dir in trained_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check for required files
        best_model = exp_dir / 'best_model.pt'
        history = exp_dir / 'history.json'
        config = exp_dir / 'config.json'
        
        if not all(f.exists() for f in [best_model, history, config]):
            continue
        
        # Load experiment info
        with open(history, 'r') as f:
            hist_data = json.load(f)
        
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        exp_info = {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'model_type': config_data['model_type'],
            'hidden_dims': config_data['hidden_dims'],
            'activation': config_data['activation'],
            'lambda_pde': config_data.get('lambda_pde', 0.0),
            'n_collocation': config_data.get('n_collocation', 0),
            'best_epoch': hist_data['best_epoch'],
            'training_time_min': hist_data['training_time'] / 60,
            'test_metrics': hist_data['test_final'],
            'has_onnx': (exp_dir / 'exports').exists(),
        }
        
        experiments.append(exp_info)
    
    return experiments


def register_all_models(experiments: List[dict], registry: ModelRegistry):
    """Register all experiments in the model registry."""
    for exp in experiments:
        name = exp['name']
        
        # Skip if already registered
        if registry.get_model(name):
            print(f"  Skipping {name} (already registered)")
            continue
        
        try:
            registry.register_model(
                name=name,
                checkpoint_path=f"{exp['path']}/best_model.pt"
            )
        except Exception as e:
            print(f"  Error registering {name}: {e}")


# =============================================================================
# Analysis and Visualization
# =============================================================================

def create_comparison_table(experiments: List[dict]) -> str:
    """Create markdown comparison table."""
    # Sort by position error
    experiments = sorted(experiments, 
                        key=lambda x: x['test_metrics']['pos_mean_mm'] 
                        if x['test_metrics'] else float('inf'))
    
    lines = [
        "# Model Comparison Results",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## All Models Ranked by Position Error\n",
        "| Rank | Model | Type | Architecture | Pos Error (mm) | Pos 95% (mm) | Train Time |",
        "|------|-------|------|--------------|----------------|--------------|------------|",
    ]
    
    for i, exp in enumerate(experiments, 1):
        metrics = exp['test_metrics']
        if not metrics:
            continue
        
        arch_str = '-'.join(map(str, exp['hidden_dims']))
        lines.append(
            f"| {i} | {exp['name']} | {exp['model_type']} | {arch_str} | "
            f"{metrics['pos_mean_mm']:.4f} | {metrics['pos_95_mm']:.4f} | "
            f"{exp['training_time_min']:.1f}m |"
        )
    
    # Add baseline comparison
    lines.extend([
        "\n## Baseline Comparison\n",
        "| Model | Position Error (mm) | vs Herab | vs BS3 |",
        "|-------|---------------------|----------|--------|",
    ])
    
    herab_err = 0.76
    bs3_err = 0.10
    
    for exp in experiments[:5]:  # Top 5 models
        metrics = exp['test_metrics']
        if not metrics:
            continue
        
        pos_err = metrics['pos_mean_mm']
        vs_herab = "✓ Better" if pos_err < herab_err else "✗ Worse"
        vs_bs3 = "✓ Better" if pos_err < bs3_err else "✗ Worse"
        
        lines.append(f"| {exp['name']} | {pos_err:.4f} | {vs_herab} | {vs_bs3} |")
    
    return '\n'.join(lines)


def plot_model_comparison(experiments: List[dict], output_dir: str):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter experiments with metrics
    valid_exps = [e for e in experiments if e['test_metrics']]
    
    if not valid_exps:
        print("No valid experiments to plot")
        return
    
    # Extract data for plotting
    names = [e['name'] for e in valid_exps]
    types = [e['model_type'] for e in valid_exps]
    pos_errors = [e['test_metrics']['pos_mean_mm'] for e in valid_exps]
    pos_95 = [e['test_metrics']['pos_95_mm'] for e in valid_exps]
    slope_errors = [e['test_metrics']['slope_mean'] for e in valid_exps]
    train_times = [e['training_time_min'] for e in valid_exps]
    
    # Color by type
    type_colors = {'mlp': 'C0', 'pinn': 'C1', 'rk_pinn': 'C2'}
    colors = [type_colors.get(t, 'C3') for t in types]
    
    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Position error bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by error
    sorted_idx = np.argsort(pos_errors)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_errors = [pos_errors[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    
    bars = ax.barh(range(len(sorted_names)), sorted_errors, color=sorted_colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('Position Error (mm)')
    ax.set_title('Model Comparison: Position Error')
    
    # Add baseline lines
    ax.axvline(0.76, color='red', linestyle='--', label='Herab (0.76 mm)')
    ax.axvline(0.10, color='green', linestyle='--', label='BS3 (0.10 mm)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_error_comparison.png', dpi=150)
    plt.savefig(output_dir / 'position_error_comparison.pdf')
    plt.close()
    
    # Plot 2: Error vs Training Time scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_type in type_colors:
        mask = [t == model_type for t in types]
        if not any(mask):
            continue
        
        x = [train_times[i] for i in range(len(mask)) if mask[i]]
        y = [pos_errors[i] for i in range(len(mask)) if mask[i]]
        labels = [names[i] for i in range(len(mask)) if mask[i]]
        
        ax.scatter(x, y, c=type_colors[model_type], label=model_type, s=100, alpha=0.7)
        
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), fontsize=6, alpha=0.7)
    
    ax.axhline(0.76, color='red', linestyle='--', label='Herab (0.76 mm)')
    ax.axhline(0.10, color='green', linestyle='--', label='BS3 (0.10 mm)')
    
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Training Time vs Accuracy Trade-off')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_vs_accuracy.png', dpi=150)
    plt.savefig(output_dir / 'time_vs_accuracy.pdf')
    plt.close()
    
    # Plot 3: Model type comparison boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Position error by type
    type_groups = {}
    for exp in valid_exps:
        t = exp['model_type']
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(exp['test_metrics']['pos_mean_mm'])
    
    ax = axes[0]
    ax.boxplot([type_groups.get(t, []) for t in type_colors.keys()],
               labels=list(type_colors.keys()))
    ax.axhline(0.76, color='red', linestyle='--', label='Herab')
    ax.axhline(0.10, color='green', linestyle='--', label='BS3')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Error by Model Type')
    ax.legend()
    
    # Slope error by type
    type_groups_slope = {}
    for exp in valid_exps:
        t = exp['model_type']
        if t not in type_groups_slope:
            type_groups_slope[t] = []
        type_groups_slope[t].append(exp['test_metrics']['slope_mean'])
    
    ax = axes[1]
    ax.boxplot([type_groups_slope.get(t, []) for t in type_colors.keys()],
               labels=list(type_colors.keys()))
    ax.set_ylabel('Slope Error')
    ax.set_title('Slope Error by Model Type')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_type_comparison.png', dpi=150)
    plt.savefig(output_dir / 'model_type_comparison.pdf')
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


def identify_best_models(experiments: List[dict]) -> dict:
    """Identify best models for different criteria."""
    valid_exps = [e for e in experiments if e['test_metrics']]
    
    if not valid_exps:
        return {}
    
    # Best by position error
    best_pos = min(valid_exps, key=lambda x: x['test_metrics']['pos_mean_mm'])
    
    # Best by 95th percentile
    best_95 = min(valid_exps, key=lambda x: x['test_metrics']['pos_95_mm'])
    
    # Best by slope error
    best_slope = min(valid_exps, key=lambda x: x['test_metrics']['slope_mean'])
    
    # Best per model type
    best_per_type = {}
    for model_type in ['mlp', 'pinn', 'rk_pinn']:
        type_exps = [e for e in valid_exps if e['model_type'] == model_type]
        if type_exps:
            best_per_type[model_type] = min(type_exps, 
                                           key=lambda x: x['test_metrics']['pos_mean_mm'])
    
    return {
        'best_position': best_pos,
        'best_95th': best_95,
        'best_slope': best_slope,
        'best_per_type': best_per_type,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--trained_dir', type=str, default='../trained_models',
                       help='Directory with trained models')
    parser.add_argument('--output_dir', type=str, default='../analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--register', action='store_true',
                       help='Register all models in registry')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POST-TRAINING ANALYSIS")
    print("=" * 60)
    
    # Scan experiments
    print(f"\nScanning: {args.trained_dir}")
    experiments = scan_experiments(args.trained_dir)
    print(f"Found {len(experiments)} completed experiments")
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Register models
    if args.register:
        print("\nRegistering models...")
        registry = ModelRegistry()
        register_all_models(experiments, registry)
    
    # Create comparison table
    print("\nGenerating comparison table...")
    table = create_comparison_table(experiments)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'comparison.md', 'w') as f:
        f.write(table)
    print(f"  Saved to: {output_dir / 'comparison.md'}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_model_comparison(experiments, str(output_dir / 'plots'))
    
    # Identify best models
    print("\nBest models:")
    best = identify_best_models(experiments)
    
    if best:
        print(f"\n  Best position error: {best['best_position']['name']}")
        print(f"    Error: {best['best_position']['test_metrics']['pos_mean_mm']:.4f} mm")
        
        print(f"\n  Best 95th percentile: {best['best_95th']['name']}")
        print(f"    95%: {best['best_95th']['test_metrics']['pos_95_mm']:.4f} mm")
        
        print(f"\n  Best slope error: {best['best_slope']['name']}")
        print(f"    Error: {best['best_slope']['test_metrics']['slope_mean']:.6f}")
        
        print("\n  Best per model type:")
        for model_type, exp in best['best_per_type'].items():
            print(f"    {model_type}: {exp['name']} ({exp['test_metrics']['pos_mean_mm']:.4f} mm)")
        
        # Save best models info
        with open(output_dir / 'best_models.json', 'w') as f:
            json.dump({
                'best_position': best['best_position']['name'],
                'best_95th': best['best_95th']['name'],
                'best_slope': best['best_slope']['name'],
                'best_per_type': {k: v['name'] for k, v in best['best_per_type'].items()},
            }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
