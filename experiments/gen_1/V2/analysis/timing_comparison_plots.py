#!/usr/bin/env python3
"""
Comprehensive timing comparison visualization for neural network extrapolators vs C++ extrapolators.

Creates:
1. Scatter plot of accuracy vs inference time for all models and extrapolators
2. Pareto frontier visualization
3. Multiple comparison views

Author: George William Scriven
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarking" / "results"
OUTPUT_DIR = RESULTS_DIR / "timing_comparison"


def load_data():
    """Load all timing and accuracy data from JSON files."""
    
    # Try to load comprehensive timing results (new format with all models)
    all_timing_path = RESULTS_DIR / "timing_results_all.json"
    if all_timing_path.exists():
        with open(all_timing_path, 'r') as f:
            nn_timing = json.load(f)
        # New format has accuracy included
        nn_accuracy = nn_timing  # Same file
    else:
        # Fallback to old format
        with open(RESULTS_DIR / "timing_results.json", 'r') as f:
            nn_timing = json.load(f)
        with open(RESULTS_DIR / "quick_stats.json", 'r') as f:
            nn_accuracy = json.load(f)
    
    # Load C++ extrapolator benchmarks
    with open(BENCHMARK_DIR / "benchmark_results.json", 'r') as f:
        cpp_benchmarks = json.load(f)
    
    return nn_timing, nn_accuracy, cpp_benchmarks


def prepare_combined_data(nn_timing, nn_accuracy, cpp_benchmarks):
    """Combine all data into a unified format for plotting."""
    
    data = []
    
    # Process neural network models
    for name, timing_info in nn_timing.items():
        model_type = timing_info.get('model_type', 'mlp')
        
        # Use batch 1024 timing as optimal throughput
        time_per_track = timing_info.get('time_per_track_us', float('inf'))
        throughput = timing_info.get('throughput_hz', 0)
        
        # Position error - check both formats (pos_mean from new format, or accuracy dict)
        if 'pos_mean' in timing_info:
            pos_error_mm = timing_info['pos_mean']
        elif name in nn_accuracy:
            pos_error_mm = nn_accuracy[name].get('pos_mean', float('inf'))
        else:
            continue
        
        pos_error_um = pos_error_mm * 1000  # Convert to micrometers
        
        # Determine category
        if model_type == 'rk_pinn' or 'rk_pinn' in name or 'rkpinn' in name:
            category = 'RK-PINN'
        elif model_type == 'pinn' or 'pinn' in name:
            category = 'PINN'
        else:
            category = 'MLP'
        
        data.append({
            'name': name,
            'category': category,
            'time_us': time_per_track,
            'throughput_hz': throughput,
            'pos_error_mm': pos_error_mm,
            'pos_error_um': pos_error_um,
            'parameters': timing_info.get('parameters', 0),
            'is_cpp': False,
        })
    
    # Process C++ extrapolators
    for name, benchmark in cpp_benchmarks.items():
        time_us = benchmark['timing']['mean_us']
        throughput = benchmark['timing']['throughput_tracks_per_sec']
        pos_error_mm = benchmark['accuracy']['mean_position_error_mm']
        
        data.append({
            'name': name,
            'category': 'C++ Extrapolator',
            'time_us': time_us,
            'throughput_hz': throughput,
            'pos_error_mm': pos_error_mm,
            'pos_error_um': pos_error_mm * 1000,
            'parameters': 0,
            'is_cpp': True,
        })
    
    return data


def compute_pareto_frontier(data, x_key='time_us', y_key='pos_error_mm'):
    """Compute Pareto frontier (minimizing both x and y)."""
    
    points = [(d[x_key], d[y_key], d['name']) for d in data]
    points = sorted(points, key=lambda p: p[0])  # Sort by x
    
    pareto_points = []
    min_y = float('inf')
    
    for x, y, name in points:
        if y < min_y:
            pareto_points.append((x, y, name))
            min_y = y
    
    return pareto_points


def create_scatter_plot_all(data, output_dir):
    """Create main scatter plot with all models and extrapolators."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme
    colors = {
        'MLP': '#2E86AB',           # Blue
        'ResidualMLP': '#F18F01',   # Orange
        'PINN': '#E63946',          # Red
        'RK-PINN': '#A23B72',       # Magenta
        'C++ Extrapolator': '#28965A',  # Green
    }
    
    markers = {
        'MLP': 'o',
        'ResidualMLP': 'D',
        'PINN': 'p',
        'RK-PINN': 's',
        'C++ Extrapolator': '^',
    }
    
    # Plot each category
    for category in ['C++ Extrapolator', 'MLP', 'ResidualMLP', 'PINN', 'RK-PINN']:
        cat_data = [d for d in data if d['category'] == category]
        if not cat_data:
            continue
        
        x = [d['time_us'] for d in cat_data]
        y = [d['pos_error_um'] for d in cat_data]
        names = [d['name'] for d in cat_data]
        
        scatter = ax.scatter(x, y, c=colors[category], marker=markers[category],
                           s=100, alpha=0.8, label=category, edgecolors='white', linewidth=0.5)
    
    # Compute and plot Pareto frontier
    pareto_data = [d for d in data if d['pos_error_um'] < 300]  # Exclude extreme outliers
    pareto_points = compute_pareto_frontier(pareto_data, 'time_us', 'pos_error_um')
    
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.6, label='Pareto Frontier')
        ax.fill_between(pareto_x, 0, pareto_y, alpha=0.1, color='gray')
    
    # Labels
    ax.set_xlabel('Inference Time per Track (μs)')
    ax.set_ylabel('Position Error (μm)')
    ax.set_title('Performance Comparison: Neural Networks vs C++ Extrapolators\nAccuracy vs Speed Trade-off')
    
    # Log scale for y-axis to see detail
    ax.set_yscale('log')
    ax.set_ylim(20, 50000)
    ax.set_xlim(0, 12)
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_all_models_accuracy_vs_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_all_models_accuracy_vs_time.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scatter_all_models_accuracy_vs_time.png/pdf")


def create_annotated_scatter_plot(data, output_dir):
    """Create scatter plot with labels for each point."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {
        'MLP': '#2E86AB',
        'ResidualMLP': '#F18F01',
        'PINN': '#E63946',
        'RK-PINN': '#A23B72',
        'C++ Extrapolator': '#28965A',
    }
    
    markers = {
        'MLP': 'o',
        'ResidualMLP': 'D',
        'PINN': 'p',
        'RK-PINN': 's',
        'C++ Extrapolator': '^',
    }
    
    # Filter out extreme outliers for better visualization
    filtered_data = [d for d in data if d['pos_error_um'] < 300]
    
    # Plot each point with annotation
    for d in filtered_data:
        ax.scatter(d['time_us'], d['pos_error_um'], 
                  c=colors[d['category']], marker=markers[d['category']],
                  s=100, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Short name for annotation
        short_name = d['name'].replace('_v1', '').replace('rkpinn_', 'RK:').replace('mlp_', 'M:').replace('resmlp_', 'R:').replace('pinn_', 'P:')
        
        ax.annotate(short_name, (d['time_us'], d['pos_error_um']),
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=7, alpha=0.8)
    
    # Compute and plot Pareto frontier
    pareto_points = compute_pareto_frontier(filtered_data, 'time_us', 'pos_error_um')
    
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.6, label='Pareto Frontier')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['MLP'], 
               markersize=10, label='MLP'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=colors['ResidualMLP'],
               markersize=10, label='ResidualMLP'),
        Line2D([0], [0], marker='p', color='w', markerfacecolor=colors['PINN'],
               markersize=10, label='PINN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['RK-PINN'],
               markersize=10, label='RK-PINN'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['C++ Extrapolator'],
               markersize=10, label='C++ Extrapolator'),
        Line2D([0], [0], linestyle='--', color='black', label='Pareto Frontier'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    ax.set_xlabel('Inference Time per Track (μs)')
    ax.set_ylabel('Position Error (μm)')
    ax.set_title('Performance Comparison with Model Labels\n(Lower-left is better: Faster & More Accurate)')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(10, 300)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_annotated_models.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_annotated_models.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scatter_annotated_models.png/pdf")


def create_throughput_comparison(data, output_dir):
    """Create horizontal bar chart comparing throughput."""
    
    # Sort by throughput
    sorted_data = sorted(data, key=lambda d: d['throughput_hz'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {
        'MLP': '#2E86AB',
        'RK-PINN': '#A23B72',
        'C++ Extrapolator': '#28965A',
    }
    
    names = [d['name'].replace('_v1', '') for d in sorted_data]
    throughputs = [d['throughput_hz'] / 1000 for d in sorted_data]  # Convert to kHz
    bar_colors = [colors[d['category']] for d in sorted_data]
    
    bars = ax.barh(range(len(names)), throughputs, color=bar_colors, edgecolor='white')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Throughput (thousand tracks/second)')
    ax.set_title('Model Throughput Comparison\n(Higher is Better)')
    
    # Add value labels
    for i, (bar, tp) in enumerate(zip(bars, throughputs)):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
               f'{tp:.0f}k', va='center', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['MLP'], label='MLP'),
        Patch(facecolor=colors['RK-PINN'], label='RK-PINN'),
        Patch(facecolor=colors['C++ Extrapolator'], label='C++ Extrapolator'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'throughput_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: throughput_comparison.png/pdf")


def create_pareto_highlight_plot(data, output_dir):
    """Create plot highlighting Pareto-optimal models."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter data for visualization
    filtered_data = [d for d in data if d['pos_error_um'] < 300]
    
    # Compute Pareto frontier
    pareto_points = compute_pareto_frontier(filtered_data, 'time_us', 'pos_error_um')
    pareto_names = {p[2] for p in pareto_points}
    
    # Colors: distinguish Pareto vs non-Pareto
    colors_pareto = {
        'MLP': '#FF6B6B',           # Red (Pareto)
        'RK-PINN': '#4ECDC4',       # Cyan (Pareto)
        'C++ Extrapolator': '#FFE66D',  # Yellow (Pareto)
    }
    colors_other = {
        'MLP': '#2E86AB',           # Blue (non-Pareto)
        'RK-PINN': '#A23B72',       # Magenta (non-Pareto)
        'C++ Extrapolator': '#28965A',  # Green (non-Pareto)
    }
    
    markers = {
        'MLP': 'o',
        'RK-PINN': 's',
        'C++ Extrapolator': '^',
    }
    
    # Plot non-Pareto points first (background)
    for d in filtered_data:
        if d['name'] not in pareto_names:
            ax.scatter(d['time_us'], d['pos_error_um'],
                      c=colors_other[d['category']], marker=markers[d['category']],
                      s=80, alpha=0.4, edgecolors='white', linewidth=0.5)
    
    # Plot Pareto points (foreground)
    for d in filtered_data:
        if d['name'] in pareto_names:
            ax.scatter(d['time_us'], d['pos_error_um'],
                      c=colors_pareto[d['category']], marker=markers[d['category']],
                      s=150, alpha=1.0, edgecolors='black', linewidth=1.5)
            
            # Label Pareto points
            short_name = d['name'].replace('_v1', '').replace('rkpinn_', 'RK:').replace('mlp_', 'M:')
            ax.annotate(short_name, (d['time_us'], d['pos_error_um']),
                       textcoords="offset points", xytext=(8, 8),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot Pareto frontier line
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'k-', linewidth=2.5, alpha=0.8, zorder=1)
        ax.fill_between(pareto_x, 0, pareto_y, alpha=0.08, color='green')
    
    ax.set_xlabel('Inference Time per Track (μs)')
    ax.set_ylabel('Position Error (μm)')
    ax.set_title('Pareto-Optimal Models Highlighted\n(Optimal Trade-off Between Speed and Accuracy)')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(20, 200)
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='MLP (Pareto)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4ECDC4',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='RK-PINN (Pareto)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FFE66D',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='C++ (Pareto)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB',
               markersize=10, alpha=0.4, label='MLP (non-Pareto)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#A23B72',
               markersize=10, alpha=0.4, label='RK-PINN (non-Pareto)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#28965A',
               markersize=10, alpha=0.4, label='C++ (non-Pareto)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier_highlight.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pareto_frontier_highlight.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: pareto_frontier_highlight.png/pdf")


def create_accuracy_vs_params_plot(data, output_dir):
    """Create scatter plot of accuracy vs model parameters (NN only)."""
    
    nn_data = [d for d in data if not d['is_cpp'] and d['pos_error_um'] < 300]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'MLP': '#2E86AB',
        'RK-PINN': '#A23B72',
    }
    
    markers = {
        'MLP': 'o',
        'RK-PINN': 's',
    }
    
    for category in ['MLP', 'RK-PINN']:
        cat_data = [d for d in nn_data if d['category'] == category]
        if not cat_data:
            continue
        
        x = [d['parameters'] / 1000 for d in cat_data]  # kParams
        y = [d['pos_error_um'] for d in cat_data]
        
        ax.scatter(x, y, c=colors[category], marker=markers[category],
                  s=100, alpha=0.8, label=category, edgecolors='white', linewidth=0.5)
        
        for d in cat_data:
            short_name = d['name'].replace('_v1', '').replace('rkpinn_', '').replace('mlp_', '')
            ax.annotate(short_name, (d['parameters']/1000, d['pos_error_um']),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=7, alpha=0.8)
    
    ax.set_xlabel('Number of Parameters (thousands)')
    ax.set_ylabel('Position Error (μm)')
    ax.set_title('Model Accuracy vs Complexity\n(Does Bigger Mean Better?)')
    
    ax.set_xscale('log')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_vs_parameters.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: accuracy_vs_parameters.png/pdf")


def create_category_comparison(data, output_dir):
    """Create grouped bar plot comparing categories."""
    
    # Filter out outliers (Kisel)
    filtered = [d for d in data if d['pos_error_um'] < 1000]
    
    # Compute statistics by category - only include categories that have data
    stats = {}
    for cat in ['C++ Extrapolator', 'MLP', 'RK-PINN']:
        cat_data = [d for d in filtered if d['category'] == cat]
        if cat_data:
            stats[cat] = {
                'mean_time': np.mean([d['time_us'] for d in cat_data]),
                'min_time': min([d['time_us'] for d in cat_data]),
                'max_time': max([d['time_us'] for d in cat_data]),
                'mean_error': np.mean([d['pos_error_um'] for d in cat_data]),
                'min_error': min([d['pos_error_um'] for d in cat_data]),
                'max_error': max([d['pos_error_um'] for d in cat_data]),
                'best_model': min(cat_data, key=lambda d: d['pos_error_um'])['name'],
            }
    
    # Only use categories that have data
    categories = [cat for cat in ['C++ Extrapolator', 'MLP', 'RK-PINN'] if cat in stats]
    color_map = {'C++ Extrapolator': '#28965A', 'MLP': '#2E86AB', 'RK-PINN': '#A23B72'}
    colors = [color_map[cat] for cat in categories]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time comparison
    ax1 = axes[0]
    x = range(len(categories))
    means = [stats[cat]['mean_time'] for cat in categories]
    mins = [stats[cat]['min_time'] for cat in categories]
    maxs = [stats[cat]['max_time'] for cat in categories]
    
    bars = ax1.bar(x, means, color=colors, edgecolor='white', alpha=0.8)
    ax1.errorbar(x, means, yerr=[np.array(means)-np.array(mins), np.array(maxs)-np.array(means)],
                fmt='none', color='black', capsize=5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel('Inference Time (μs)')
    ax1.set_title('Average Inference Time by Category\n(Lower is Better)')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Accuracy comparison
    ax2 = axes[1]
    means = [stats[cat]['mean_error'] for cat in categories]
    mins = [stats[cat]['min_error'] for cat in categories]
    maxs = [stats[cat]['max_error'] for cat in categories]
    
    bars = ax2.bar(x, means, color=colors, edgecolor='white', alpha=0.8)
    ax2.errorbar(x, means, yerr=[np.array(means)-np.array(mins), np.array(maxs)-np.array(means)],
                fmt='none', color='black', capsize=5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Position Error (μm)')
    ax2.set_title('Average Position Error by Category\n(Lower is Better)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'category_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: category_comparison.png/pdf")


def print_summary_table(data):
    """Print summary table of all models."""
    
    print("\n" + "="*100)
    print("TIMING COMPARISON SUMMARY")
    print("="*100)
    
    # Sort by position error
    sorted_data = sorted(data, key=lambda d: d['pos_error_um'])
    
    print(f"\n{'Model':<25} {'Category':<18} {'Time (μs)':<12} {'Error (μm)':<12} {'Throughput (k/s)':<15}")
    print("-"*100)
    
    for d in sorted_data:
        if d['pos_error_um'] < 1000:  # Skip extreme outliers
            print(f"{d['name']:<25} {d['category']:<18} {d['time_us']:<12.2f} {d['pos_error_um']:<12.1f} {d['throughput_hz']/1000:<15.0f}")
    
    # Find Pareto optimal points
    pareto_points = compute_pareto_frontier([d for d in data if d['pos_error_um'] < 300], 'time_us', 'pos_error_um')
    
    print("\n" + "="*100)
    print("PARETO-OPTIMAL MODELS (Best trade-off between speed and accuracy)")
    print("="*100)
    for x, y, name in pareto_points:
        d = next(dd for dd in data if dd['name'] == name)
        print(f"  • {name}: {x:.2f} μs, {y:.1f} μm error - {d['category']}")
    
    # Best in each category
    print("\n" + "="*100)
    print("BEST MODEL IN EACH CATEGORY")
    print("="*100)
    
    categories = set(d['category'] for d in data)
    for cat in categories:
        cat_data = [d for d in data if d['category'] == cat and d['pos_error_um'] < 1000]
        if cat_data:
            best = min(cat_data, key=lambda d: d['pos_error_um'])
            fastest = min(cat_data, key=lambda d: d['time_us'])
            print(f"\n{cat}:")
            print(f"  Most Accurate: {best['name']} - {best['pos_error_um']:.1f} μm")
            print(f"  Fastest:       {fastest['name']} - {fastest['time_us']:.2f} μs")


def main():
    """Main function to generate all plots."""
    
    print("Loading data...")
    nn_timing, nn_accuracy, cpp_benchmarks = load_data()
    
    print("Preparing combined data...")
    data = prepare_combined_data(nn_timing, nn_accuracy, cpp_benchmarks)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots in {OUTPUT_DIR}...")
    
    # Generate all plots
    create_scatter_plot_all(data, OUTPUT_DIR)
    create_annotated_scatter_plot(data, OUTPUT_DIR)
    create_throughput_comparison(data, OUTPUT_DIR)
    create_pareto_highlight_plot(data, OUTPUT_DIR)
    create_accuracy_vs_params_plot(data, OUTPUT_DIR)
    create_category_comparison(data, OUTPUT_DIR)
    
    # Print summary
    print_summary_table(data)
    
    print(f"\n✅ All plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
