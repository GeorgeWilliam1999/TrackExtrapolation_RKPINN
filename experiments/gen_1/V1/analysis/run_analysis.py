#!/usr/bin/env python3
"""
Quick Analysis Runner for Track Extrapolator Models

Usage:
    python run_analysis.py --quick          # Quick analysis with 10k samples
    python run_analysis.py --full           # Full analysis with 100k samples
    python run_analysis.py --model mlp_wide_v1  # Analyze specific model
"""

import argparse
import json
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'analysis'))

from analyze_models import TrackExtrapolatorAnalyzer
from physics_analysis import PhysicsAnalyzer
from trajectory_visualizer import TrajectoryVisualizer


def run_quick_analysis(models_dir: Path, data_path: Path, output_dir: Path, n_samples: int = 10000):
    """Run quick analysis with minimal samples."""
    print("=" * 60)
    print("QUICK MODEL ANALYSIS")
    print("=" * 60)
    
    analyzer = TrackExtrapolatorAnalyzer(models_dir, data_path)
    analyzer.load_data(n_samples=n_samples)
    analyzer.load_all_models(pattern='*_v1')
    
    if len(analyzer.models) == 0:
        print("No models found!")
        return
    
    # Compute stats
    model_names = list(analyzer.models.keys())
    stats = analyzer.compute_statistical_summary(model_names)
    
    # Sort by performance
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['pos_mean'])
    
    # Print results
    print(f"\n{'Rank':<5} {'Model':<35} {'Pos Err (mm)':>12} {'Params':>10}")
    print("-" * 70)
    for rank, (name, s) in enumerate(sorted_stats[:20], 1):
        print(f"{rank:<5} {name:<35} {s['pos_mean']:>12.4f} {s['parameters']:>10,}")
    
    # Save to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'quick_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n🏆 Best: {sorted_stats[0][0]} ({sorted_stats[0][1]['pos_mean']:.4f} mm)")
    print(f"Results saved to {output_dir / 'quick_stats.json'}")


def run_full_analysis(models_dir: Path, data_path: Path, output_dir: Path, n_samples: int = 100000):
    """Run comprehensive analysis with all visualizations."""
    print("=" * 60)
    print("FULL MODEL ANALYSIS")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TrackExtrapolatorAnalyzer(models_dir, data_path)
    analyzer.load_data(n_samples=n_samples)
    analyzer.load_all_models(pattern='*_v1')
    
    if len(analyzer.models) == 0:
        print("No models found!")
        return
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(output_dir)
    
    # Run physics analysis
    print("\nRunning physics analysis...")
    physics = PhysicsAnalyzer()
    physics.load_data(data_path, n_samples=n_samples)
    
    model_names = list(analyzer.models.keys())
    for name in model_names:
        physics.load_model(models_dir / name)
    
    # Select top models for detailed analysis
    stats = analyzer.compute_statistical_summary(model_names)
    sorted_models = sorted(stats.items(), key=lambda x: x[1]['pos_mean'])
    top_models = [m[0] for m in sorted_models[:8]]
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    print("  - Lorentz force analysis...")
    physics.analyze_lorentz_force(top_models, save_path=output_dir / 'lorentz_force.png')
    
    print("  - Phase space analysis...")
    physics.analyze_phase_space(top_models, save_path=output_dir / 'phase_space.png')
    
    print("  - Systematic error analysis...")
    physics.analyze_systematic_errors(top_models, save_path=output_dir / 'systematic_errors.png')
    
    print("  - PINN constraint analysis...")
    physics.analyze_pinn_constraints(top_models, save_path=output_dir / 'pinn_constraints.png')
    
    # Run trajectory visualization
    print("\nGenerating trajectory visualizations...")
    viz = TrajectoryVisualizer(models_dir, data_path)
    viz.load_data(n_samples=10000)
    
    for name in top_models[:4]:
        viz.load_model(models_dir / name)
    
    print("  - 3D trajectories...")
    viz.plot_3d_trajectories(top_models[:4], n_tracks=5, save_path=output_dir / 'trajectories_3d.png')
    
    print("  - Trajectory gallery...")
    viz.plot_trajectory_gallery(top_models[:4], save_path=output_dir / 'trajectory_gallery.png')
    
    print("  - Error heatmap...")
    viz.plot_error_heatmap(top_models[0], save_path=output_dir / 'error_heatmap.png')
    
    print("  - Charge-separated trajectories...")
    viz.plot_charge_separated_trajectories(top_models[:4], save_path=output_dir / 'charge_separated.png')
    
    print(f"\n✅ Full analysis complete! Results in {output_dir}")


def analyze_single_model(model_name: str, models_dir: Path, data_path: Path, output_dir: Path):
    """Detailed analysis of a single model."""
    print(f"Analyzing model: {model_name}")
    
    model_dir = models_dir / model_name
    if not model_dir.exists():
        print(f"Model not found: {model_dir}")
        return
    
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(model_dir / 'config.json') as f:
        config = json.load(f)
    
    print(f"\nModel Configuration:")
    print(f"  Type: {config['model_type']}")
    print(f"  Parameters: {config['parameters']:,}")
    print(f"  Hidden dims: {config['hidden_dims']}")
    print(f"  Activation: {config['activation']}")
    
    # Load results if available
    results_path = model_dir / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"\nTraining Results:")
        print(f"  Test Position Error: {results.get('test_pos_error_mm', 'N/A')} mm")
        print(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")
    
    # Create visualizations
    import matplotlib
    matplotlib.use('Agg')
    
    viz = TrajectoryVisualizer(models_dir, data_path)
    viz.load_data(n_samples=10000)
    viz.load_model(model_dir)
    
    print("\nGenerating visualizations...")
    viz.plot_error_heatmap(model_name, save_path=output_dir / 'error_heatmap.png')
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Track Extrapolator Model Analysis')
    parser.add_argument('--quick', action='store_true', help='Quick analysis (10k samples)')
    parser.add_argument('--full', action='store_true', help='Full analysis (100k samples)')
    parser.add_argument('--model', type=str, help='Analyze specific model')
    parser.add_argument('--models-dir', type=str, 
                       default='/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/trained_models')
    parser.add_argument('--data', type=str,
                       default='/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/data_generation/data/training_50M.npz')
    parser.add_argument('--output', type=str,
                       default='/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/analysis/results')
    parser.add_argument('--n-samples', type=int, default=None, help='Number of samples')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    
    if args.model:
        analyze_single_model(args.model, models_dir, data_path, output_dir)
    elif args.full:
        n_samples = args.n_samples or 100000
        run_full_analysis(models_dir, data_path, output_dir, n_samples)
    else:  # Default to quick
        n_samples = args.n_samples or 10000
        run_quick_analysis(models_dir, data_path, output_dir, n_samples)


if __name__ == '__main__':
    main()
