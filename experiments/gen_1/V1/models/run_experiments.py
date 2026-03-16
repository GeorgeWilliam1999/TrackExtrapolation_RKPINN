#!/usr/bin/env python3
"""
================================================================================
Run All Training Experiments
================================================================================

Master script for running training experiments across all architectures.
Provides a unified interface for:
- Single model training
- Hyperparameter sweeps
- Architecture comparison experiments
- Reproducible experiment management

Usage Examples:
    # Train single model
    python run_experiments.py --model mlp --preset medium --name baseline_mlp

    # Train all architectures with same config
    python run_experiments.py --all-models --preset medium --epochs 100

    # Run preset experiment suite
    python run_experiments.py --experiment architecture_comparison

    # Hyperparameter sweep
    python run_experiments.py --sweep configs/hp_sweep.yaml

Author: G. Scriven
Date: January 2026
LHCb Track Extrapolation Project
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# Training Configuration
# =============================================================================

# Default: train until convergence with early stopping
# max_epochs is a safety limit, early stopping handles actual convergence
DEFAULT_MAX_EPOCHS = 500
DEFAULT_PATIENCE = 30  # Generous patience for convergence
DEFAULT_MIN_DELTA = 1e-7  # Tight tolerance for convergence detection

# Predefined experiment configurations
# All experiments train until convergence (early stopping) not fixed epochs
EXPERIMENTS = {
    # =========================================================================
    # Architecture Comparisons (same size, different models)
    # =========================================================================
    'architecture_comparison': {
        'description': 'Compare MLP, PINN, and RK_PINN with medium architecture until convergence',
        'models': ['mlp', 'pinn', 'rk_pinn'],
        'preset': 'medium',
        'epochs': DEFAULT_MAX_EPOCHS,  # Max epochs (early stopping determines actual)
        'common_args': {
            'batch_size': 2048,
            'lr': 1e-3,
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
    },
    
    # =========================================================================
    # Architecture Size Sweeps (all models)
    # =========================================================================
    'mlp_architecture_sweep': {
        'description': 'Sweep MLP architecture sizes until convergence',
        'models': ['mlp'],
        'presets': ['tiny', 'small', 'medium', 'large', 'wide'],
        'epochs': DEFAULT_MAX_EPOCHS,
        'common_args': {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
    },
    
    'pinn_architecture_sweep': {
        'description': 'Sweep PINN architecture sizes until convergence',
        'models': ['pinn'],
        'presets': ['tiny', 'small', 'medium', 'large', 'wide'],
        'epochs': DEFAULT_MAX_EPOCHS,
        'common_args': {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
            'lambda_pde': 1.0,
            'lambda_ic': 1.0,
        }
    },
    
    'rk_pinn_architecture_sweep': {
        'description': 'Sweep RK_PINN architecture sizes until convergence',
        'models': ['rk_pinn'],
        'presets': ['tiny', 'small', 'medium', 'large', 'wide'],
        'epochs': DEFAULT_MAX_EPOCHS,
        'common_args': {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
            'lambda_pde': 1.0,
            'lambda_ic': 1.0,
        }
    },
    
    'all_architecture_sweeps': {
        'description': 'Full architecture sweep for ALL model types until convergence',
        'models': ['mlp', 'pinn', 'rk_pinn'],
        'presets': ['tiny', 'small', 'medium', 'large', 'wide'],
        'epochs': DEFAULT_MAX_EPOCHS,
        'common_args': {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
    },
    
    # =========================================================================
    # Hyperparameter Sweeps
    # =========================================================================
    'pinn_lambda_sweep': {
        'description': 'Sweep physics loss weight for PINN until convergence',
        'models': ['pinn'],
        'preset': 'medium',
        'epochs': DEFAULT_MAX_EPOCHS,
        'sweep_param': 'lambda_pde',
        'sweep_values': [0.001, 0.01, 0.1, 1.0, 10.0],
        'common_args': {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
    },
    
    'rk_pinn_lambda_sweep': {
        'description': 'Sweep physics loss weight for RK_PINN until convergence',
        'models': ['rk_pinn'],
        'preset': 'medium',
        'epochs': DEFAULT_MAX_EPOCHS,
        'sweep_param': 'lambda_pde',
        'sweep_values': [0.001, 0.01, 0.1, 1.0, 10.0],
        'common_args': {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
    },
    
    # =========================================================================
    # Quick Tests
    # =========================================================================
    'quick_test': {
        'description': 'Quick test run for debugging (limited data & epochs)',
        'models': ['mlp'],
        'preset': 'tiny',
        'epochs': 10,
        'common_args': {
            'max_samples': 10000,
            'patience': 5,
        }
    },
    
    'quick_test_all': {
        'description': 'Quick test of all model types',
        'models': ['mlp', 'pinn', 'rk_pinn'],
        'preset': 'tiny',
        'epochs': 10,
        'common_args': {
            'max_samples': 10000,
            'patience': 5,
        }
    },
    
    # =========================================================================
    # Production Training
    # =========================================================================
    'production_training': {
        'description': 'Full production training until convergence (large networks)',
        'models': ['mlp', 'pinn', 'rk_pinn'],
        'preset': 'large',
        'epochs': 1000,  # High limit, early stopping handles convergence
        'common_args': {
            'batch_size': 4096,
            'patience': 50,  # Extra patience for large models
            'min_delta': 1e-8,  # Tight convergence
        }
    },
}


def run_training(
    model: str,
    preset: Optional[str] = None,
    hidden_dims: Optional[List[int]] = None,
    name: Optional[str] = None,
    epochs: int = 100,
    data_path: Optional[str] = None,
    extra_args: Optional[Dict] = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a single training job.
    
    Args:
        model: Model type ('mlp', 'pinn', 'rk_pinn')
        preset: Architecture preset ('tiny', 'small', 'medium', 'large')
        hidden_dims: Custom hidden layer sizes (overrides preset)
        name: Experiment name
        epochs: Number of epochs
        data_path: Path to training data
        extra_args: Additional arguments
        dry_run: Print command without executing
        
    Returns:
        CompletedProcess result
    """
    # Build command
    cmd = [sys.executable, 'train.py']
    
    cmd.extend(['--model', model])
    
    if preset:
        cmd.extend(['--preset', preset])
    
    if hidden_dims:
        cmd.extend(['--hidden_dims'] + [str(d) for d in hidden_dims])
    
    cmd.extend(['--epochs', str(epochs)])
    
    if name:
        cmd.extend(['--name', name])
    
    if data_path:
        cmd.extend(['--data_path', data_path])
    
    # Add extra arguments
    if extra_args:
        for key, value in extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            elif isinstance(value, list):
                cmd.extend([f'--{key}'] + [str(v) for v in value])
            else:
                cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {model.upper()}" + (f" ({preset})" if preset else ""))
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN - not executing]")
        return None
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Exit code: {result.returncode}")
    
    return result


def run_experiment(
    experiment_name: str,
    data_path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, any]:
    """
    Run a predefined experiment suite.
    
    Args:
        experiment_name: Name of experiment from EXPERIMENTS dict
        data_path: Override default data path
        dry_run: Print commands without executing
        
    Returns:
        Dictionary of results
    """
    if experiment_name not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return {}
    
    exp = EXPERIMENTS[experiment_name]
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT: {experiment_name}")
    print(f"# {exp['description']}")
    print(f"{'#'*60}")
    
    results = {
        'experiment': experiment_name,
        'description': exp['description'],
        'start_time': datetime.now().isoformat(),
        'runs': [],
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle preset sweep
    if 'presets' in exp:
        for preset in exp['presets']:
            for model in exp['models']:
                name = f"{experiment_name}_{model}_{preset}_{timestamp}"
                
                run_result = run_training(
                    model=model,
                    preset=preset,
                    name=name,
                    epochs=exp.get('epochs', 100),
                    data_path=data_path,
                    extra_args=exp.get('common_args', {}),
                    dry_run=dry_run,
                )
                
                results['runs'].append({
                    'model': model,
                    'preset': preset,
                    'name': name,
                    'returncode': run_result.returncode if run_result else None,
                })
    
    # Handle parameter sweep
    elif 'sweep_param' in exp:
        param = exp['sweep_param']
        for value in exp['sweep_values']:
            for model in exp['models']:
                name = f"{experiment_name}_{model}_{param}_{value}_{timestamp}"
                
                extra_args = exp.get('common_args', {}).copy()
                extra_args[param] = value
                
                run_result = run_training(
                    model=model,
                    preset=exp.get('preset'),
                    name=name,
                    epochs=exp.get('epochs', 100),
                    data_path=data_path,
                    extra_args=extra_args,
                    dry_run=dry_run,
                )
                
                results['runs'].append({
                    'model': model,
                    'preset': exp.get('preset'),
                    param: value,
                    'name': name,
                    'returncode': run_result.returncode if run_result else None,
                })
    
    # Standard multi-model run
    else:
        for model in exp['models']:
            name = f"{experiment_name}_{model}_{timestamp}"
            
            run_result = run_training(
                model=model,
                preset=exp.get('preset'),
                name=name,
                epochs=exp.get('epochs', 100),
                data_path=data_path,
                extra_args=exp.get('common_args', {}),
                dry_run=dry_run,
            )
            
            results['runs'].append({
                'model': model,
                'preset': exp.get('preset'),
                'name': name,
                'returncode': run_result.returncode if run_result else None,
            })
    
    results['end_time'] = datetime.now().isoformat()
    
    # Save experiment log
    if not dry_run:
        log_path = Path('experiments') / f"{experiment_name}_{timestamp}.json"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nExperiment log saved to: {log_path}")
    
    return results


def run_all_models(
    preset: str = 'medium',
    epochs: int = 100,
    data_path: Optional[str] = None,
    extra_args: Optional[Dict] = None,
    dry_run: bool = False,
) -> Dict[str, any]:
    """
    Train all model architectures with the same configuration.
    
    Args:
        preset: Architecture preset
        epochs: Number of epochs
        data_path: Path to training data
        extra_args: Additional arguments
        dry_run: Print commands without executing
        
    Returns:
        Dictionary of results
    """
    models = ['mlp', 'pinn', 'rk_pinn']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'#'*60}")
    print(f"# TRAINING ALL MODELS: {preset} preset")
    print(f"{'#'*60}")
    
    results = {
        'preset': preset,
        'epochs': epochs,
        'start_time': datetime.now().isoformat(),
        'runs': [],
    }
    
    for model in models:
        name = f"all_models_{model}_{preset}_{timestamp}"
        
        run_result = run_training(
            model=model,
            preset=preset,
            name=name,
            epochs=epochs,
            data_path=data_path,
            extra_args=extra_args,
            dry_run=dry_run,
        )
        
        results['runs'].append({
            'model': model,
            'name': name,
            'returncode': run_result.returncode if run_result else None,
        })
    
    results['end_time'] = datetime.now().isoformat()
    
    return results


def list_experiments():
    """Print available experiments."""
    print("\n" + "=" * 70)
    print("Available Experiments (all train until convergence via early stopping)")
    print("=" * 70)
    
    for name, exp in EXPERIMENTS.items():
        print(f"\n  {name}:")
        print(f"    {exp['description']}")
        print(f"    Models: {exp['models']}")
        if 'preset' in exp:
            print(f"    Preset: {exp['preset']}")
        if 'presets' in exp:
            print(f"    Presets: {exp['presets']}")
        if 'sweep_param' in exp:
            print(f"    Sweep: {exp['sweep_param']} = {exp['sweep_values']}")
        
        # Show convergence settings
        common = exp.get('common_args', {})
        patience = common.get('patience', DEFAULT_PATIENCE)
        min_delta = common.get('min_delta', DEFAULT_MIN_DELTA)
        max_epochs = exp.get('epochs', DEFAULT_MAX_EPOCHS)
        print(f"    Convergence: patience={patience}, min_delta={min_delta:.0e}, max_epochs={max_epochs}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run training experiments for track extrapolation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python run_experiments.py --model mlp --preset medium --name my_experiment

  # Train all architectures
  python run_experiments.py --all-models --preset large --epochs 200

  # Run predefined experiment
  python run_experiments.py --experiment architecture_comparison

  # List available experiments
  python run_experiments.py --list-experiments

  # Dry run (show commands without executing)
  python run_experiments.py --experiment quick_test --dry-run
"""
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model', type=str, choices=['mlp', 'pinn', 'rk_pinn'],
                      help='Train single model type')
    mode.add_argument('--all-models', action='store_true',
                      help='Train all model architectures')
    mode.add_argument('--experiment', type=str,
                      help='Run predefined experiment')
    mode.add_argument('--list-experiments', action='store_true',
                      help='List available experiments')
    
    # Model config
    parser.add_argument('--preset', type=str, 
                        choices=['tiny', 'small', 'medium', 'large', 'wide'],
                        default='medium',
                        help='Architecture preset')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                        help='Custom hidden layer dimensions')
    parser.add_argument('--activation', type=str, default='silu',
                        help='Activation function')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # PINN specific
    parser.add_argument('--lambda-pde', type=float, default=1.0,
                        help='PDE loss weight (PINN/RK_PINN)')
    parser.add_argument('--lambda-ic', type=float, default=1.0,
                        help='Initial condition loss weight (PINN/RK_PINN)')
    
    # Data
    parser.add_argument('--data-path', type=str,
                        default='../data_generation/data/training_50M.npz',
                        help='Path to training data')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum training samples')
    
    # Experiment
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    # Execution
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List experiments
    if args.list_experiments:
        list_experiments()
        return
    
    # Build extra args
    extra_args = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'activation': args.activation,
        'lambda_pde': args.lambda_pde,
        'lambda_ic': args.lambda_ic,
    }
    if args.max_samples:
        extra_args['max_samples'] = args.max_samples
    
    # Single model training
    if args.model:
        run_training(
            model=args.model,
            preset=args.preset,
            hidden_dims=args.hidden_dims,
            name=args.name,
            epochs=args.epochs,
            data_path=args.data_path,
            extra_args=extra_args,
            dry_run=args.dry_run,
        )
    
    # All models
    elif args.all_models:
        run_all_models(
            preset=args.preset,
            epochs=args.epochs,
            data_path=args.data_path,
            extra_args=extra_args,
            dry_run=args.dry_run,
        )
    
    # Predefined experiment
    elif args.experiment:
        run_experiment(
            experiment_name=args.experiment,
            data_path=args.data_path,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
