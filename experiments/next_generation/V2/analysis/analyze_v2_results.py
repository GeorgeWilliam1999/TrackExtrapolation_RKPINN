#!/usr/bin/env python3
"""
V2 Training Analysis Script

This script analyzes the shallow-wide V2 models when training completes.
Run after condor jobs finish to generate comparison with V1.

Usage:
    python analyze_v2_results.py [--wait]
    
Options:
    --wait    Wait for training to complete before analyzing
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / 'trained_models'
RESULTS_DIR = PROJECT_ROOT / 'analysis' / 'results'
PLOTS_DIR = PROJECT_ROOT / 'analysis' / 'plots'

# Reference baseline
CPP_RK4_TIME_US = 2.50  # μs/track

def check_training_complete():
    """Check if all V2 models have completed training."""
    v2_models = list(TRAINED_MODELS_DIR.glob('*_v2_*'))
    completed = 0
    total = len(v2_models)
    
    for model_dir in v2_models:
        history_file = model_dir / 'training_history.json'
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                if len(history.get('train_loss', [])) > 0:
                    completed += 1
            except:
                pass
    
    return completed, total

def load_v2_results():
    """Load all V2 training results."""
    results = []
    
    for model_dir in sorted(TRAINED_MODELS_DIR.glob('*_v2_*')):
        config_file = model_dir / 'config.json'
        history_file = model_dir / 'training_history.json'
        
        if not config_file.exists():
            continue
            
        with open(config_file) as f:
            config = json.load(f)
        
        result = {
            'name': model_dir.name,
            'model_type': config.get('model_type', 'unknown').upper(),
            'hidden_dims': config.get('hidden_dims', []),
            'activation': config.get('activation', 'silu'),
        }
        
        # Calculate parameters
        dims = config.get('hidden_dims', [64])
        n_params = 6 * dims[0] + dims[0]  # Input layer
        for i in range(len(dims) - 1):
            n_params += dims[i] * dims[i+1] + dims[i+1]
        n_params += dims[-1] * 4 + 4  # Output layer
        result['params'] = n_params
        
        # Load training history if available
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                result['epochs'] = len(history.get('train_loss', []))
                result['train_loss'] = history.get('train_loss', [])[-1] if history.get('train_loss') else None
                result['val_loss'] = min(history.get('val_loss', [])) if history.get('val_loss') else None
            except:
                result['epochs'] = 0
                result['train_loss'] = None
                result['val_loss'] = None
        else:
            result['epochs'] = 0
            result['train_loss'] = None
            result['val_loss'] = None
        
        results.append(result)
    
    return pd.DataFrame(results)

def load_v1_results():
    """Load V1 training results for comparison."""
    results = []
    
    for model_dir in sorted(TRAINED_MODELS_DIR.iterdir()):
        if '_v2_' in model_dir.name or not model_dir.is_dir():
            continue
            
        config_file = model_dir / 'config.json'
        history_file = model_dir / 'training_history.json'
        
        if not config_file.exists():
            continue
            
        with open(config_file) as f:
            config = json.load(f)
        
        result = {
            'name': model_dir.name,
            'model_type': config.get('model_type', 'unknown').upper(),
        }
        
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                result['val_loss'] = min(history.get('val_loss', [])) if history.get('val_loss') else None
            except:
                result['val_loss'] = None
        else:
            result['val_loss'] = None
        
        results.append(result)
    
    return pd.DataFrame(results)

def analyze_v2():
    """Main analysis function for V2 results."""
    print("=" * 70)
    print("V2 SHALLOW-WIDE MODEL ANALYSIS")
    print("=" * 70)
    
    # Check completion status
    completed, total = check_training_complete()
    print(f"\nTraining Status: {completed}/{total} models complete")
    
    if completed == 0:
        print("\n⚠️  No V2 models have completed training yet!")
        print("    Run 'condor_q' to check job status.")
        return
    
    # Load results
    v2_df = load_v2_results()
    v1_df = load_v1_results()
    
    # Filter to models with results
    v2_valid = v2_df[v2_df['val_loss'].notna()]
    v1_valid = v1_df[v1_df['val_loss'].notna()]
    
    if len(v2_valid) == 0:
        print("\n⚠️  No V2 validation loss data available yet.")
        return
    
    # Summary by model type
    print("\n" + "-" * 70)
    print("V2 Results by Architecture Type")
    print("-" * 70)
    
    for model_type in ['MLP', 'PINN', 'RKPINN', 'RK_PINN']:
        subset = v2_valid[v2_valid['model_type'] == model_type]
        if len(subset) > 0:
            print(f"\n{model_type}:")
            print(f"  Models: {len(subset)}")
            print(f"  Best val loss: {subset['val_loss'].min():.6f}")
            print(f"  Mean val loss: {subset['val_loss'].mean():.6f}")
    
    # Top 10 V2 models
    print("\n" + "-" * 70)
    print("Top 10 V2 Models by Validation Loss")
    print("-" * 70)
    top10 = v2_valid.nsmallest(10, 'val_loss')
    for i, row in enumerate(top10.itertuples(), 1):
        print(f"  {i}. {row.name:<35} {row.val_loss:.6f}")
    
    # Comparison with V1
    print("\n" + "-" * 70)
    print("V2 vs V1 Comparison")
    print("-" * 70)
    
    v1_best = v1_valid['val_loss'].min()
    v2_best = v2_valid['val_loss'].min()
    
    print(f"\n  V1 Best: {v1_best:.6f}")
    print(f"  V2 Best: {v2_best:.6f}")
    
    if v2_best < v1_best:
        improvement = 100 * (v1_best - v2_best) / v1_best
        print(f"\n  ✅ V2 is {improvement:.1f}% better than V1!")
    else:
        degradation = 100 * (v2_best - v1_best) / v1_best
        print(f"\n  ❌ V2 is {degradation:.1f}% worse than V1")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v2_valid.to_csv(RESULTS_DIR / 'v2_training_results.csv', index=False)
    print(f"\n📁 Results saved to: {RESULTS_DIR / 'v2_training_results.csv'}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze V2 training results')
    parser.add_argument('--wait', action='store_true', help='Wait for training to complete')
    args = parser.parse_args()
    
    if args.wait:
        import time
        print("Waiting for V2 training to complete...")
        while True:
            completed, total = check_training_complete()
            print(f"\r  Progress: {completed}/{total} models complete", end='', flush=True)
            if completed == total and total > 0:
                print("\n")
                break
            time.sleep(60)
    
    analyze_v2()

if __name__ == '__main__':
    main()
