#!/usr/bin/env python3
"""
Comprehensive Timing Benchmark for ALL Track Extrapolator Models

Includes: MLP, PINN, RK-PINN
Compares against C++ extrapolator baselines.

Author: George William Scriven
Date: January 2026
"""

import json
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from architectures import create_model, MODEL_REGISTRY


def load_model(model_dir: Path, device: torch.device):
    """Load a trained model from directory."""
    config_path = model_dir / 'config.json'
    model_path = model_dir / 'best_model.pt'
    
    if not config_path.exists() or not model_path.exists():
        return None, None
    
    with open(config_path) as f:
        config = json.load(f)
    
    model_type = config.get('model_type', 'mlp')
    hidden_dims = config.get('hidden_dims', [128, 128, 64])
    activation = config.get('activation', 'silu')
    
    # Map model types
    model_type_map = {
        'mlp': 'mlp',
        'pinn': 'pinn',
        'rk_pinn': 'rk_pinn',
    }
    
    mt = model_type_map.get(model_type, 'mlp')
    
    try:
        model = create_model(
            mt,
            hidden_dims=hidden_dims,
            activation=activation,
            input_dim=6,
            output_dim=4
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        
        return model, {'model_type': model_type, 'hidden_dims': hidden_dims, 'parameters': n_params}
    except Exception as e:
        print(f"  Error loading {model_dir.name}: {e}")
        return None, None


def benchmark_model(model, X_tensor, batch_sizes=[1, 32, 256, 1024], 
                   n_warmup=10, n_runs=100, device_type='cpu'):
    """Benchmark inference time for a model."""
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(X_tensor):
            continue
        
        X_batch = X_tensor[:batch_size]
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(X_batch)
        
        if device_type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            if device_type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(X_batch)
            
            if device_type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        time_per_track = avg_time / batch_size * 1e6  # microseconds
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            'batch_time_us': avg_time * 1e6,
            'time_per_track_us': time_per_track,
            'throughput_hz': throughput
        }
    
    return results


def compute_accuracy(model, X_tensor, Y_np, device):
    """Compute position and slope errors."""
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Position error
    pos_errors = np.sqrt((predictions[:, 0] - Y_np[:, 0])**2 + 
                        (predictions[:, 1] - Y_np[:, 1])**2)
    
    # Slope error
    slope_errors = np.sqrt((predictions[:, 2] - Y_np[:, 2])**2 + 
                          (predictions[:, 3] - Y_np[:, 3])**2)
    
    return {
        'pos_mean': float(np.mean(pos_errors)),
        'pos_std': float(np.std(pos_errors)),
        'pos_median': float(np.median(pos_errors)),
        'pos_68': float(np.percentile(pos_errors, 68)),
        'pos_90': float(np.percentile(pos_errors, 90)),
        'pos_95': float(np.percentile(pos_errors, 95)),
        'slope_mean_mrad': float(np.mean(slope_errors) * 1000),
        'slope_median_mrad': float(np.median(slope_errors) * 1000),
    }


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL TIMING BENCHMARK")
    print("="*70)
    
    # Paths
    models_dir = Path(__file__).parent.parent / 'trained_models'
    data_path = Path(__file__).parent.parent / 'data_generation' / 'data' / 'training_50M.npz'
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load test data
    print("\nLoading test data...")
    data = np.load(data_path)
    X = data['X'].astype(np.float32)
    Y = data['Y'].astype(np.float32)[:, :4]
    
    # Add dz column
    dz = 2300.0
    X = np.hstack([X, np.full((len(X), 1), dz, dtype=np.float32)])
    
    # Use test set (last 10%)
    n_test = min(100000, len(X) // 10)
    np.random.seed(42)
    idx = np.random.choice(len(X) // 10, n_test, replace=False)
    test_idx = idx + len(X) - len(X) // 10
    
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    print(f"Test samples: {len(X_test):,}")
    
    # Find all trained models
    model_dirs = sorted(models_dir.glob("*_v1"))
    print(f"\nFound {len(model_dirs)} model directories")
    
    # Categorize models
    mlp_models = [d for d in model_dirs if d.name.startswith('mlp_')]
    pinn_models = [d for d in model_dirs if d.name.startswith('pinn_')]
    rkpinn_models = [d for d in model_dirs if d.name.startswith('rkpinn_')]
    
    print(f"  MLP models: {len(mlp_models)}")
    print(f"  PINN models: {len(pinn_models)}")
    print(f"  RK-PINN models: {len(rkpinn_models)}")
    
    # Benchmark all models
    all_results = {}
    batch_sizes = [1, 32, 256, 1024]
    
    for category, dirs in [('mlp', mlp_models),
                           ('pinn', pinn_models), ('rkpinn', rkpinn_models)]:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING {category.upper()} MODELS ({len(dirs)} models)")
        print("="*70)
        
        for model_dir in tqdm(dirs, desc=category):
            model, config = load_model(model_dir, device)
            if model is None:
                continue
            
            # Timing benchmark
            timing = benchmark_model(model, X_tensor, batch_sizes, device_type='cpu')
            
            # Accuracy
            accuracy = compute_accuracy(model, X_tensor, Y_test, device)
            
            # Get best throughput (batch 1024)
            best_batch = 1024
            if best_batch in timing:
                time_per_track = timing[best_batch]['time_per_track_us']
                throughput = timing[best_batch]['throughput_hz']
            else:
                time_per_track = timing[max(timing.keys())]['time_per_track_us']
                throughput = timing[max(timing.keys())]['throughput_hz']
            
            all_results[model_dir.name] = {
                'model_type': config['model_type'],
                'parameters': config['parameters'],
                'batch_timings': timing,
                'time_per_track_us': time_per_track,
                'throughput_hz': throughput,
                'device': 'cpu',
                **accuracy
            }
            
            print(f"  {model_dir.name}: {time_per_track:.2f} μs, "
                  f"{accuracy['pos_mean']*1000:.1f} μm, {config['parameters']:,} params")
    
    # Save results
    output_path = results_dir / 'timing_results_all.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY BY MODEL TYPE")
    print("="*70)
    
    for category in ['mlp', 'resmlp', 'pinn', 'rk_pinn']:
        cat_results = {k: v for k, v in all_results.items() 
                       if v['model_type'] == category}
        if not cat_results:
            continue
        
        times = [v['time_per_track_us'] for v in cat_results.values()]
        errors = [v['pos_mean'] * 1000 for v in cat_results.values()]  # μm
        
        print(f"\n{category.upper()}:")
        print(f"  Models: {len(cat_results)}")
        print(f"  Time: {np.min(times):.2f} - {np.max(times):.2f} μs")
        print(f"  Error: {np.min(errors):.1f} - {np.max(errors):.1f} μm")
        
        # Best model
        best_acc = min(cat_results.items(), key=lambda x: x[1]['pos_mean'])
        best_speed = min(cat_results.items(), key=lambda x: x[1]['time_per_track_us'])
        print(f"  Best accuracy: {best_acc[0]} ({best_acc[1]['pos_mean']*1000:.1f} μm)")
        print(f"  Best speed: {best_speed[0]} ({best_speed[1]['time_per_track_us']:.2f} μs)")
    
    return all_results


if __name__ == '__main__':
    main()
