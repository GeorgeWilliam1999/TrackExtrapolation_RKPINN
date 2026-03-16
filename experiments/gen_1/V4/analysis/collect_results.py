#!/usr/bin/env python3
"""
Cross-Version Results Collector
Extracts and compares performance metrics across V1, V2, V3 experiments.
"""
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE = Path("/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation")

def collect_v1_results():
    """Extract physical metrics from V1 model history.json files."""
    results = []
    models_dir = BASE / "V1" / "trained_models"
    
    # Also load timing data
    timing = {}
    timing_csv = BASE / "V1" / "results" / "timing_benchmarks.csv"
    if timing_csv.exists():
        with open(timing_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                timing[row['model']] = float(row['time_per_track_us'])
    
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == 'README.md':
            continue
        
        history_path = model_dir / 'history.json'
        config_path = model_dir / 'config.json'
        norm_path = model_dir / 'normalization.json'
        
        if not history_path.exists() or not config_path.exists():
            continue
        
        with open(history_path) as f:
            history = json.load(f)
        with open(config_path) as f:
            config = json.load(f)
        
        # Get test_final metrics (physical units already computed during training)
        test = history.get('test_final', {})
        if not test:
            # Fall back to last val epoch
            val_list = history.get('val', [])
            if val_list:
                test = val_list[-1]
        
        # Get normalization info
        norm = {}
        if norm_path.exists():
            with open(norm_path) as f:
                norm = json.load(f)
        
        # Count parameters from model checkpoint
        model_path = model_dir / 'best_model.pt'
        n_params = 0
        if model_path.exists() and HAS_TORCH:
            try:
                cp = torch.load(model_path, map_location='cpu', weights_only=False)
                state = cp.get('model_state_dict', cp)
                n_params = sum(v.numel() for k, v in state.items() 
                             if 'weight' in k or 'bias' in k)
            except:
                n_params = config.get('parameters', 0)
        else:
            n_params = config.get('parameters', 0)
        
        results.append({
            'version': 'V1',
            'name': model_dir.name,
            'model_type': config.get('model_type', 'unknown'),
            'hidden_dims': str(config.get('hidden_dims', [])),
            'activation': config.get('activation', 'silu'),
            'epochs': config.get('epochs', 10),
            'dz': 'fixed 8000',
            'n_params': n_params,
            'pos_mean_mm': test.get('pos_mean_mm', None),
            'pos_95_mm': test.get('pos_95_mm', None),
            'slope_mean': test.get('slope_mean', None),
            'x_mean_mm': test.get('x_mean_mm', None),
            'y_mean_mm': test.get('y_mean_mm', None),
            'tx_mean': test.get('tx_mean', None),
            'ty_mean': test.get('ty_mean', None),
            'val_loss': history.get('best_val_loss', None),
            'time_us': timing.get(model_dir.name, None),
            'input_std_dz': norm.get('input_std', [0,0,0,0,0,0])[5] if norm else None,
        })
    
    return results


def collect_v2_results():
    """Extract V2 results from v2_model_results.csv."""
    results = []
    csv_path = BASE / "V2" / "results" / "v2_model_results.csv"
    
    if not csv_path.exists():
        print(f"V2 results not found: {csv_path}")
        return results
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'version': 'V2',
                'name': row['name'],
                'model_type': row['model_type'],
                'hidden_dims': row['hidden_dims'],
                'activation': 'silu',  # V2 used silu
                'epochs': None,
                'dz': 'fixed 8000',
                'n_params': int(row.get('parameters', 0)),
                'pos_mean_mm': float(row['pos_mean_mm']) if row.get('pos_mean_mm') else None,
                'pos_95_mm': float(row['pos_90_mm']) if row.get('pos_90_mm') else None,  # V2 has P90, not P95
                'slope_mean': float(row['slope_mean_mrad']) / 1000 if row.get('slope_mean_mrad') else None,
                'x_mean_mm': None,
                'y_mean_mm': None,
                'tx_mean': None,
                'ty_mean': None,
                'val_loss': float(row['val_loss']) if row.get('val_loss') else None,
                'time_us': float(row['time_us']) if row.get('time_us') else None,
                'input_std_dz': None,
            })
    
    return results


def collect_v3_results():
    """Extract V3 results from benchmark_results_v3.csv."""
    results = []
    csv_path = BASE / "V3" / "analysis" / "benchmark_results_v3.csv"
    
    if not csv_path.exists():
        print(f"V3 results not found: {csv_path}")
        return results
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_type = row.get('type', 'unknown')
            # Skip reference methods for now (Linear, Parabolic, RK4)
            results.append({
                'version': 'V3',
                'name': row['model'],
                'model_type': model_type,
                'hidden_dims': '',
                'activation': 'silu',
                'epochs': None,
                'dz': 'variable 500-12000',
                'n_params': int(row.get('n_params', 0)),
                'pos_mean_mm': float(row['pos_mae_mm']) if row.get('pos_mae_mm') else None,
                'pos_rmse_mm': float(row['pos_rmse_mm']) if row.get('pos_rmse_mm') else None,
                'pos_95_mm': float(row['pos_p95_mm']) if row.get('pos_p95_mm') else None,
                'slope_mean': float(row['slope_mae']) if row.get('slope_mae') else None,
                'slope_rmse': float(row['slope_rmse']) if row.get('slope_rmse') else None,
                'x_rmse_mm': float(row['x_rmse_mm']) if row.get('x_rmse_mm') else None,
                'y_rmse_mm': float(row['y_rmse_mm']) if row.get('y_rmse_mm') else None,
                'tx_rmse': float(row['tx_rmse']) if row.get('tx_rmse') else None,
                'ty_rmse': float(row['ty_rmse']) if row.get('ty_rmse') else None,
                'val_loss': None,
                'time_us': float(row['mean_time_us']) if row.get('mean_time_us') else None,
                'time_type': 'C++ single-sample',
                'input_std_dz': None,
            })
    
    return results


def main():
    print("=" * 120)
    print("CROSS-VERSION RESULTS COMPARISON")
    print("=" * 120)
    
    v1 = collect_v1_results()
    v2 = collect_v2_results()
    v3 = collect_v3_results()
    
    # ---- V1 Summary ----
    print(f"\n{'='*120}")
    print("V1 RESULTS — Fixed dz=8000mm, 50M samples, 10 epochs, PyTorch batched timing")
    print(f"{'='*120}")
    
    # Sort by pos_mean
    v1_valid = [r for r in v1 if r['pos_mean_mm'] is not None]
    v1_sorted = sorted(v1_valid, key=lambda x: x['pos_mean_mm'])
    
    print(f"\n{'Rank':<4} {'Model':<30} {'Type':<8} {'Dims':<20} {'Pos Mean':>10} {'Pos P95':>10} {'Slope':>10} {'Time':>8} {'Params':>10}")
    print("-" * 120)
    for i, r in enumerate(v1_sorted[:25], 1):
        time_str = f"{r['time_us']:.2f}" if r['time_us'] else "—"
        slope_str = f"{r['slope_mean']:.6f}" if r['slope_mean'] else "—"
        print(f"{i:<4} {r['name']:<30} {r['model_type']:<8} {r['hidden_dims']:<20} "
              f"{r['pos_mean_mm']:>10.4f} {r['pos_95_mm']:>10.4f} {slope_str:>10} "
              f"{time_str:>8} {r['n_params']:>10,}")
    
    # ---- V2 Summary ----
    print(f"\n{'='*120}")
    print("V2 RESULTS — Fixed dz=8000mm, 50M samples, PyTorch batched timing")
    print(f"{'='*120}")
    
    v2_sorted = sorted(v2, key=lambda x: x['pos_mean_mm'] if x['pos_mean_mm'] and x['pos_mean_mm'] < 1000 else 999999)
    
    print(f"\n{'Rank':<4} {'Model':<30} {'Type':<8} {'Dims':<20} {'Pos Mean':>10} {'Pos P90':>10} {'Slope':>10} {'Time':>8}")
    print("-" * 110)
    for i, r in enumerate(v2_sorted[:20], 1):
        pos = f"{r['pos_mean_mm']:.4f}" if r['pos_mean_mm'] and r['pos_mean_mm'] < 100 else f"{r['pos_mean_mm']:.1f}" if r['pos_mean_mm'] else "—"
        p90 = f"{r['pos_95_mm']:.4f}" if r['pos_95_mm'] and r['pos_95_mm'] < 100 else f"{r['pos_95_mm']:.1f}" if r['pos_95_mm'] else "—"
        slope_str = f"{r['slope_mean']:.6f}" if r['slope_mean'] and r['slope_mean'] < 1 else f"{r['slope_mean']:.2f}" if r['slope_mean'] else "—"
        time_str = f"{r['time_us']:.2f}" if r['time_us'] else "—"
        print(f"{i:<4} {r['name']:<30} {r['model_type']:<8} {r['hidden_dims']:<20} "
              f"{pos:>10} {p90:>10} {slope_str:>10} {time_str:>8}")
    
    # ---- V3 Summary ----
    print(f"\n{'='*120}")
    print("V3 RESULTS — Variable dz=[500,12000]mm, 100M samples, C++ single-sample timing")
    print(f"{'='*120}")
    
    print(f"\n{'Rank':<4} {'Model':<30} {'Type':<10} {'Pos MAE':>10} {'Pos RMSE':>10} {'Pos P95':>10} {'Slope MAE':>10} {'Slope RMSE':>10} {'Time(µs)':>10} {'Params':>10}")
    print("-" * 130)
    for i, r in enumerate(v3, 1):
        pos_mae = f"{r['pos_mean_mm']:.3f}" if r.get('pos_mean_mm') else "—"
        pos_rmse = f"{r.get('pos_rmse_mm', 0):.3f}" if r.get('pos_rmse_mm') else "—"
        p95 = f"{r.get('pos_95_mm', 0):.3f}" if r.get('pos_95_mm') else "—"
        slope_mae = f"{r.get('slope_mean', 0):.6f}" if r.get('slope_mean') else "—"
        slope_rmse = f"{r.get('slope_rmse', 0):.6f}" if r.get('slope_rmse') else "—"
        time_str = f"{r['time_us']:.2f}" if r.get('time_us') else "—"
        print(f"{i:<4} {r['name']:<30} {r['model_type']:<10} "
              f"{pos_mae:>10} {pos_rmse:>10} {p95:>10} {slope_mae:>10} {slope_rmse:>10} "
              f"{time_str:>10} {r['n_params']:>10,}")
    
    # ---- Cross-Version Best MLP Comparison ----
    print(f"\n{'='*120}")
    print("CROSS-VERSION COMPARISON — Best MLPs")
    print(f"{'='*120}")
    
    # Best per version
    best_v1_mlp = [r for r in v1_sorted if r['model_type'] == 'mlp'][:5]
    best_v2_mlp = [r for r in v2_sorted if r['model_type'] == 'mlp'][:5]
    best_v3_mlp = [r for r in v3 if r['model_type'] == 'MLP']
    
    print(f"\n{'Ver':<4} {'Model':<30} {'Dims':<20} {'Pos Mean/MAE':>12} {'Pos RMSE':>10} {'Pos P95':>10} {'Slope':>10} {'Time(µs)':>10} {'dz':>20}")
    print("-" * 130)
    
    for r in best_v1_mlp:
        time_str = f"{r['time_us']:.2f}" if r['time_us'] else "—"
        print(f"{'V1':<4} {r['name']:<30} {r['hidden_dims']:<20} "
              f"{r['pos_mean_mm']:>12.4f} {'—':>10} {r['pos_95_mm']:>10.4f} "
              f"{r['slope_mean']:>10.6f} {time_str:>10} {'fixed 8000':>20}")
    
    print()
    for r in best_v2_mlp:
        time_str = f"{r['time_us']:.2f}" if r['time_us'] else "—"
        slope_str = f"{r['slope_mean']:.6f}" if r['slope_mean'] else "—"
        print(f"{'V2':<4} {r['name']:<30} {r['hidden_dims']:<20} "
              f"{r['pos_mean_mm']:>12.4f} {'—':>10} {r.get('pos_95_mm', 0):>10.4f} "
              f"{slope_str:>10} {time_str:>10} {'fixed 8000':>20}")
    
    print()
    for r in best_v3_mlp:
        time_str = f"{r['time_us']:.2f}" if r.get('time_us') else "—"
        slope_str = f"{r.get('slope_mean', 0):.6f}" if r.get('slope_mean') else "—"
        rmse_str = f"{r.get('pos_rmse_mm', 0):.3f}" if r.get('pos_rmse_mm') else "—"
        p95_str = f"{r.get('pos_95_mm', 0):.3f}" if r.get('pos_95_mm') else "—"
        print(f"{'V3':<4} {r['name']:<30} {'—':<20} "
              f"{r.get('pos_mean_mm', 0):>12.3f} {rmse_str:>10} {p95_str:>10} "
              f"{slope_str:>10} {time_str:>10} {'var 500-12000':>20}")
    
    # ---- PINN Comparison ----
    print(f"\n{'='*120}")
    print("CROSS-VERSION COMPARISON — PINNs")
    print(f"{'='*120}")
    
    best_v1_pinn = [r for r in v1_sorted if 'pinn' in r['model_type']][:5]
    best_v2_pinn = [r for r in v2_sorted if 'pinn' in r['model_type']][:5]
    best_v3_pinn = [r for r in v3 if r['model_type'] == 'PINN']
    
    print(f"\n{'Ver':<4} {'Model':<35} {'Pos Mean/MAE':>12} {'Slope':>12} {'Time(µs)':>10} {'Notes':>30}")
    print("-" * 110)
    
    for r in best_v1_pinn:
        time_str = f"{r['time_us']:.2f}" if r['time_us'] else "—"
        print(f"{'V1':<4} {r['name']:<35} {r['pos_mean_mm']:>12.4f} "
              f"{r['slope_mean']:>12.6f} {time_str:>10} {'IC failure, no z_frac':>30}")
    
    print()
    for r in best_v2_pinn:
        time_str = f"{r['time_us']:.2f}" if r['time_us'] else "—"
        slope_str = f"{r['slope_mean']:.4f}" if r['slope_mean'] else "—"
        print(f"{'V2':<4} {r['name']:<35} {r['pos_mean_mm']:>12.1f} "
              f"{slope_str:>12} {time_str:>10} {'residual fix, still broken':>30}")
    
    print()
    for r in best_v3_pinn:
        time_str = f"{r['time_us']:.2f}" if r.get('time_us') else "—"
        slope_str = f"{r.get('slope_mean', 0):.6f}" if r.get('slope_mean') else "—"
        print(f"{'V3':<4} {r['name']:<35} {r.get('pos_mean_mm', 0):>12.1f} "
              f"{slope_str:>12} {time_str:>10} {'lin z_frac, great slopes':>30}")
    
    # ---- Key Patterns ----
    print(f"\n{'='*120}")
    print("KEY PATTERNS & OBSERVATIONS")
    print(f"{'='*120}")
    
    if best_v1_mlp and best_v2_mlp:
        # Find V2 single-layer models
        v2_single_512 = [r for r in v2_sorted if 'single_512' in r['name']]
        v2_single_str = f"{v2_single_512[0]['pos_mean_mm']:.4f}mm" if v2_single_512 else "N/A"
        
        v3_best_rmse = best_v3_mlp[0].get('pos_rmse_mm', 0) if best_v3_mlp else 0
        v3_best_name = best_v3_mlp[0]['name'] if best_v3_mlp else 'N/A'
        
        print(f"""
1. MLP ACCURACY PROGRESSION (fixed dz=8000mm → variable dz):
   V1 best: {best_v1_mlp[0]['name']:30s} → {best_v1_mlp[0]['pos_mean_mm']:.4f} mm (mean pos err)
   V2 best: {best_v2_mlp[0]['name']:30s} → {best_v2_mlp[0]['pos_mean_mm']:.4f} mm (mean pos err)
   V3 best: {v3_best_name:30s} → {v3_best_rmse:.3f} mm (pos RMSE, variable dz)
   
   V1→V2 improvement: {best_v1_mlp[0]['pos_mean_mm'] / best_v2_mlp[0]['pos_mean_mm']:.1f}× (shallow-wide architecture)
   V2→V3 regression:  {v3_best_rmse / best_v2_mlp[0]['pos_mean_mm']:.0f}× worse (variable dz is harder)

2. ARCHITECTURE DEPTH vs WIDTH:
   V1 narrow-deep vs wide: Check the table above for comparison.
   V2 single-layer [512] at {v2_single_str} vs 2-layer [512,256] at {best_v2_mlp[0]['pos_mean_mm']:.4f}mm

3. PINN STATUS:
   V1: IC failure — network ignores z_frac entirely
   V2: Residual fix — IC guaranteed, but 664-1567mm errors (catastrophic)
   V3: Supervised collocation — 49-58mm pos error, but EXCELLENT slopes (0.00025 RMSE)
   
   PINN slope RMSE vs MLP slope RMSE:
   V3 PINN: ~0.00025 (best slopes of ANY model)
   V3 MLP:  ~0.009-0.012 (36-48× worse than PINN)

4. TIMING COMPARISONS (CAUTION: different measurement methods):
   V1/V2: PyTorch batched inference (GPU)  → 0.8-5 µs
   V3: C++ single-sample inference (CPU)   → 65-466 µs
   These are NOT comparable. C++ single-sample is the deployment-relevant metric.
   V3 RK4 reference: 85.2 µs

5. NORMALIZATION ISSUES:
   V1/V2: input_std[dz] ≈ 1e-9 (all dz=8000) → cannot generalize
   V3: input_std[dz] ≈ 3300 (variable dz) → healthy normalization
""")
    
    # Save all results to CSV
    output_path = BASE / "V4" / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = v1 + v2 + v3
    csv_path = output_path / "cross_version_results.csv"
    
    if all_results:
        keys = set()
        for r in all_results:
            keys.update(r.keys())
        keys = sorted(keys)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nAll results saved to: {csv_path}")
        print(f"Total models: V1={len(v1)}, V2={len(v2)}, V3={len(v3)}")


if __name__ == '__main__':
    main()
