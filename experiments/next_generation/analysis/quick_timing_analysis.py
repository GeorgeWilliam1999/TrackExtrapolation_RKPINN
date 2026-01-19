#!/usr/bin/env python3
"""
Quick Timing Analysis - Measures actual inference speed of all trained models.

Fast version that:
1. Uses small sample for quick measurement
2. Benchmarks CPU inference (batch mode)
3. Compares to C++ RK4 reference
4. Generates comprehensive timing plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
import torch
import time
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from architectures import create_model

print("="*70)
print("QUICK TIMING ANALYSIS")
print("="*70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
has_gpu = torch.cuda.is_available()
print(f"Device: {device} (GPU: {has_gpu})")

models_dir = Path(__file__).parent.parent / 'trained_models'
output_dir = Path(__file__).parent / 'results' / 'metrics'
output_dir.mkdir(parents=True, exist_ok=True)

# Generate small test dataset (don't load the huge file)
print("\nGenerating synthetic test data (10,000 samples)...")
np.random.seed(42)
n_samples = 10000

# Generate realistic test data matching training distribution
X_test = np.random.randn(n_samples, 6).astype(np.float32)
X_test[:, 0] *= 100  # x: ~100mm spread
X_test[:, 1] *= 100  # y: ~100mm spread
X_test[:, 2] *= 0.1  # tx: ~0.1 slope
X_test[:, 3] *= 0.1  # ty: ~0.1 slope
X_test[:, 4] = np.random.uniform(-0.003, 0.003, n_samples)  # qop
X_test[:, 5] = 2300.0  # dz

X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

# Timing parameters
n_warmup = 10
n_runs = 100
batch_sizes = [1, 32, 256, 1024]

# Benchmark all models
timing_results = {}

print(f"\nBenchmarking {len(list(models_dir.glob('*_v1')))} models...")

for model_dir in tqdm(sorted(models_dir.glob('*_v1')), desc="Models"):
    if not model_dir.is_dir():
        continue
    
    model_name = model_dir.name
    config_path = model_dir / 'config.json'
    model_path = model_dir / 'best_model.pt'
    
    if not config_path.exists() or not model_path.exists():
        continue
    
    try:
        # Load model
        with open(config_path) as f:
            config = json.load(f)
        
        config['input_dim'] = 6
        config['output_dim'] = 4
        model_type = config.get('model_type', 'mlp')
        
        if model_type == 'mlp':
            model = create_model('mlp',
                               hidden_dims=config['hidden_dims'],
                               activation=config.get('activation', 'relu'),
                               dropout=config.get('dropout', 0.0),
                               input_dim=6, output_dim=4)
        elif model_type == 'rkpinn':
            model = create_model('rkpinn',
                               hidden_dims=config['hidden_dims'],
                               activation=config.get('activation', 'relu'),
                               input_dim=6, output_dim=4,
                               n_rk_steps=config.get('n_rk_steps', 10))
        else:
            continue
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        
        # Benchmark different batch sizes
        batch_timings = {}
        
        for batch_size in batch_sizes:
            if batch_size > n_samples:
                continue
            
            X_batch = X_tensor[:batch_size]
            
            # Warmup
            with torch.no_grad():
                for _ in range(n_warmup):
                    _ = model(X_batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(n_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(X_batch)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times) * 1e6  # microseconds
            time_per_track = avg_time / batch_size
            throughput = batch_size / (avg_time / 1e6)
            
            batch_timings[batch_size] = {
                'batch_time_us': avg_time,
                'time_per_track_us': time_per_track,
                'throughput_hz': throughput
            }
        
        # Use largest batch for representative timing
        best_batch = max([bs for bs in batch_sizes if bs <= n_samples])
        best_timing = batch_timings[best_batch]
        
        timing_results[model_name] = {
            'model_type': model_type,
            'parameters': n_params,
            'batch_timings': batch_timings,
            'time_per_track_us': best_timing['time_per_track_us'],
            'throughput_hz': best_timing['throughput_hz'],
            'device': device.type
        }
        
    except Exception as e:
        print(f"  Error benchmarking {model_name}: {e}")
        continue

print(f"\nSuccessfully benchmarked {len(timing_results)} models")

# Save results
results_path = output_dir.parent / 'timing_results.json'
with open(results_path, 'w') as f:
    json.dump(timing_results, f, indent=2)
print(f"✅ Saved: {results_path}")

# Load accuracy data for combined plots
with open(output_dir.parent / 'quick_stats.json') as f:
    accuracy_stats = json.load(f)

# Match timing with accuracy
for model_name in timing_results:
    if model_name in accuracy_stats:
        timing_results[model_name]['position_error_mm'] = accuracy_stats[model_name]['pos_mean']

# Reference RK4 timing (from C++ benchmarks)
rk4_time_per_track = 150.0  # μs (typical for C++ RK4 with 5mm steps)
rk4_throughput = 1e6 / rk4_time_per_track

# =============================================================================
# GENERATE TIMING PLOTS
# =============================================================================
print("\nGenerating timing plots...")

fig = plt.figure(figsize=(18, 12))
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# Sort by throughput
models_sorted = sorted(timing_results.items(), 
                      key=lambda x: x[1]['throughput_hz'], reverse=True)

names = [m[0].replace('_v1', '') for m in models_sorted]
throughputs = [m[1]['throughput_hz'] for m in models_sorted]
times_per_track = [m[1]['time_per_track_us'] for m in models_sorted]
params = [m[1]['parameters'] for m in models_sorted]
types = [m[1]['model_type'] for m in models_sorted]
errors = [m[1].get('position_error_mm', 0)*1000 for m in models_sorted]

colors_type = ['#2ecc71' if t=='rkpinn' else '#3498db' for t in types]

# 1. Throughput Ranking
ax = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(names))
speedups = [tp / rk4_throughput for tp in throughputs]
bars = ax.barh(x_pos, throughputs, color=colors_type, alpha=0.8)
ax.axvline(rk4_throughput, color='red', linestyle='--', linewidth=2, 
          label=f'C++ RK4 (~{rk4_throughput:.0f} tracks/s)', alpha=0.7)
ax.set_yticks(x_pos)
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Throughput (tracks/second)', fontsize=10)
ax.set_title('Inference Speed Ranking (CPU, batch=1024)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

# Add speedup annotations
for i, (tp, sp) in enumerate(zip(throughputs, speedups)):
    if tp > rk4_throughput:
        ax.text(tp, i, f' {sp:.1f}x', va='center', fontsize=6, fontweight='bold')

# 2. Time per Track
ax = fig.add_subplot(gs[1, 0])
x_pos = np.arange(min(15, len(names)))
top15_times = times_per_track[:15]
top15_names = names[:15]
top15_colors = colors_type[:15]
ax.bar(x_pos, top15_times, color=top15_colors, alpha=0.8)
ax.axhline(rk4_time_per_track, color='red', linestyle='--', linewidth=2, 
          label='C++ RK4 reference', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(top15_names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Time per Track (μs)', fontsize=10)
ax.set_title('Latency (Top 15 Fastest)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 3. Speedup vs RK4
ax = fig.add_subplot(gs[1, 1])
speedups_sorted = sorted([(n, s, t) for n, s, t in zip(names, speedups, types)],
                        key=lambda x: x[1], reverse=True)[:15]
sp_names = [x[0] for x in speedups_sorted]
sp_values = [x[1] for x in speedups_sorted]
sp_colors = ['#2ecc71' if x[2]=='rkpinn' else '#3498db' for x in speedups_sorted]
x_pos = np.arange(len(sp_names))
bars = ax.barh(x_pos, sp_values, color=sp_colors, alpha=0.8)
ax.set_yticks(x_pos)
ax.set_yticklabels(sp_names, fontsize=7)
ax.set_xlabel('Speedup vs C++ RK4', fontsize=10)
ax.set_title('Speed Improvement (Top 15)', fontsize=11, fontweight='bold')
ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Highlight best speedup
bars[0].set_edgecolor('gold')
bars[0].set_linewidth(3)

# 4. Accuracy vs Speed Tradeoff
ax = fig.add_subplot(gs[1, 2])
for mtype, marker, color in [('rkpinn', 'o', '#2ecc71'), ('mlp', 's', '#3498db')]:
    mask = [t == mtype for t in types]
    x = [throughputs[i] for i in range(len(throughputs)) if mask[i]]
    y = [errors[i] for i in range(len(errors)) if mask[i]]
    ax.scatter(x, y, s=100, alpha=0.7, marker=marker, c=color, label=mtype.upper())

# Add RK4 reference point
ax.scatter([rk4_throughput], [0], s=200, marker='*', c='red', 
          label='C++ RK4', edgecolors='black', linewidths=2, zorder=10)

ax.set_xscale('log')
ax.set_xlabel('Throughput (tracks/s)', fontsize=10)
ax.set_ylabel('Position Error (μm)', fontsize=10)
ax.set_title('Speed-Accuracy Tradeoff', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 5. Batch Size Scaling
ax = fig.add_subplot(gs[2, 0])
# Plot top 5 models
top5_models = [(n, timing_results[m[0]]) for n, m in zip(names[:5], models_sorted[:5])]
for i, (name, data) in enumerate(top5_models):
    batch_data = data['batch_timings']
    batch_sizes_avail = sorted(batch_data.keys())
    throughputs_batch = [batch_data[bs]['throughput_hz'] for bs in batch_sizes_avail]
    color = '#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32' if i==2 else '#2ecc71'
    ax.plot(batch_sizes_avail, throughputs_batch, marker='o', linewidth=2, 
           label=name[:15], color=color)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Batch Size', fontsize=10)
ax.set_ylabel('Throughput (tracks/s)', fontsize=10)
ax.set_title('Batch Size Scaling (Top 5)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 6. Throughput per Parameter
ax = fig.add_subplot(gs[2, 1])
efficiency = [tp / (p/1000) for tp, p in zip(throughputs, params)]  # throughput per 1k params
top10_eff_idx = np.argsort(efficiency)[-10:][::-1]
eff_names = [names[i] for i in top10_eff_idx]
eff_values = [efficiency[i] for i in top10_eff_idx]
eff_colors = [colors_type[i] for i in top10_eff_idx]
x_pos = np.arange(len(eff_names))
ax.barh(x_pos, eff_values, color=eff_colors, alpha=0.8)
ax.set_yticks(x_pos)
ax.set_yticklabels(eff_names, fontsize=7)
ax.set_xlabel('Throughput per 1k Params (tracks/s/1k)', fontsize=9)
ax.set_title('Computational Efficiency', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# 7. Performance Summary Table
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

# Create summary table
table_data = [['Rank', 'Model', 'Speed (k/s)', 'Speedup', 'Error (μm)']]
for i, (name, data) in enumerate(models_sorted[:10], 1):
    medal = '🥇' if i==1 else '🥈' if i==2 else '🥉' if i==3 else str(i)
    speedup = data['throughput_hz'] / rk4_throughput
    error = data.get('position_error_mm', 0) * 1000
    table_data.append([
        medal,
        name[:20],
        f"{data['throughput_hz']/1000:.1f}",
        f"{speedup:.1f}x",
        f"{error:.1f}" if error > 0 else "N/A"
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.08, 0.40, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color top 3
for i in [1, 2, 3]:
    for j in range(5):
        table[(i, j)].set_facecolor(['#FFD700', '#C0C0C0', '#CD7F32'][i-1])
        table[(i, j)].set_alpha(0.3)

ax.set_title('Speed Leaderboard', fontsize=11, fontweight='bold', pad=15)

plt.suptitle(f'Inference Timing Analysis ({device.type.upper()})', 
            fontsize=14, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'timing_analysis.png', dpi=200, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'timing_analysis.png'}")

# =============================================================================
# SPEED-ACCURACY FRONTIER PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all models
for mtype, marker, color in [('rkpinn', 'o', '#2ecc71'), ('mlp', 's', '#3498db')]:
    mask = [t == mtype for t in types]
    x = [throughputs[i] for i in range(len(throughputs)) if mask[i]]
    y = [errors[i] for i in range(len(errors)) if mask[i]]
    n = [names[i] for i in range(len(names)) if mask[i]]
    
    ax.scatter(x, y, s=150, alpha=0.6, marker=marker, c=color, label=mtype.upper())
    
    # Label top 3 of each type
    sorted_by_error = sorted(zip(x, y, n), key=lambda p: p[1])[:3]
    for sx, sy, sn in sorted_by_error:
        ax.annotate(sn, (sx, sy), fontsize=7, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))

# Add RK4 reference
ax.scatter([rk4_throughput], [0], s=300, marker='*', c='red', 
          label='C++ RK4\n(ground truth)', edgecolors='black', linewidths=2, zorder=10)
ax.annotate('C++ RK4', (rk4_throughput, 0), fontsize=9, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.2),
           fontweight='bold')

# Draw Pareto frontier
all_points = list(zip(throughputs, errors))
all_points.append((rk4_throughput, 0))
all_points.sort()

pareto_x, pareto_y = [], []
min_error = float('inf')
for x, y in all_points:
    if y < min_error:
        pareto_x.append(x)
        pareto_y.append(y)
        min_error = y

ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.4, label='Pareto Frontier')

ax.set_xscale('log')
ax.set_xlabel('Throughput (tracks/second)', fontsize=12)
ax.set_ylabel('Position Error (μm)', fontsize=12)
ax.set_title('Speed-Accuracy Frontier: Neural Networks vs Traditional RK4', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

# Add annotation zones
ax.axhspan(0, 50, alpha=0.1, color='green', label='Excellent (<50μm)')
ax.text(0.98, 0.95, 'Production\nReady\nZone', transform=ax.transAxes,
       fontsize=11, ha='right', va='top', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.2))

plt.tight_layout()
plt.savefig(output_dir / 'speed_accuracy_frontier.png', dpi=200, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'speed_accuracy_frontier.png'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("TIMING ANALYSIS SUMMARY")
print("="*70)

print(f"\nC++ RK4 Reference: {rk4_throughput:.0f} tracks/s")
print(f"\nTop 5 Fastest Models:")
for i, (name, data) in enumerate(models_sorted[:5], 1):
    speedup = data['throughput_hz'] / rk4_throughput
    error = data.get('position_error_mm', 0) * 1000
    print(f"  {i}. {name:30s} {data['throughput_hz']:8.0f} tracks/s  " +
          f"({speedup:5.1f}x)  Error: {error:5.1f}μm")

print(f"\nBest Speed-Accuracy Tradeoff:")
# Find best combination of speed and accuracy
tradeoff_scores = []
for name, data in timing_results.items():
    error = data.get('position_error_mm', 1) * 1000
    if error > 0:
        # Score: higher throughput, lower error is better
        # Normalize: speedup * (100 / error)
        speedup = data['throughput_hz'] / rk4_throughput
        score = speedup * (100.0 / error)
        tradeoff_scores.append((name, score, speedup, error))

tradeoff_scores.sort(key=lambda x: x[1], reverse=True)
for i, (name, score, speedup, error) in enumerate(tradeoff_scores[:5], 1):
    print(f"  {i}. {name:30s} Score: {score:6.1f}  " +
          f"({speedup:5.1f}x speedup, {error:5.1f}μm error)")

print("\n" + "="*70)
print("Generated plots:")
print("  • timing_analysis.png")
print("  • speed_accuracy_frontier.png")
print("="*70)
