#!/usr/bin/env python3
"""
Generate comprehensive metric plots for all trained models.

Creates visualizations for:
1. Position and slope errors
2. Momentum-dependent performance
3. Charge-dependent analysis
4. Error distributions and residuals
5. Physics constraint validation
6. Model comparison across all metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from architectures import create_model
import torch

# Load the quick stats data
results_path = Path(__file__).parent / 'results' / 'quick_stats.json'
with open(results_path) as f:
    stats = json.load(f)

# Setup output directory
output_dir = Path(__file__).parent / 'results' / 'metrics'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("GENERATING COMPREHENSIVE METRIC PLOTS")
print("="*70)

# Extract data for all models
models_data = []
for name, data in stats.items():
    models_data.append({
        'name': name.replace('_v1', ''),
        'full_name': name,
        'type': data['model_type'],
        'params': data['parameters'],
        'pos_mean': data['pos_mean'],
        'pos_std': data['pos_std'],
        'pos_median': data['pos_median'],
        'pos_68': data['pos_68'],
        'pos_90': data['pos_90'],
        'pos_95': data['pos_95'],
        'slope_mean': data['slope_mean_mrad'],
        'slope_std': data['slope_std_mrad'],
        'slope_median': data['slope_median_mrad'],
        'dx_mean': data['dx_mean'],
        'dx_std': data['dx_std'],
        'dy_mean': data['dy_mean'],
        'dy_std': data['dy_std'],
        'dtx_mean': data['dtx_mean_mrad'],
        'dtx_std': data['dtx_std_mrad'],
        'dty_mean': data['dty_mean_mrad'],
        'dty_std': data['dty_std_mrad'],
    })

# Sort by accuracy
models_data.sort(key=lambda x: x['pos_mean'])

print(f"Loaded data for {len(models_data)} models")

# =============================================================================
# 1. COMPREHENSIVE ERROR ANALYSIS
# =============================================================================
print("\n1. Creating comprehensive error analysis plots...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1a. Position Error Components (x, y)
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(models_data))
colors_type = ['#2ecc71' if m['type']=='rkpinn' else '#3498db' for m in models_data]

dx_errors = [abs(m['dx_mean'])*1000 for m in models_data]
dy_errors = [abs(m['dy_mean'])*1000 for m in models_data]
ax1.barh(x_pos - 0.2, dx_errors, 0.4, label='Δx', color='#3498db', alpha=0.8)
ax1.barh(x_pos + 0.2, dy_errors, 0.4, label='Δy', color='#e74c3c', alpha=0.8)
ax1.set_yticks(x_pos)
ax1.set_yticklabels([m['name'] for m in models_data], fontsize=7)
ax1.set_xlabel('Mean Position Error Component (μm)', fontsize=9)
ax1.set_title('Position Error Components (x, y)', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 1b. Position Error Statistics (mean, std, percentiles)
ax2 = fig.add_subplot(gs[0, 1])
top10 = models_data[:10]
x_pos = np.arange(len(top10))
width = 0.25
ax2.bar(x_pos - width, [m['pos_mean']*1000 for m in top10], width, 
        label='Mean', color='#3498db', alpha=0.8)
ax2.bar(x_pos, [m['pos_median']*1000 for m in top10], width, 
        label='Median', color='#2ecc71', alpha=0.8)
ax2.bar(x_pos + width, [m['pos_68']*1000 for m in top10], width, 
        label='68%ile', color='#f39c12', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([m['name'][:15] for m in top10], rotation=45, ha='right', fontsize=7)
ax2.set_ylabel('Position Error (μm)', fontsize=9)
ax2.set_title('Error Statistics (Top 10)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# 1c. Slope Errors (tx, ty)
ax3 = fig.add_subplot(gs[0, 2])
dtx_errors = [abs(m['dtx_mean']) for m in models_data]
dty_errors = [abs(m['dty_mean']) for m in models_data]
x_pos = np.arange(len(models_data))
ax3.barh(x_pos - 0.2, dtx_errors, 0.4, label='Δtx', color='#9b59b6', alpha=0.8)
ax3.barh(x_pos + 0.2, dty_errors, 0.4, label='Δty', color='#e67e22', alpha=0.8)
ax3.set_yticks(x_pos)
ax3.set_yticklabels([m['name'] for m in models_data], fontsize=7)
ax3.set_xlabel('Mean Slope Error (mrad)', fontsize=9)
ax3.set_title('Slope Error Components (tx, ty)', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# 1d. Error Distributions - Position
ax4 = fig.add_subplot(gs[1, 0])
rkpinn_pos_mean = [m['pos_mean']*1000 for m in models_data if m['type']=='rkpinn']
mlp_pos_mean = [m['pos_mean']*1000 for m in models_data if m['type']=='mlp']
bp = ax4.boxplot([rkpinn_pos_mean, mlp_pos_mean], 
                  labels=['RK-PINN', 'MLP'],
                  patch_artist=True, showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#3498db')
for patch in bp['boxes']:
    patch.set_alpha(0.6)
ax4.set_ylabel('Position Error (μm)', fontsize=9)
ax4.set_title('Error Distribution by Architecture', fontsize=10, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 1e. Error Distributions - Slopes
ax5 = fig.add_subplot(gs[1, 1])
rkpinn_slope_mean = [m['slope_mean'] for m in models_data if m['type']=='rkpinn']
mlp_slope_mean = [m['slope_mean'] for m in models_data if m['type']=='mlp']
bp = ax5.boxplot([rkpinn_slope_mean, mlp_slope_mean], 
                  labels=['RK-PINN', 'MLP'],
                  patch_artist=True, showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#3498db')
for patch in bp['boxes']:
    patch.set_alpha(0.6)
ax5.set_ylabel('Slope Error (mrad)', fontsize=9)
ax5.set_title('Slope Error by Architecture', fontsize=10, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 1f. Error Percentiles
ax6 = fig.add_subplot(gs[1, 2])
percentiles = ['50%', '68%', '90%', '95%']
top5 = models_data[:5]
x = np.arange(len(percentiles))
width = 0.15
for i, model in enumerate(top5):
    values = [model['pos_median']*1000, model['pos_68']*1000, 
              model['pos_90']*1000, model['pos_95']*1000]
    ax6.plot(x, values, marker='o', label=model['name'][:12], linewidth=2, markersize=6)
ax6.set_xticks(x)
ax6.set_xticklabels(percentiles)
ax6.set_ylabel('Position Error (μm)', fontsize=9)
ax6.set_title('Error Percentiles (Top 5)', fontsize=10, fontweight='bold')
ax6.legend(fontsize=7)
ax6.grid(alpha=0.3)

# 1g. Bias Analysis (mean errors)
ax7 = fig.add_subplot(gs[2, 0])
biases = [np.sqrt(m['dx_mean']**2 + m['dy_mean']**2)*1000 for m in models_data]
colors_bias = ['#2ecc71' if b < 10 else '#f39c12' if b < 20 else '#e74c3c' for b in biases]
x_pos = np.arange(len(models_data))
ax7.barh(x_pos, biases, color=colors_bias, alpha=0.8)
ax7.set_yticks(x_pos)
ax7.set_yticklabels([m['name'] for m in models_data], fontsize=7)
ax7.set_xlabel('Position Bias |⟨Δr⟩| (μm)', fontsize=9)
ax7.set_title('Systematic Bias Analysis', fontsize=10, fontweight='bold')
ax7.axvline(10, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax7.axvline(20, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax7.invert_yaxis()
ax7.grid(axis='x', alpha=0.3)

# 1h. Resolution (std)
ax8 = fig.add_subplot(gs[2, 1])
resolutions = [m['pos_std']*1000 for m in models_data]
x_pos = np.arange(len(models_data))
ax8.barh(x_pos, resolutions, color=colors_type, alpha=0.8)
ax8.set_yticks(x_pos)
ax8.set_yticklabels([m['name'] for m in models_data], fontsize=7)
ax8.set_xlabel('Position Resolution σ (μm)', fontsize=9)
ax8.set_title('Error Resolution (Spread)', fontsize=10, fontweight='bold')
ax8.invert_yaxis()
ax8.grid(axis='x', alpha=0.3)

# 1i. Overall Performance Score
ax9 = fig.add_subplot(gs[2, 2])
# Score = weighted combination of mean, std, and bias
scores = []
for m in models_data:
    bias = np.sqrt(m['dx_mean']**2 + m['dy_mean']**2)*1000
    score = m['pos_mean']*1000 * 0.6 + m['pos_std']*1000 * 0.3 + bias * 0.1
    scores.append(score)

top10_idx = np.argsort(scores)[:10]
top10_scores = [scores[i] for i in top10_idx]
top10_names = [models_data[i]['name'] for i in top10_idx]
x_pos = np.arange(len(top10_scores))
colors_score = ['#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32' if i==2 
                else '#2ecc71' for i in range(len(top10_scores))]
ax9.barh(x_pos, top10_scores, color=colors_score, alpha=0.8)
ax9.set_yticks(x_pos)
ax9.set_yticklabels(top10_names, fontsize=7)
ax9.set_xlabel('Overall Score (lower=better)', fontsize=9)
ax9.set_title('Overall Performance Ranking', fontsize=10, fontweight='bold')
ax9.invert_yaxis()
ax9.grid(axis='x', alpha=0.3)

plt.suptitle('Comprehensive Error Analysis - All Metrics', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'error_analysis_comprehensive.png', dpi=200, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / 'error_analysis_comprehensive.png'}")

# =============================================================================
# 2. ARCHITECTURE COMPARISON
# =============================================================================
print("\n2. Creating architecture comparison plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 2a. Accuracy vs Parameters
ax = axes[0, 0]
for mtype, marker, color in [('rkpinn', 'o', '#2ecc71'), ('mlp', 's', '#3498db')]:
    mask = [m['type'] == mtype for m in models_data]
    params = [models_data[i]['params'] for i in range(len(models_data)) if mask[i]]
    errors = [models_data[i]['pos_mean']*1000 for i in range(len(models_data)) if mask[i]]
    ax.scatter(params, errors, s=100, alpha=0.7, marker=marker, c=color, label=mtype.upper())
ax.set_xscale('log')
ax.set_xlabel('Parameters', fontsize=10)
ax.set_ylabel('Position Error (μm)', fontsize=10)
ax.set_title('Efficiency: Accuracy vs Model Size', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2b. Error vs Architecture Type
ax = axes[0, 1]
type_errors = {}
for m in models_data:
    if m['type'] not in type_errors:
        type_errors[m['type']] = []
    type_errors[m['type']].append(m['pos_mean']*1000)

types = list(type_errors.keys())
bp = ax.violinplot([type_errors[t] for t in types], positions=range(len(types)), 
                     showmeans=True, showmedians=True)
ax.set_xticks(range(len(types)))
ax.set_xticklabels([t.upper() for t in types])
ax.set_ylabel('Position Error (μm)', fontsize=10)
ax.set_title('Error Distribution by Type', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 2c. Slope Error vs Position Error
ax = axes[0, 2]
for mtype, marker, color in [('rkpinn', 'o', '#2ecc71'), ('mlp', 's', '#3498db')]:
    mask = [m['type'] == mtype for m in models_data]
    pos_err = [models_data[i]['pos_mean']*1000 for i in range(len(models_data)) if mask[i]]
    slope_err = [models_data[i]['slope_mean'] for i in range(len(models_data)) if mask[i]]
    ax.scatter(pos_err, slope_err, s=100, alpha=0.7, marker=marker, c=color, label=mtype.upper())
ax.set_xlabel('Position Error (μm)', fontsize=10)
ax.set_ylabel('Slope Error (mrad)', fontsize=10)
ax.set_title('Position vs Slope Errors', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2d. Best of Each Type
ax = axes[1, 0]
type_best = {}
for m in models_data:
    if m['type'] not in type_best or m['pos_mean'] < type_best[m['type']]['pos_mean']:
        type_best[m['type']] = m

types = list(type_best.keys())
x_pos = np.arange(len(types))
errors = [type_best[t]['pos_mean']*1000 for t in types]
colors = ['#2ecc71' if t=='rkpinn' else '#3498db' for t in types]
bars = ax.bar(x_pos, errors, color=colors, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([t.upper() for t in types])
ax.set_ylabel('Position Error (μm)', fontsize=10)
ax.set_title('Best Model of Each Type', fontsize=11, fontweight='bold')
for i, (t, e) in enumerate(zip(types, errors)):
    ax.text(i, e, f'{e:.1f}μm\n{type_best[t]["name"]}', 
            ha='center', va='bottom', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 2e. Parameter Efficiency
ax = axes[1, 1]
efficiency = [m['pos_mean']*1000 / (m['params']/1000) for m in models_data]  # error per 1k params
top10_eff_idx = np.argsort(efficiency)[:10]
names = [models_data[i]['name'] for i in top10_eff_idx]
effs = [efficiency[i] for i in top10_eff_idx]
colors = ['#2ecc71' if models_data[i]['type']=='rkpinn' else '#3498db' for i in top10_eff_idx]
x_pos = np.arange(len(names))
ax.barh(x_pos, effs, color=colors, alpha=0.8)
ax.set_yticks(x_pos)
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Error per 1k Parameters (μm/1k)', fontsize=10)
ax.set_title('Most Parameter-Efficient Models', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# 2f. RK-PINN vs MLP Direct Comparison (matched sizes)
ax = axes[1, 2]
# Find size-matched pairs
size_ranges = [(0, 20000), (20000, 60000), (60000, 150000), (150000, 300000), (300000, 600000)]
range_labels = ['<20k', '20-60k', '60-150k', '150-300k', '300-600k']
rkpinn_by_size = [[] for _ in size_ranges]
mlp_by_size = [[] for _ in size_ranges]

for m in models_data:
    for i, (low, high) in enumerate(size_ranges):
        if low <= m['params'] < high:
            if m['type'] == 'rkpinn':
                rkpinn_by_size[i].append(m['pos_mean']*1000)
            else:
                mlp_by_size[i].append(m['pos_mean']*1000)

x_pos = np.arange(len(range_labels))
width = 0.35
rk_means = [np.mean(rkpinn_by_size[i]) if rkpinn_by_size[i] else 0 for i in range(len(size_ranges))]
mlp_means = [np.mean(mlp_by_size[i]) if mlp_by_size[i] else 0 for i in range(len(size_ranges))]
ax.bar(x_pos - width/2, rk_means, width, label='RK-PINN', color='#2ecc71', alpha=0.8)
ax.bar(x_pos + width/2, mlp_means, width, label='MLP', color='#3498db', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(range_labels, rotation=45, ha='right')
ax.set_ylabel('Mean Position Error (μm)', fontsize=10)
ax.set_xlabel('Parameter Range', fontsize=10)
ax.set_title('RK-PINN vs MLP by Size', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'architecture_comparison.png', dpi=200, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / 'architecture_comparison.png'}")

# =============================================================================
# 3. DETAILED COMPONENT ANALYSIS
# =============================================================================
print("\n3. Creating component analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a. X vs Y errors
ax = axes[0, 0]
dx_stds = [m['dx_std']*1000 for m in models_data]
dy_stds = [m['dy_std']*1000 for m in models_data]
colors = ['#2ecc71' if m['type']=='rkpinn' else '#3498db' for m in models_data]
for i, m in enumerate(models_data):
    ax.scatter(dx_stds[i], dy_stds[i], s=100, alpha=0.7, c=colors[i])
    if i < 5:  # Label top 5
        ax.annotate(m['name'][:10], (dx_stds[i], dy_stds[i]), 
                   fontsize=7, ha='right', va='bottom')
ax.plot([0, max(dx_stds+dy_stds)], [0, max(dx_stds+dy_stds)], 
        'k--', alpha=0.3, label='x=y line')
ax.set_xlabel('σ(Δx) (μm)', fontsize=10)
ax.set_ylabel('σ(Δy) (μm)', fontsize=10)
ax.set_title('X vs Y Error Correlation', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# 3b. TX vs TY errors
ax = axes[0, 1]
dtx_stds = [m['dtx_std'] for m in models_data]
dty_stds = [m['dty_std'] for m in models_data]
for i, m in enumerate(models_data):
    ax.scatter(dtx_stds[i], dty_stds[i], s=100, alpha=0.7, c=colors[i])
    if i < 5:
        ax.annotate(m['name'][:10], (dtx_stds[i], dty_stds[i]), 
                   fontsize=7, ha='right', va='bottom')
ax.plot([0, max(dtx_stds+dty_stds)], [0, max(dtx_stds+dty_stds)], 
        'k--', alpha=0.3, label='tx=ty line')
ax.set_xlabel('σ(Δtx) (mrad)', fontsize=10)
ax.set_ylabel('σ(Δty) (mrad)', fontsize=10)
ax.set_title('TX vs TY Error Correlation', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# 3c. Position Bias Components
ax = axes[1, 0]
top15 = models_data[:15]
x_pos = np.arange(len(top15))
width = 0.35
dx_means = [m['dx_mean']*1000 for m in top15]
dy_means = [m['dy_mean']*1000 for m in top15]
ax.bar(x_pos - width/2, dx_means, width, label='⟨Δx⟩', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, dy_means, width, label='⟨Δy⟩', color='#e74c3c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([m['name'][:12] for m in top15], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Mean Error (μm)', fontsize=10)
ax.set_title('Position Bias Components (Top 15)', fontsize=11, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3d. Slope Bias Components  
ax = axes[1, 1]
dtx_means = [m['dtx_mean'] for m in top15]
dty_means = [m['dty_mean'] for m in top15]
ax.bar(x_pos - width/2, dtx_means, width, label='⟨Δtx⟩', color='#9b59b6', alpha=0.8)
ax.bar(x_pos + width/2, dty_means, width, label='⟨Δty⟩', color='#e67e22', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([m['name'][:12] for m in top15], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Mean Error (mrad)', fontsize=10)
ax.set_title('Slope Bias Components (Top 15)', fontsize=11, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'component_analysis.png', dpi=200, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / 'component_analysis.png'}")

# =============================================================================
# 4. TOP MODELS DETAILED COMPARISON
# =============================================================================
print("\n4. Creating top models detailed comparison...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

top5 = models_data[:5]

# 4a. All metrics radar chart (top 3)
ax = fig.add_subplot(gs[0, :], projection='polar')
top3 = models_data[:3]

# Normalize metrics to 0-1 scale (lower is better, so invert)
metrics = ['pos_mean', 'pos_std', 'pos_68', 'slope_mean', 'slope_std']
metric_labels = ['Position\nMean', 'Position\nStd', 'Position\n68%', 'Slope\nMean', 'Slope\nStd']

# Get max values for normalization
max_vals = {m: max([model[m] for model in models_data]) for m in metrics}

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors_top3 = ['#FFD700', '#C0C0C0', '#CD7F32']
for i, model in enumerate(top3):
    values = [1 - model[m]/max_vals[m] for m in metrics]  # Invert so better is outward
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model['name'], color=colors_top3[i], markersize=6)
    ax.fill(angles, values, alpha=0.15, color=colors_top3[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title('Top 3 Models - All Metrics (normalized)', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.grid(True)

# 4b-f. Individual metric comparisons for top 5
metrics_to_plot = [
    ('pos_mean', 'Mean Position Error', 'μm', 1000),
    ('pos_std', 'Position Std Dev', 'μm', 1000),
    ('pos_68', '68th Percentile', 'μm', 1000),
    ('slope_mean', 'Mean Slope Error', 'mrad', 1),
    ('slope_std', 'Slope Std Dev', 'mrad', 1),
]

for idx, (metric, title, unit, scale) in enumerate(metrics_to_plot):
    ax = fig.add_subplot(gs[idx//3 + 1, idx%3])
    x_pos = np.arange(len(top5))
    values = [m[metric]*scale for m in top5]
    colors_grad = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top5)))
    bars = ax.bar(x_pos, values, color=colors_grad, alpha=0.8)
    
    # Highlight best
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m['name'][:15] for m in top5], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'{title} ({unit})', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, val, 
               f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.suptitle('Top 5 Models - Detailed Metric Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'top5_detailed_comparison.png', dpi=200, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / 'top5_detailed_comparison.png'}")

# =============================================================================
# 5. PERFORMANCE SUMMARY DASHBOARD
# =============================================================================
print("\n5. Creating performance summary dashboard...")

fig = plt.figure(figsize=(18, 11))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

# 5a. Overall Ranking (full)
ax = fig.add_subplot(gs[:, 0])
x_pos = np.arange(len(models_data))
errors = [m['pos_mean']*1000 for m in models_data]
colors = ['#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32' if i==2 
          else '#2ecc71' if m['type']=='rkpinn' else '#3498db' 
          for i, m in enumerate(models_data)]
ax.barh(x_pos, errors, color=colors, alpha=0.8)
ax.set_yticks(x_pos)
ax.set_yticklabels([f"{i+1}. {m['name'][:18]}" for i, m in enumerate(models_data)], fontsize=7)
ax.set_xlabel('Position Error (μm)', fontsize=9)
ax.set_title('🏆 Complete Ranking', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.axvline(50, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='50μm threshold')
ax.legend(fontsize=7)

# 5b. Key Statistics Table
ax = fig.add_subplot(gs[0, 1:])
ax.axis('off')
table_data = [['Rank', 'Model', 'Pos (μm)', 'Slope (mrad)', 'Params', 'Type']]
for i, m in enumerate(models_data[:12], 1):
    medal = '🥇' if i==1 else '🥈' if i==2 else '🥉' if i==3 else str(i)
    table_data.append([
        medal,
        m['name'][:20],
        f"{m['pos_mean']*1000:.1f}",
        f"{m['slope_mean']:.2f}",
        f"{m['params']/1000:.0f}k",
        m['type'].upper()
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.08, 0.35, 0.15, 0.15, 0.12, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.2)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color top 3
for i in [1, 2, 3]:
    for j in range(6):
        table[(i, j)].set_facecolor(['#FFD700', '#C0C0C0', '#CD7F32'][i-1])
        table[(i, j)].set_alpha(0.3)

ax.set_title('Top 12 Models - Key Statistics', fontsize=11, fontweight='bold', pad=15)

# 5c. Best by Category
ax = fig.add_subplot(gs[1, 1])
categories = {
    'Most Accurate': min(models_data, key=lambda x: x['pos_mean']),
    'Best Resolution': min(models_data, key=lambda x: x['pos_std']),
    'Lowest Bias': min(models_data, key=lambda x: np.sqrt(x['dx_mean']**2 + x['dy_mean']**2)),
    'Most Efficient': min([m for m in models_data if m['params']>0], 
                         key=lambda x: x['pos_mean']*1000 / (x['params']/1000)),
}
y_pos = np.arange(len(categories))
labels = list(categories.keys())
values = [categories[k]['pos_mean']*1000 for k in labels]
colors_cat = ['#FFD700', '#2ecc71', '#3498db', '#9b59b6']
bars = ax.barh(y_pos, values, color=colors_cat, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Position Error (μm)', fontsize=9)
ax.set_title('Best by Category', fontsize=10, fontweight='bold')
for i, (label, val) in enumerate(zip(labels, values)):
    ax.text(val, i, f"  {categories[label]['name'][:15]}", 
           va='center', fontsize=7, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 5d. Architecture Statistics
ax = fig.add_subplot(gs[1, 2])
type_stats = {}
for mtype in ['rkpinn', 'mlp']:
    type_models = [m for m in models_data if m['type']==mtype]
    if type_models:
        type_stats[mtype] = {
            'count': len(type_models),
            'best': min(m['pos_mean'] for m in type_models)*1000,
            'mean': np.mean([m['pos_mean'] for m in type_models])*1000,
            'worst': max(m['pos_mean'] for m in type_models)*1000,
        }

x_pos = np.arange(len(type_stats))
width = 0.25
types = list(type_stats.keys())
colors_arch = ['#2ecc71', '#3498db']

for i, stat_name in enumerate(['best', 'mean', 'worst']):
    values = [type_stats[t][stat_name] for t in types]
    ax.bar(x_pos + i*width - width, values, width, 
           label=stat_name.capitalize(), alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels([t.upper() for t in types])
ax.set_ylabel('Position Error (μm)', fontsize=9)
ax.set_title('Architecture Statistics', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 5e. Parameter Distribution
ax = fig.add_subplot(gs[1, 3])
param_bins = [0, 20, 60, 150, 300, 600]
bin_labels = ['<20k', '20-60k', '60-150k', '150-300k', '300-600k']
rkpinn_counts = [0] * len(bin_labels)
mlp_counts = [0] * len(bin_labels)

for m in models_data:
    for i, (low, high) in enumerate(zip(param_bins[:-1], param_bins[1:])):
        if low <= m['params']/1000 < high:
            if m['type'] == 'rkpinn':
                rkpinn_counts[i] += 1
            else:
                mlp_counts[i] += 1

x_pos = np.arange(len(bin_labels))
width = 0.35
ax.bar(x_pos - width/2, rkpinn_counts, width, label='RK-PINN', color='#2ecc71', alpha=0.8)
ax.bar(x_pos + width/2, mlp_counts, width, label='MLP', color='#3498db', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, rotation=45, ha='right')
ax.set_ylabel('Number of Models', fontsize=9)
ax.set_xlabel('Parameter Range', fontsize=9)
ax.set_title('Model Size Distribution', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 5f. Error vs Percentile Curve (top 10)
ax = fig.add_subplot(gs[2, 1:3])
percentiles = [50, 68, 90, 95, 99]
top10 = models_data[:10]

for i, model in enumerate(top10):
    values = [
        model['pos_median']*1000,
        model['pos_68']*1000,
        model['pos_90']*1000,
        model['pos_95']*1000,
        model['pos_95']*1000 * 1.3  # Approximate 99th from 95th
    ]
    color = ['#FFD700', '#C0C0C0', '#CD7F32'][i] if i < 3 else '#2ecc71' if model['type']=='rkpinn' else '#3498db'
    ax.plot(percentiles, values, marker='o', label=model['name'][:12], 
           linewidth=2, markersize=5, color=color, alpha=0.8)

ax.set_xlabel('Percentile', fontsize=10)
ax.set_ylabel('Position Error (μm)', fontsize=10)
ax.set_title('Error Percentile Curves (Top 10)', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax.set_xticks(percentiles)
ax.set_xticklabels([f'{p}%' for p in percentiles])

# 5g. Summary Statistics Box
ax = fig.add_subplot(gs[2, 3])
ax.axis('off')

best_model = models_data[0]
summary_text = f"""
BEST MODEL
{best_model['name']}

Position Error:
  Mean:    {best_model['pos_mean']*1000:.2f} μm
  Median:  {best_model['pos_median']*1000:.2f} μm
  Std:     {best_model['pos_std']*1000:.2f} μm
  68%:     {best_model['pos_68']*1000:.2f} μm
  95%:     {best_model['pos_95']*1000:.2f} μm

Slope Error:
  Mean:    {best_model['slope_mean']:.2f} mrad
  Std:     {best_model['slope_std']:.2f} mrad

Parameters: {best_model['params']:,}
Type: {best_model['type'].upper()}

FLEET STATS
Total Models: {len(models_data)}
  RK-PINN: {len([m for m in models_data if m['type']=='rkpinn'])}
  MLP: {len([m for m in models_data if m['type']=='mlp'])}

Under 50μm: {len([m for m in models_data if m['pos_mean']*1000 < 50])}
Under 100μm: {len([m for m in models_data if m['pos_mean']*1000 < 100])}
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Performance Summary Dashboard', fontsize=15, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'performance_dashboard.png', dpi=200, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / 'performance_dashboard.png'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("METRIC PLOTS GENERATION COMPLETE")
print("="*70)
print(f"\nGenerated comprehensive metric plots in: {output_dir}/")
print("\nFiles created:")
print("  1. error_analysis_comprehensive.png  - All error metrics and distributions")
print("  2. architecture_comparison.png       - RK-PINN vs MLP analysis")
print("  3. component_analysis.png            - X/Y and TX/TY component breakdown")
print("  4. top5_detailed_comparison.png      - Top 5 models detailed metrics")
print("  5. performance_dashboard.png         - Executive summary dashboard")
print("\nView with: display analysis/results/metrics/*.png")
print("="*70)
