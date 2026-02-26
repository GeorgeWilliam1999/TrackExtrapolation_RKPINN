#!/usr/bin/env python3
"""
V5 Results Collector
Scans V5/trained_models/ for completed training runs and collects metrics
into a summary CSV for easy comparison.

Usage:
    cd experiments/next_generation/
    python V5/analysis/collect_results.py
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE = Path("/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation")
MODELS_DIR = BASE / "V5" / "trained_models"
OUTPUT_CSV = BASE / "V5" / "analysis" / "v5_results.csv"


def count_parameters(model_path: Path) -> int:
    """Count model parameters from checkpoint."""
    if not model_path.exists() or not HAS_TORCH:
        return 0
    try:
        cp = torch.load(model_path, map_location='cpu', weights_only=False)
        state = cp.get('model_state_dict', cp)
        return sum(v.numel() for v in state.values())
    except Exception:
        return 0


def load_history(model_dir: Path) -> Optional[Dict]:
    """Load training history from JSON."""
    path = model_dir / 'history.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config(model_dir: Path) -> Optional[Dict]:
    """Load config from saved copy in model directory."""
    path = model_dir / 'config.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract key metrics from a single trained model."""
    config = load_config(model_dir)
    history = load_history(model_dir)

    if config is None:
        return None

    model_type = config.get('model', {}).get('type', 'unknown')
    hidden_dims = config.get('model', {}).get('hidden_dims', [])
    activation = config.get('model', {}).get('activation', 'silu')

    result = {
        'name': model_dir.name,
        'model_type': model_type,
        'hidden_dims': str(hidden_dims),
        'activation': activation,
        'description': config.get('description', ''),
        'has_best_model': (model_dir / 'best_model.pt').exists(),
        'has_final_model': (model_dir / 'final_model.pt').exists(),
        'n_parameters': count_parameters(model_dir / 'best_model.pt'),
    }

    # Model-specific fields
    if model_type == 'pde':
        result['pde_mode'] = config.get('pde_mode', 'pure')
    elif model_type == 'compositional':
        result['n_steps'] = config.get('model', {}).get('n_steps', 8)

    # Training metrics from history
    if history:
        train_hist = history.get('train', [])
        val_hist = history.get('val', [])

        if val_hist:
            # Best val loss
            best_val = min(val_hist, key=lambda x: x.get('total', float('inf')))
            best_epoch = val_hist.index(best_val) + 1
            result['best_val_loss'] = best_val.get('total', None)
            result['best_epoch'] = best_epoch
            result['total_epochs'] = len(val_hist)

            # Final val loss
            result['final_val_loss'] = val_hist[-1].get('total', None)

            # Component losses at best epoch (if available)
            for key in ['ic', 'endpoint', 'collocation', 'pde']:
                if key in best_val:
                    result[f'best_val_{key}'] = best_val[key]

        if train_hist:
            result['final_train_loss'] = train_hist[-1].get('total', None)

    # From best_model checkpoint
    best_path = model_dir / 'best_model.pt'
    if best_path.exists() and HAS_TORCH:
        try:
            cp = torch.load(best_path, map_location='cpu', weights_only=False)
            result['checkpoint_val_loss'] = cp.get('val_loss', None)
            result['checkpoint_epoch'] = cp.get('epoch', None)
        except Exception:
            pass

    return result


def main():
    print("=" * 60)
    print("V5 Results Collection")
    print("=" * 60)

    if not MODELS_DIR.exists():
        print(f"No trained models directory: {MODELS_DIR}")
        return

    model_dirs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir()])

    if not model_dirs:
        print(f"No model directories found in {MODELS_DIR}")
        return

    print(f"\nFound {len(model_dirs)} model directories:")
    results = []

    for model_dir in model_dirs:
        metrics = extract_metrics(model_dir)
        if metrics:
            status = "trained" if metrics['has_best_model'] else "incomplete"
            val_loss = metrics.get('best_val_loss', 'N/A')
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.6f}"
            print(f"  {metrics['name']:40s} {status:12s} val_loss={val_loss}")
            results.append(metrics)
        else:
            print(f"  {model_dir.name:40s} no config found")

    if not results:
        print("\nNo results to save.")
        return

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Gather all keys across results
    all_keys = []
    for r in results:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("V5 Model Comparison")
    print("=" * 60)
    print(f"{'Name':40s} {'Type':15s} {'Val Loss':>12s} {'Epoch':>6s} {'Params':>10s}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: x.get('best_val_loss', float('inf'))):
        val = r.get('best_val_loss', None)
        val_str = f"{val:.6f}" if val is not None else "N/A"
        epoch_str = str(r.get('best_epoch', '')) if r.get('best_epoch') else "N/A"
        params = r.get('n_parameters', 0)
        params_str = f"{params:,}" if params else "N/A"
        print(f"  {r['name']:40s} {r['model_type']:15s} {val_str:>12s} {epoch_str:>6s} {params_str:>10s}")

    print("=" * 60)


if __name__ == '__main__':
    main()
