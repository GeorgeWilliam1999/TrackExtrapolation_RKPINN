#!/usr/bin/env python3
"""
Parse C++ benchmark results and compute statistics.

Reads ROOT ntuple from TrackExtrapolatorTesterSOA and extracts:
- Timing statistics (mean, median, P90, P95, P99)
- Accuracy metrics (position errors, slope errors)
- Throughput calculations

Outputs:
- benchmark_results.json (full statistics)
- benchmark_summary.csv (quick reference)
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def parse_root_file(root_file: str) -> Dict:
    """
    Parse ROOT file from TrackExtrapolatorTesterSOA.
    
    Returns:
        Dictionary with statistics for each extrapolator
    """
    try:
        import uproot
        import awkward as ak
    except ImportError:
        print("ERROR: uproot not installed. Install with:")
        print("  pip install uproot awkward")
        print("\nAlternatively, install PyROOT from conda-forge")
        return None
    
    results = {}
    
    with uproot.open(root_file) as f:
        # Get list of trees (one per extrapolator)
        tree_names = [key.split(';')[0] for key in f.keys() if 'TNtuple' in str(f[key])]
        
        print(f"\nFound {len(tree_names)} extrapolators in ROOT file")
        
        for tree_name in tree_names:
            tree = f[tree_name]
            
            # Extract data
            data = tree.arrays(library="pd")
            
            # Compute timing statistics (convert ns to μs)
            times_us = data['time'] / 1000.0
            
            # Compute accuracy statistics (mm)
            pos_errors = np.sqrt(data['dx']**2 + data['dy']**2)
            
            results[tree_name] = {
                'timing': {
                    'mean_us': float(times_us.mean()),
                    'median_us': float(times_us.median()),
                    'std_us': float(times_us.std()),
                    'p90_us': float(times_us.quantile(0.90)),
                    'p95_us': float(times_us.quantile(0.95)),
                    'p99_us': float(times_us.quantile(0.99)),
                    'min_us': float(times_us.min()),
                    'max_us': float(times_us.max()),
                },
                'accuracy': {
                    'mean_position_error_mm': float(pos_errors.mean()),
                    'median_position_error_mm': float(pos_errors.median()),
                    'p95_position_error_mm': float(pos_errors.quantile(0.95)),
                    'max_position_error_mm': float(pos_errors.max()),
                    'mean_dx_mm': float(data['dx'].mean()),
                    'mean_dy_mm': float(data['dy'].mean()),
                    'mean_dtx': float(data['dtx'].mean()),
                    'mean_dty': float(data['dty'].mean()),
                },
                'metadata': {
                    'n_tracks': len(data),
                    'success_rate': float((data['success'] == 1).sum() / len(data)),
                },
            }
            
            # Compute throughput
            results[tree_name]['timing']['throughput_tracks_per_sec'] = 1e6 / results[tree_name]['timing']['mean_us']
    
    return results


def save_json_results(results: Dict, output_file: Path):
    """Save full results to JSON."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved full results to {output_file}")


def save_csv_summary(results: Dict, output_file: Path):
    """Save summary table to CSV."""
    rows = []
    for name, data in results.items():
        rows.append({
            'extrapolator': name,
            'mean_us': data['timing']['mean_us'],
            'median_us': data['timing']['median_us'],
            'p95_us': data['timing']['p95_us'],
            'throughput_tr_per_s': data['timing']['throughput_tracks_per_sec'],
            'mean_error_mm': data['accuracy']['mean_position_error_mm'],
            'p95_error_mm': data['accuracy']['p95_position_error_mm'],
            'success_rate': data['metadata']['success_rate'],
            'n_tracks': data['metadata']['n_tracks'],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('mean_us')  # Sort by speed
    df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"✓ Saved summary table to {output_file}")


def print_summary_table(results: Dict):
    """Print formatted summary to console."""
    print("\n" + "="*100)
    print("C++ Track Extrapolator Benchmark Results")
    print("="*100)
    print(f"{'Extrapolator':<30} {'Mean (μs)':<12} {'P95 (μs)':<12} {'Throughput':<15} {'Error (mm)':<12}")
    print("-"*100)
    
    # Sort by speed (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['timing']['mean_us'])
    
    for name, data in sorted_results:
        mean_us = data['timing']['mean_us']
        p95_us = data['timing']['p95_us']
        throughput = data['timing']['throughput_tracks_per_sec']
        error_mm = data['accuracy']['mean_position_error_mm']
        
        # Clean up name
        display_name = name.replace('TrackExtrapolatorTesterSOA/', '')
        
        print(f"{display_name:<30} {mean_us:>10.2f}   {p95_us:>10.2f}   "
              f"{throughput:>10.0f} tr/s   {error_mm:>8.3f}")
    
    print("="*100)
    
    # Print speedup comparison vs slowest (usually RK4)
    slowest_name, slowest_data = max(results.items(), key=lambda x: x[1]['timing']['mean_us'])
    slowest_time = slowest_data['timing']['mean_us']
    
    print(f"\nSpeedup vs {slowest_name.replace('TrackExtrapolatorTesterSOA/', '')}:")
    print("-"*60)
    for name, data in sorted_results:
        speedup = slowest_time / data['timing']['mean_us']
        display_name = name.replace('TrackExtrapolatorTesterSOA/', '')
        print(f"  {display_name:<35} {speedup:>6.1f}×")


def main():
    parser = argparse.ArgumentParser(description="Parse C++ benchmark results")
    parser.add_argument('root_file', help='ROOT file from TrackExtrapolatorTesterSOA')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse ROOT file
    print(f"Parsing {args.root_file}...")
    results = parse_root_file(args.root_file)
    
    if results is None:
        return 1
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    save_json_results(results, output_dir / 'benchmark_results.json')
    save_csv_summary(results, output_dir / 'benchmark_summary.csv')
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETE!")
    print("="*100)
    print(f"\nResults saved to {output_dir}/")
    print("  - benchmark_results.json  (full statistics)")
    print("  - benchmark_summary.csv   (quick reference)")
    print("\nNext step: Analyze results in analysis notebook")
    
    return 0


if __name__ == "__main__":
    exit(main())
