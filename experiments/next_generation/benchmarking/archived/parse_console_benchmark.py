#!/usr/bin/env python3
"""
Parse benchmark results from console output and generate publication-ready JSON.
This script processes the output from ExtrapolatorTester algorithm.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np

def parse_benchmark_log(log_file):
    """
    Parse the benchmark log file and extract all extrapolation results.
    
    Expected format:
    ExtrapolatorTester INFO Extrapolator: <name> - Mean error: <value> mm
    """
    
    results = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for extrapolator result lines
            if 'ExtrapolatorTester' in line or 'BenchmarkExtrapolators' in line:
                # Parse extrapolator name and errors
                if 'Extrapolator:' in line:
                    # Example: "BenchmarkExtrapolators INFO Extrapolator: Reference - Mean error: 0.104 mm"
                    match = re.search(r'Extrapolator:\s+(\w+).*Mean error:\s+([\d.]+)\s+mm', line)
                    if match:
                        name = match.group(1)
                        mean_error = float(match.group(2))
                        
                        if name not in results:
                            results[name] = {
                                'name': name,
                                'errors': [],
                                'times': [],
                                'success_count': 0,
                                'fail_count': 0
                            }
                        
                        results[name]['errors'].append(mean_error)
                
                # Look for timing information
                if 'Time per extrapolation:' in line:
                    match = re.search(r'Time per extrapolation:\s+([\d.]+)\s+ns', line)
                    if match and name in results:
                        results[name]['times'].append(float(match.group(1)))
                
                # Look for success/failure counts
                if 'Success:' in line:
                    match = re.search(r'Success:\s+(\d+)', line)
                    if match and name in results:
                        results[name]['success_count'] = int(match.group(1))
                
                if 'Failed:' in line:
                    match = re.search(r'Failed:\s+(\d+)', line)
                    if match and name in results:
                        results[name]['fail_count'] = int(match.group(1))
    
    # Compute statistics for each extrapolator
    benchmark_results = []
    for name, data in results.items():
        if data['errors']:
            errors = np.array(data['errors'])
            times = np.array(data['times']) if data['times'] else np.array([0])
            
            benchmark_results.append({
                'extrapolator': name,
                'mean_error_mm': float(np.mean(errors)),
                'std_error_mm': float(np.std(errors)),
                'min_error_mm': float(np.min(errors)),
                'max_error_mm': float(np.max(errors)),
                'median_error_mm': float(np.median(errors)),
                'mean_time_ns': float(np.mean(times)) if times.size > 0 else None,
                'std_time_ns': float(np.std(times)) if times.size > 0 else None,
                'success_rate': data['success_count'] / (data['success_count'] + data['fail_count']) 
                    if (data['success_count'] + data['fail_count']) > 0 else 1.0,
                'n_samples': len(errors)
            })
    
    return {
        'benchmark_type': 'console_output',
        'extrapolators': sorted(benchmark_results, key=lambda x: x['mean_error_mm'])
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_console_benchmark.py <log_file> [output_json]")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('results/benchmark_results.json')
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"Parsing benchmark log: {log_file}")
    results = parse_benchmark_log(log_file)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nFound {len(results['extrapolators'])} extrapolators:")
    for ext in results['extrapolators']:
        print(f"  - {ext['extrapolator']:20s}: {ext['mean_error_mm']:.4f} ± {ext['std_error_mm']:.4f} mm "
              f"({ext['n_samples']} samples)")
        if ext['mean_time_ns']:
            print(f"                              Time: {ext['mean_time_ns']:.2f} ± {ext['std_time_ns']:.2f} ns")

if __name__ == '__main__':
    main()
