#!/usr/bin/env python3
"""
Quick analysis of C++ extrapolator benchmark logs.
Extracts accuracy comparisons from test output.
"""

import json
import re
import sys
import statistics
from pathlib import Path

def parse_propagation_results(log_file):
    """Extract track state propagation results from log."""
    
    # Store all states keyed by initial conditions
    all_states = {}
    current_key = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match propagation header
            prop_match = re.search(r'SUCCESS Propagating\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)\s+with q\*p = ([-\d.]+)', line)
            if prop_match:
                # Create a unique key for this state
                current_key = (
                    float(prop_match.group(1)),
                    float(prop_match.group(2)),
                    float(prop_match.group(3)),
                    float(prop_match.group(4)),
                    float(prop_match.group(5))
                )
                
                if current_key not in all_states:
                    all_states[current_key] = {
                        'x0': current_key[0],
                        'y0': current_key[1],
                        'tx0': current_key[2],
                        'ty0': current_key[3],
                        'qop': current_key[4],
                        'extrapolators': {}
                    }
            
            # Match extrapolator result
            result_match = re.search(r'SUCCESS\s+(\w+)\s+->\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)', line)
            if result_match and current_key is not None:
                extrap_name = result_match.group(1)
                all_states[current_key]['extrapolators'][extrap_name] = {
                    'x': float(result_match.group(2)),
                    'y': float(result_match.group(3)),
                    'tx': float(result_match.group(4)),
                    'ty': float(result_match.group(5))
                }
    
    # Convert to list and calculate errors
    results = list(all_states.values())
    
    for state in results:
        if 'Reference' in state['extrapolators']:
            ref = state['extrapolators']['Reference']
            for name, vals in state['extrapolators'].items():
                if name != 'Reference':
                    dx = vals['x'] - ref['x']
                    dy = vals['y'] - ref['y']
                    error = (dx**2 + dy**2)**0.5
                    vals['position_error_mm'] = error
    
    return results

def summarize_accuracy(results):
    """Calculate summary statistics for each extrapolator."""
    
    if not results:
        print("No results found!")
        return
    
    # Collect errors for each extrapolator
    extrapolator_errors = {}
    
    for state in results:
        for name, vals in state['extrapolators'].items():
            if name != 'Reference' and 'position_error_mm' in vals:
                if name not in extrapolator_errors:
                    extrapolator_errors[name] = []
                extrapolator_errors[name].append(vals['position_error_mm'])
    
    # Print summary
    print("\n" + "="*80)
    print("ACCURACY SUMMARY (Position Error vs Reference RK4)")
    print("="*80)
    print(f"\nTotal track states analyzed: {len(results)}")
    print(f"Propagation distance: 4000 mm (z: 3000 → 7000)\n")
    
    print(f"{'Extrapolator':<25} {'Mean Error (mm)':<18} {'Max Error (mm)':<18} {'Min Error (mm)':<15}")
    print("-" * 90)
    
    for name in sorted(extrapolator_errors.keys()):
        errors = extrapolator_errors[name]
        mean_err = sum(errors) / len(errors)
        max_err = max(errors)
        min_err = min(errors)
        
        print(f"{name:<25} {mean_err:>15.4f}   {max_err:>15.4f}   {min_err:>15.4f}")
    
    print("="*90 + "\n")

def main():
    # Default log file
    log_file = Path("/data/bfys/gscriven/TE_stack/test_qmt.log")
    
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"Analyzing: {log_file}")
    
    results = parse_propagation_results(log_file)
    accuracy_summary = summarize_accuracy(results)
    
    # Save results to JSON if output file specified
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-friendly format
        json_results = {
            "benchmark_type": "console_log_analysis",
            "total_states": len(next(iter(results.values()))['errors']),
            "propagation_distance_mm": 4000,
            "extrapolators": []
        }
        
        for extrap_name, data in results.items():
            errors = data['errors']
            json_results["extrapolators"].append({
                "extrapolator": extrap_name,
                "mean_error_mm": float(statistics.mean(errors)),
                "std_error_mm": float(statistics.stdev(errors)) if len(errors) > 1 else 0.0,
                "min_error_mm": float(min(errors)),
                "max_error_mm": float(max(errors)),
                "median_error_mm": float(statistics.median(errors)),
                "mean_time_ns": None,  # Not available from console log
                "std_time_ns": None,
                "success_rate": 1.0,
                "n_samples": len(errors)
            })
        
        # Sort by accuracy
        json_results["extrapolators"].sort(key=lambda x: x["mean_error_mm"])
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
