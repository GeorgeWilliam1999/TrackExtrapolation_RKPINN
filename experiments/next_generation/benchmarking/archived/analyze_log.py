#!/usr/bin/env python3
"""
Quick analysis of C++ extrapolator benchmark logs.

Extracts accuracy comparisons from test output.
"""

import re
import sys
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
            
            # Match extrapolator result (note: extrapolator name has leading spaces)
            result_match = re.search(r'SUCCESS\s+(\w+)\s+->\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)', line)
            if result_match and current_key is not None:
                extrap_name = result_match.group(1)
                current_state['extrapolators'][extrap_name] = {
                    'x': float(result_match.group(2)),
                    'y': float(result_match.group(3)),
                    'tx': float(result_match.group(4)),
                    'ty': float(result_match.group(5))
                }
                
                # If we have Reference, calculate errors for others
                if extrap_name == 'Reference':
                    ref = current_state['extrapolators']['Reference']
                    for name, vals in current_state['extrapolators'].items():
                        if name != 'Reference':
                            dx = vals['x'] - ref['x']
                            dy = vals['y'] - ref['y']
                            error = (dx**2 + dy**2)**0.5
                            vals['position_error_mm'] = error
                
                # Store when we finish each extrapolator comparison
                # (Don't wait for all 9 - they come in sequence across all states)
    
    # Calculate errors after collecting all data
    for state in results:
        if 'Reference' in state['extrapolators']:
            ref = state['extrapolators']['Reference']
            for name, vals in state['extrapolators'].items():
                if name != 'Reference':
                    dx = vals['x'] - ref['x']
                    dy = vals['y'] - ref['y']
                    error = (dx**2 + dy**2)**0.5
                    vals['position_error_mm'] = error
    
    # Store each state when we see it
    if current_state and current_state['extrapolators']:
        results.append(current_state)
    
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
    
    print(f"{'Extrapolator':<20} {'Mean Error (mm)':<18} {'Max Error (mm)':<18} {'Min Error (mm)':<15}")
    print("-" * 80)
    
    for name in sorted(extrapolator_errors.keys()):
        errors = extrapolator_errors[name]
        mean_err = sum(errors) / len(errors)
        max_err = max(errors)
        min_err = min(errors)
        
        print(f"{name:<20} {mean_err:>15.3f}   {max_err:>15.3f}   {min_err:>15.3f}")
    
    print("="*80 + "\n")

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
    summarize_accuracy(results)

if __name__ == "__main__":
    main()
