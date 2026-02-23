#!/usr/bin/env python3
"""
Benchmark C++ Extrapolators

This script provides a Python interface to benchmark the C++ track extrapolators.
It generates test data and analyzes the C++ benchmark results.

Author: G. Scriven
Date: 2025-01-14
"""

import numpy as np
import argparse
from pathlib import Path
import subprocess
import json
from typing import Dict, List


def generate_test_tracks(n_tracks: int = 1000, output_file: str = 'test_tracks.txt'):
    """
    Generate test tracks for C++ benchmarking.
    
    Outputs simple text format that C++ can read:
    x y tx ty qop z_start z_end
    """
    print(f"Generating {n_tracks} test tracks...")
    
    # Parameter ranges (same as ExtrapolatorTester.cpp)
    qop_range = (-0.0004, 0.0004)  # ±2.5 GeV
    tx_range = (-0.3, 0.3)
    ty_range = (-0.25, 0.25)
    z_start = 3000.0
    z_end = 7000.0
    
    tracks = []
    for _ in range(n_tracks):
        x = np.random.uniform(-800, 800)
        y = np.random.uniform(-600, 600)
        tx = np.random.uniform(*tx_range)
        ty = np.random.uniform(*ty_range)
        qop = np.random.uniform(*qop_range)
        
        tracks.append([x, y, tx, ty, qop, z_start, z_end])
    
    tracks = np.array(tracks)
    np.savetxt(output_file, tracks, 
               fmt='%.6f', 
               header='x y tx ty qop z_start z_end',
               comments='')
    
    print(f"✓ Saved {n_tracks} tracks to {output_file}")
    return tracks


def parse_cpp_benchmark_results(root_file: str) -> Dict:
    """
    Parse ROOT file from TrackExtrapolatorTesterSOA.
    
    Note: This requires ROOT Python bindings. If not available,
    you'll need to convert ROOT → CSV first using ROOT macros.
    """
    try:
        import ROOT
        
        f = ROOT.TFile(root_file)
        results = {}
        
        # Iterate through ntuples (one per extrapolator)
        for key in f.GetListOfKeys():
            name = key.GetName()
            tree = f.Get(name)
            
            # Extract timing data
            times = []
            for entry in tree:
                times.append(entry.time)  # nanoseconds
            
            times = np.array(times)
            
            results[name] = {
                'mean_ns': float(np.mean(times)),
                'median_ns': float(np.median(times)),
                'p90_ns': float(np.percentile(times, 90)),
                'p95_ns': float(np.percentile(times, 95)),
                'p99_ns': float(np.percentile(times, 99)),
                'std_ns': float(np.std(times)),
                'n_tracks': len(times)
            }
        
        f.Close()
        return results
    
    except ImportError:
        print("⚠ ROOT Python bindings not available")
        print("   You'll need to manually convert ROOT → CSV")
        print("   Or install PyROOT: pip install root-numpy")
        return {}


def print_benchmark_table(results: Dict):
    """Print formatted benchmark results table."""
    print("\n" + "=" * 90)
    print("C++ Extrapolator Benchmark Results")
    print("=" * 90)
    print(f"{'Extrapolator':<30} {'Mean (μs)':<12} {'P90 (μs)':<12} {'P95 (μs)':<12} {'Throughput':<15}")
    print("-" * 90)
    
    for name, data in sorted(results.items()):
        mean_us = data['mean_ns'] / 1000.0
        p90_us = data['p90_ns'] / 1000.0
        p95_us = data['p95_ns'] / 1000.0
        throughput = 1e6 / data['mean_ns']  # tracks/second
        
        print(f"{name:<30} {mean_us:>10.2f}   {p90_us:>10.2f}   {p95_us:>10.2f}   {throughput:>10.0f} tr/s")
    
    print("=" * 90)


def save_results_json(results: Dict, output_file: str):
    """Save benchmark results to JSON."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark C++ track extrapolators")
    
    parser.add_argument('--generate-tracks', action='store_true',
                       help='Generate test tracks file')
    parser.add_argument('--n-tracks', type=int, default=1000,
                       help='Number of test tracks (default: 1000)')
    parser.add_argument('--parse-results', type=str,
                       help='Parse ROOT file with benchmark results')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.generate_tracks:
        generate_test_tracks(args.n_tracks)
    
    if args.parse_results:
        print(f"\nParsing benchmark results from {args.parse_results}...")
        results = parse_cpp_benchmark_results(args.parse_results)
        
        if results:
            print_benchmark_table(results)
            save_results_json(results, args.output)
        else:
            print("\n⚠ Could not parse ROOT file")
            print("\nManual steps:")
            print("1. Open ROOT file in ROOT:")
            print("   root -l benchmark_results.root")
            print("2. Convert to CSV:")
            print("   ntuple->Scan(\"*\", \"\", \"\", 1000000, 0);")
            print("3. Save and parse CSV instead")
    
    if not args.generate_tracks and not args.parse_results:
        print("\nC++ Benchmark Workflow:")
        print("=" * 70)
        print("\n1. Generate test tracks:")
        print("   python benchmark_cpp.py --generate-tracks --n-tracks 1000")
        print("\n2. Run C++ benchmark in LHCb environment:")
        print("   gaudirun.py tests/options/benchmark_extrapolators.py")
        print("\n3. Parse results:")
        print("   python benchmark_cpp.py --parse-results benchmark.root")
        print("\n" + "=" * 70)
        print("\nNote: You'll need to create tests/options/benchmark_extrapolators.py")
        print("      Configuration should use TrackExtrapolatorTesterSOA algorithm")


if __name__ == "__main__":
    main()
