# V1 Benchmarking

C++ vs Python performance benchmarks for V1 models.

## Scripts

- `benchmark_cpp.py` - Benchmark C++ extrapolators
- `parse_benchmark_results.py` - Parse benchmark output
- `analyze_benchmarks.ipynb` - Analysis notebook

## Results

See `results/` directory for benchmark data.

## Key Findings

| Extrapolator | Time/track | Speedup |
|--------------|------------|---------|
| C++ RK4 (CashKarp) | 2.50 μs | Baseline |
| MLP tiny (Python) | 1.10 μs | **2.3×** |
| MLP medium (Python) | 1.47 μs | **1.7×** |
