# Benchmarking Guide - Complete Reference

**Last Updated:** January 14, 2025  
**Status:** Phase 1 Complete - Publication Ready

This is the comprehensive guide for C++ track extrapolator benchmarking.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Benchmark Results](#benchmark-results)
3. [Analysis Tools](#analysis-tools)
4. [Workflow](#workflow)
5. [Data Access](#data-access)
6. [Publication Figures](#publication-figures)

---

## Quick Start

### Run Analysis (Jupyter Notebook)

```bash
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/benchmarking
conda activate TE
jupyter notebook analyze_benchmarks.ipynb
# Run all cells to generate figures and exports
```

### View Results

**Figures:** `results/plots/` (PNG 300 DPI + PDF vector)  
**Statistics:** `results/benchmark_results.json`  
**Summary:** `results/benchmark_summary.csv`  
**LaTeX:** `results/benchmark_latex.tex`

---

## Benchmark Results

### Test Configuration

**Date:** January 14, 2025  
**Test Grid:** 1,210 track states  
**Propagation:** z = 3000mm → 7000mm (4m distance)  
**Momentum:** 3-100 GeV range  
**Environment:** LHCb Rec v39.3, x86_64_v2-el9-gcc13

### Extrapolators Tested

| Name | Type | Order | Notes |
|------|------|-------|-------|
| **BogackiShampine3** | RK3 | 3rd order | ⭐ Recommended |
| **RungeKutta** | RK4 | 4th order | Reference baseline |
| **Verner9** | RK9 | 9th order | Highest precision |
| **Verner7** | RK7 | 7th order | High precision |
| **Tsitouras5** | RK5 | 5th order | Balanced |
| **DormandPrince5** | RK5(4) | 5th order | DOPRI5 |
| **CashKarp** | RK5(4) | 5th order | Classic RK5 |
| **Herab** | Helix | Analytical | Fast approximation |
| **Kisel** | Analytical | - | Geometry dependent |

### Accuracy Results (Mean Error over 4m)

| Extrapolator | Mean (mm) | RMS (mm) | Max (mm) | Speed |
|--------------|-----------|----------|----------|-------|
| **Verner9** | **0.08** | 0.12 | 0.45 | Slow |
| **BogackiShampine3** | **0.10** | 0.14 | 2.10 | **Fast** ⭐ |
| RungeKutta | 0.12 | 0.18 | 1.32 | Medium |
| Verner7 | 0.24 | 0.34 | 5.22 | Slow |
| Tsitouras5 | 0.29 | 0.41 | 5.32 | Medium |
| DormandPrince5 | 0.35 | 0.48 | 6.85 | Fast |
| CashKarp | 0.38 | 0.52 | 7.12 | Fast |
| **Herab** | **0.76** | 1.24 | 31.58 | **Very Fast** |
| Kisel | 39.82 | 85.45 | 944.30 | Fast |

### Timing Results (Microseconds per Track)

| Extrapolator | Mean (μs) | Median (μs) | P95 (μs) |
|--------------|-----------|-------------|----------|
| **Herab** | **1.95** | **1.82** | **2.45** |
| DormandPrince5 | 8.34 | 7.98 | 11.23 |
| CashKarp | 8.56 | 8.12 | 11.58 |
| **BogackiShampine3** | **9.12** | **8.67** | **12.34** |
| Tsitouras5 | 10.45 | 9.89 | 14.12 |
| RungeKutta | 12.67 | 11.98 | 17.23 |
| Verner7 | 15.34 | 14.56 | 20.87 |
| Verner9 | 18.92 | 17.88 | 25.45 |

### Key Findings

✅ **Best Overall:** BogackiShampine3
- Excellent accuracy (0.1mm mean error)
- Fast execution (9.1μs per track)
- **Recommended for production**

✅ **Fastest:** Herab
- Good accuracy (0.76mm mean)
- Very fast (1.95μs per track)
- Ideal for track seeding

✅ **Highest Precision:** Verner9
- Best accuracy (0.08mm mean)
- Slower (18.9μs per track)
- Use for reference/validation

⚠️ **Avoid:** Kisel
- Unreliable (40mm mean, up to 944mm max)
- Geometry dependent failures

### Performance Analysis

**Adaptive Stepping Statistics (Tsitouras5 example):**
- Average steps: 11.56 per propagation
- Average step length: 428mm
- Failed steps: 19.2% (triggers re-stepping)
- Increased steps: 61.3% (error control)

**Key Insight:** Adaptive RK methods use ~10-15 steps for 4m propagation with ~350-430mm effective step size, achieving sub-mm to mm-level accuracy.

---

## Analysis Tools

### Primary Tool: Jupyter Notebook

**File:** `analyze_benchmarks.ipynb`

**Generates:**
- 4 publication-quality figure sets (PNG + PDF)
- Statistical comparisons (CSV)
- LaTeX tables for papers
- JSON data export

**Cells:**
1. Load data from `benchmark_results.json`
2. Statistical analysis (mean, median, percentiles)
3. Accuracy vs speed scatter plot
4. Timing distribution plots
5. Accuracy distribution plots
6. Performance comparison bar charts
7. Data exports (CSV, LaTeX)

### Supporting Tools

**parse_benchmark_results.py** - Convert ROOT to JSON
```bash
python parse_benchmark_results.py ../../benchmark_results.root
# Output: results/benchmark_results.json
```

**benchmark_cpp.py** - Automated test runner
```python
from benchmark_cpp import run_benchmark
config = {'extrapolators': ['BogackiShampine3', 'RungeKutta'], ...}
results = run_benchmark(config)
```

### Archived Tools (Reference Only)

Located in `archived/`:
- `analyze_log.py` - Old log parser
- `analyze_log_quick.py` - Quick analysis
- `parse_console_benchmark.py` - Console parser

Use Jupyter notebook instead - these are kept for reference.

---

## Workflow

### Complete Benchmark Cycle

```bash
# 1. Build C++ code
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators
make  # or lb-project-build

# 2. Run benchmarks
cd build.x86_64_v3-el9-gcc13-opt
./run TrackExtrapolators.TrackExtrapolatorTesterSOA

# 3. Parse results
cd ../experiments/next_generation/benchmarking
python parse_benchmark_results.py ../../benchmark_results.root

# 4. Generate analysis
jupyter notebook analyze_benchmarks.ipynb
# Run all cells

# 5. View outputs
ls results/plots/  # Publication figures
cat results/benchmark_summary.csv  # Quick reference
```

### Quick Analysis Workflow

```bash
# Parse existing ROOT file
python parse_benchmark_results.py ../../benchmark_results.root

# View JSON statistics
python -c "
import json
with open('results/benchmark_results.json') as f:
    data = json.load(f)
    for name, stats in data.items():
        print(f'{name}: {stats[\"accuracy\"][\"mean_error\"]:.2f}mm')
"
```

---

## Data Access

### File Locations

```
benchmarking/
├── results/
│   ├── benchmark_results.json      # Full statistics
│   ├── benchmark_summary.csv       # Quick reference
│   ├── benchmark_latex.tex         # Paper tables
│   └── plots/                      # Figures
│       ├── accuracy_vs_time.png/pdf
│       ├── timing_distribution.png/pdf
│       ├── accuracy_distribution.png/pdf
│       └── performance_comparison.png/pdf
```

### Loading in Python

```python
import json
import numpy as np
import pandas as pd

# Load full statistics
with open('results/benchmark_results.json') as f:
    data = json.load(f)

# Access specific extrapolator
bs3 = data['BogackiShampine3']
print(f"Mean error: {bs3['accuracy']['mean_error']:.2f} mm")
print(f"Mean time: {bs3['timing']['mean']:.2f} μs")

# Load CSV summary
df = pd.read_csv('results/benchmark_summary.csv')
print(df[['Extrapolator', 'Mean_Error_mm', 'Mean_Time_us']])
```

### Loading in Spreadsheet

**Excel/LibreOffice:**
1. Open `results/benchmark_summary.csv`
2. Create pivot tables and charts
3. Use for presentations

### Loading in LaTeX

```latex
\documentclass{article}
\usepackage{booktabs}
\begin{document}
\input{results/benchmark_latex.tex}
\end{document}
```

### Accessing Plots

**For Presentations (PNG):**
- 300 DPI resolution
- Suitable for slides
- Path: `results/plots/*.png`

**For Publications (PDF):**
- Vector graphics
- Scalable quality
- Path: `results/plots/*.pdf`

---

## Publication Figures

### Figure 1: Accuracy vs Speed

**File:** `accuracy_vs_time.png` / `.pdf`

**Shows:** Scatter plot of mean error vs mean execution time
- BogackiShampine3 in optimal zone (low error, fast)
- Herab fastest but higher error
- Verner9 most accurate but slower

**Caption:** "Trade-off between accuracy and speed for 7 track extrapolators over 4m propagation in LHCb magnetic field."

### Figure 2: Timing Distributions

**File:** `timing_distribution.png` / `.pdf`

**Shows:** Box plots of execution time distributions
- Median, quartiles, outliers
- Compare all extrapolators

**Caption:** "Execution time distributions showing median, quartiles, and outliers for each extrapolator."

### Figure 3: Accuracy Distributions

**File:** `accuracy_distribution.png` / `.pdf`

**Shows:** Histograms of position errors
- Error distributions across all tracks
- Identify tail behavior

**Caption:** "Position error distributions after 4m propagation, demonstrating sub-mm to mm-level accuracy for RK methods."

### Figure 4: Performance Comparison

**File:** `performance_comparison.png` / `.pdf`

**Shows:** Bar chart comparing key metrics
- Side-by-side comparison
- Normalized scores

**Caption:** "Comprehensive performance comparison across timing, accuracy, and reliability metrics."

---

## Recommendations

### For Production Tracking

**Use: BogackiShampine3**
- Best accuracy-speed trade-off
- 0.1mm mean error, 9μs per track
- Proven in LHCb production

### For Fast Seeding

**Use: Herab**
- Fastest execution (2μs per track)
- Acceptable accuracy (0.76mm mean)
- Ideal for initial track finding

### For Reference/Validation

**Use: Verner9 or RungeKutta**
- Highest precision (0.08-0.12mm)
- Use as ground truth
- Validate other methods

### Avoid

**Don't use: Kisel**
- Unreliable (40mm mean, up to 944mm max)
- Geometry-dependent failures
- Not recommended for general use

---

## Troubleshooting

### ROOT File Not Found

**Error:** Cannot open benchmark_results.root

**Solution:**
```bash
cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/build.*
./run TrackExtrapolators.TrackExtrapolatorTesterSOA
# Check for ROOT file in current directory
```

### Jupyter Kernel Issues

**Error:** Kernel not found or crashes

**Solution:**
```bash
conda activate TE
python -m ipykernel install --user --name TE --display-name "Python (TE)"
jupyter notebook analyze_benchmarks.ipynb
```

### Missing Python Packages

**Error:** ModuleNotFoundError

**Solution:**
```bash
conda activate TE
pip install numpy matplotlib pandas seaborn jupyter
```

---

## Contact & Support

**Maintained by:** G. Scriven  
**Last Benchmark Run:** January 14, 2025  
**LHCb Framework:** Rec v39.3  

**Documentation:**
- This file (complete reference)
- `data_generation/README.md` - Next phase
- `../INDEX.md` - Project navigation

**Status:** Phase 1 (Benchmarking) Complete ✅
