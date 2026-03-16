# Documentation Index: PINN Architecture Investigation

This directory contains comprehensive documentation of the PINN architecture issue discovered in February 2026, the root cause analysis, and V4 solutions.

## 📚 Main Documents

### 1. [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) ⭐ **START HERE**

**Comprehensive technical document covering:**
- Complete architectural analysis of V2/V3 PINN failure
- Mathematical proofs of the linear ansatz limitation
- Experimental evidence from trajectory visualization
- Why collocation points couldn't overcome the bottleneck
- Impact of variable magnetic field on position dependence
- Three V4 solutions with implementation details
- Experimental validation plan
- Lessons learned for physics-informed networks

**Length:** ~11,000 words  
**Audience:** Researchers, PhD students, anyone implementing PINNs for variable-coefficient PDEs  
**Key Sections:**
- Section 2: The fundamental flaw (mathematical analysis)
- Section 4: Experimental evidence (data and plots)
- Section 7: V4 architectural solutions (PINNZFracInput, QuadraticResidual, PDE-Residual)
- Section 10: Lessons learned (general principles)

---

### 2. [README.md](README.md) - V4 Project Overview

**Experiment plan and context:**
- V3 summary and lessons learned
- V4 strategy: MLP width scaling + PINN investigation
- Detailed experiment plans (P1-P6)
- Success criteria and timeline
- Links to PINN diagnosis

**Audience:** Project team, collaborators  
**Use:** Planning experiments, tracking progress

---

### 3. Cross-References to Other Directories

These documents in other directories provide supporting evidence and analysis:

#### V2 Directory

**[V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md)**
- Executive summary of the issue
- Evidence from V2 experiments (fixed dz=8000mm)
- Performance comparison: MLP vs PINN
- Four proposed fixes with code examples
- Quick reference diagrams

**[V2/QUICK_REFERENCE.md](../V2/QUICK_REFERENCE.md)**
- One-page visual summary
- ASCII diagrams of architecture
- Field variation plot
- Quick lookup table of fixes

**[V2/analysis/v2_model_analysis.ipynb](../V2/analysis/v2_model_analysis.ipynb)**
- Jupyter notebook with V2 model evaluation
- Performance metrics and Pareto frontiers
- Section 8: Links to PINN architecture issue

**[V2/README.md](../V2/README.md)**
- Updated with critical finding banner
- Summary of what went wrong
- Pointer to full documentation

#### Physics Exploration

**[physics_exploration.ipynb](../physics_exploration.ipynb) - Section 8**
- Mathematical analysis of trajectory physics
- Magnetic field variation study
- PINN trajectory predictions at intermediate z
- Code to load and analyze PINN models
- MLP vs PINN comparison experiments
- Visualization of linear vs nonlinear trajectories

**Key cells:**
- "Critical Analysis: Why V2 PINN Models Fail" (markdown)
- PINN forward pass trace (code)
- Trajectory comparison plots (code + output)
- Field variation analysis (code + plots)
- MLP vs PINN head-to-head (code + results)

---

## 🔍 Document Organization by Use Case

### "I want to understand what went wrong"

1. Start with [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) Section 1-2 (background + fundamental flaw)
2. See visual diagrams in [V2/QUICK_REFERENCE.md](../V2/QUICK_REFERENCE.md)
3. Run code in [physics_exploration.ipynb](../physics_exploration.ipynb) Section 8 to reproduce

### "I want to implement the fix"

1. Read [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) Section 7-8 (solutions + implementation)
2. See code in [training/train_v4.py](training/train_v4.py) (`PINNZFracInput` class)
3. Follow experiment plan in [README.md](README.md) Section "Experiment Plan: PINN Training Investigation"

### "I want the evidence"

1. See [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) Section 4 (experimental evidence)
2. Check tables in [V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md) Section "Evidence"
3. Run experiments in [physics_exploration.ipynb](../physics_exploration.ipynb) Section 8

### "I want lessons for my own PINN"

1. Read [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) Section 10 (lessons learned)
2. See [V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md) Section "Lessons Learned"
3. Consider general principles for variable-coefficient PDEs

### "I want a quick summary"

1. [V2/QUICK_REFERENCE.md](../V2/QUICK_REFERENCE.md) - one page with diagrams
2. [V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md) - executive summary
3. [README.md](README.md) - "Executive Summary of PINN Issue" box

---

## 📊 Key Figures and Tables

### Performance Data

**MLPs vs PINNs on Variable dz (V3):**

| Model | Pos Error (mm) | Slope Error | Has z Input? |
|:------|---------------:|------------:|:------------:|
| V3 MLP | 1.01 | 0.0115 | ✓ |
| V3 PINN | 49.4 | 0.000249 | ❌ |
| V4 PINNZFracInput (expected) | <1 | ~0.0005 | ✓ |

**MLPs vs PINNs on Fixed dz=8000mm (V2):**

| Model | Pos Error (mm) | Slope Error | Has z Input? |
|:------|---------------:|------------:|:------------:|
| V2 MLP | 0.08 | 0.0092 | ✓ |
| V2 PINN | 0.15 | 0.00025 | ❌ |

### Trajectory Comparison

See [physics_exploration.ipynb](../physics_exploration.ipynb) for plots showing:
- PINN predictions: perfectly linear in z_frac
- RK4 ground truth: nonlinear curvature
- Errors growing systematically with distance

### Field Variation

LHCb dipole field variation quantifies why position-dependence matters:
- By(0mm) = 1.1 T → By(8000mm) = 0.4 T
- Factor of 3× variation in curvature
- Linear corrections cannot capture this

---

## 🎯 Quick Navigation

**For different audiences:**

| You are... | Start here... | Then read... |
|:-----------|:-------------|:-------------|
| **PhD student / researcher** | [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) | [physics_exploration.ipynb](../physics_exploration.ipynb) |
| **Software engineer implementing fix** | [training/train_v4.py](training/train_v4.py) | [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) Section 7-8 |
| **Project collaborator** | [README.md](README.md) | [V2/QUICK_REFERENCE.md](../V2/QUICK_REFERENCE.md) |
| **Reviewer / external researcher** | [V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md) | [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) |
| **Just curious** | [V2/QUICK_REFERENCE.md](../V2/QUICK_REFERENCE.md) | [V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md) |

---

## 💡 Key Insights (TL;DR)

1. **Root Cause**: V2/V3 PINN encoder did not include z_frac as input → position-independent corrections → linear ansatz → cannot represent nonlinear trajectories in variable field

2. **Evidence**: 
   - PINNs: 50mm position error, 0.0003 slope error
   - MLPs: 1mm position error, 0.009 slope error
   - PINNs are 2× worse on positions, 37× better on slopes!

3. **Why**: Positions evolve nonlinearly (quadratic in z), slopes evolve approximately linearly → PINN's linear ansatz works for slopes but fails for positions

4. **Variable Field Impact**: LHCb field varies 3× → curvature varies 3× → position information essential

5. **Collocation Didn't Help**: Architecture was bottleneck, not supervision. Increasing collocation 5→50 gave no improvement.

6. **V4 Fix**: Add z_frac to encoder input → network can learn position-dependent corrections → expected <1mm position error while maintaining slope accuracy

7. **General Lesson**: Physics-informed architectures must have sufficient expressivity to represent the physics. Too much constraint can be worse than too little.

---

## 📝 Citation

If you use these findings in your work, please cite:

```
@techreport{scriven2026pinn,
  title={PINN Architecture Diagnosis: Root Cause Analysis of Position Errors in Variable-Field Track Extrapolation},
  author={Scriven, G.},
  institution={LHCb Collaboration, Nikhef},
  year={2026},
  month={February},
  note={Technical documentation for V4 track extrapolator experiments}
}
```

**Or reference:**
- V4 PINN_ARCHITECTURE_DIAGNOSIS.md
- V2 PINN_ARCHITECTURE_ISSUE.md
- physics_exploration.ipynb Section 8

---

## ✅ Validation Status

| Finding | Status | Evidence Location |
|:--------|:-------|:------------------|
| V2/V3 PINN has linear constraints | ✅ Confirmed | [physics_exploration.ipynb](../physics_exploration.ipynb) Section 8.4 |
| Encoder missing z_frac input | ✅ Confirmed | [V2/models/architectures.py](../V2/models/architectures.py) lines 547-567 |
| Predictions are perfectly linear | ✅ Confirmed | trajectory visualization in notebook |
| Field varies 3× across tracking | ✅ Confirmed | field map analysis |
| Collocation doesn't help | ✅ Confirmed | V3 results: col5 ≈ col10 ≈ col50 |
| MLPs beat PINNs on position | ✅ Confirmed | V2 analysis: 0.08mm vs 0.15mm |
| PINNs beat MLPs on slopes | ✅ Confirmed | V3 results: 0.00025 vs 0.0115 |
| V4 fix (PINNZFracInput) works | 🔄 In progress | Training on cluster |

---

## 📧 Contact

For questions or discussions about this analysis:
- G. Scriven (LHCb Collaboration)
- [physics_exploration.ipynb](../physics_exploration.ipynb) can be run interactively
- See [README.md](README.md) for experiment progress updates

---

*Last Updated: February 20, 2026*  
*Status: V4 experiments ongoing - validation of fixes in progress*
