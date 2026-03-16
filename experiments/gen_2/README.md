# gen_2 — Reference Papers

This directory contains reference papers collected for future PINN architecture research. These inform potential next-generation approaches beyond the current V1–V5 experiments in `gen_1/`.

---

## Papers

| File | Title | Relevance |
|------|-------|-----------|
| `NeurIPS-2018-neural-ordinary-differential-equations-Paper.pdf` | Neural Ordinary Differential Equations (Chen et al., NeurIPS 2018) | Foundation for treating NN as continuous-depth ODE solvers — potential alternative to discrete PINN architectures |
| `ASR-PINN.pdf` | Adaptive Step-size Runge-Kutta PINNs | Adaptive integration step sizes within PINN training — relevant to variable dz problem |
| `1-s2.0-S00219Long_time_integration_ODE_PINNS99122009184-main.pdf` | Long-time Integration of ODEs with PINNs | Addresses PINN accuracy degradation over long integration domains — directly relevant to LHCb's 8000+ mm extrapolations |
| `Parameter estimation and modeling of nonlinear dynamical systems based on Runge–Kutta physics-informed neural network.pdf` | RK-PINN for Parameter Estimation in Nonlinear Dynamical Systems | RK-structured PINN for parameter estimation — could inform improved RK_PINN architectures |

---

## Context

The V4 PINN architecture diagnosis ([gen_1/V4/PINN_ARCHITECTURE_DIAGNOSIS.md](../gen_1/V4/PINN_ARCHITECTURE_DIAGNOSIS.md)) identified fundamental limitations in how PINNs handle spatially-varying magnetic fields over long integration domains. These papers explore approaches that may address those limitations in future experiment generations.

---

*Last Updated: March 2026*
