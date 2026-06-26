#!/usr/bin/env python3
"""Emit Phase-A configs for the 'Break the ~3 mm floor' program (2026-06-25).

E1 — field-free multi-step unroll of the deployable kick head (pinn_v2,
     kick_scaled_head, [96,96] tanh) over N sub-steps. Tier-0 (no field lookup).
E5 — multi-step hybrid integrator (neural_rk4: RK4 on the Lorentz ODE + learned
     RHS correction) on the real gen-4 field, to set the field-access accuracy bar.

Everything else is held identical to the wave-2 / arch×cost scan so numbers compare
directly. Deployable shapes only (Allen-faithful).
"""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"

BASE = dict(
    seed=42,
    data_path=f"{LAB}/data/train_wave2_deploy.npz",
    max_samples=6_000_000, train_fraction=0.8, val_fraction=0.1,
    model_type="pinn_v2", activation="tanh", dropout=0.0, n_collocation=2,
    kick_scaled_head=True, pde_scale_mode="fixed_L", pde_ref_length=5161.0,
    loss="residual_rel", resid_scale_pos=0.05, resid_scale_slope=2.0e-5,
    resid_alpha=0.0, resid_huber_delta=8.0, resid_weight_mode="none",
    balance_sign=False, select_metric="utt_median_dx_um",
    batch_size=4096, epochs=120, learning_rate=7.0e-4, weight_decay=1.0e-4,
    warmup_epochs=12, grad_clip=1.0, patience=40, min_delta=1.0e-7,
    physics_warmup_epochs=15, lambda_pde=0.0, lambda_ic=0.0,
    # neural_rk4 keys (read directly by _build_model; harmless for pinn_v2):
    n_rk_steps=1, correction_scale_init=1.0e-3, disable_correction=False,
    n_unroll=1,
    checkpoint_dir=f"{LAB}/trained_models",
    use_mlflow=True, mlflow_experiment_name="gen_3_track_extrapolation",
    device="cuda", num_workers=2, pin_memory=True,
)


def write(name, **over):
    cfg = dict(BASE); cfg.update(over); cfg["experiment_name"] = name
    (HERE / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{name:16s} type={cfg['model_type']:10s} dims={cfg['hidden_dims']!s:12s} "
          f"n_unroll={cfg['n_unroll']} n_rk_steps={cfg['n_rk_steps']} "
          f"disable_corr={cfg['disable_correction']}")


# ---- E1: field-free multi-step unroll of the deployable kick head ----
for N in (2, 4, 8):
    write(f"e1_unroll{N}", model_type="pinn_v2", hidden_dims=[96, 96],
          activation="tanh", kick_scaled_head=True, n_unroll=N)

# ---- E5: multi-step hybrid integrator on the real field (sets the bar) ----
for N in (2, 4):
    write(f"e5_hybrid_n{N}", model_type="neural_rk4", hidden_dims=[64, 64],
          activation="tanh", n_rk_steps=N, correction_scale_init=1.0e-3,
          disable_correction=False)
