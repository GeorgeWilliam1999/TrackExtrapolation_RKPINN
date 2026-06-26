#!/usr/bin/env python3
"""Phase-A follow-up configs (2026-06-25): push E1 toward the 1 mm target.

E1 broke the ~3 mm floor (unroll8 -> 1.40 mm UT->T) but was NOT converged
(best_epoch ~120) and improved monotonically with steps. So:
  e1_unroll8_long : same 8 steps, longer schedule  -> isolates "more epochs"
  e1_unroll16     : 16 steps, longer schedule       -> "more steps"
  e1_unroll32     : 32 steps                          -> diminishing-returns probe
All pinn_v2 [96,96] kick_scaled_head tanh, field-free, gen-4 (Allen-faithful).
"""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"

BASE = dict(
    seed=42,
    data_path=f"{LAB}/data/train_wave2_deploy.npz",
    max_samples=6_000_000, train_fraction=0.8, val_fraction=0.1,
    model_type="pinn_v2", hidden_dims=[96, 96], activation="tanh", dropout=0.0,
    n_collocation=2, kick_scaled_head=True, pde_scale_mode="fixed_L", pde_ref_length=5161.0,
    loss="residual_rel", resid_scale_pos=0.05, resid_scale_slope=2.0e-5,
    resid_alpha=0.0, resid_huber_delta=8.0, resid_weight_mode="none",
    balance_sign=False, select_metric="utt_median_dx_um",
    batch_size=4096, learning_rate=7.0e-4, weight_decay=1.0e-4,
    warmup_epochs=15, grad_clip=1.0, min_delta=1.0e-7,
    physics_warmup_epochs=15, lambda_pde=0.0, lambda_ic=0.0,
    n_rk_steps=1, correction_scale_init=1.0e-3, disable_correction=False,
    checkpoint_dir=f"{LAB}/trained_models",
    use_mlflow=True, mlflow_experiment_name="gen_3_track_extrapolation",
    device="cuda", num_workers=2, pin_memory=True,
)


def write(name, **over):
    cfg = dict(BASE); cfg.update(over); cfg["experiment_name"] = name
    (HERE / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{name:18s} n_unroll={cfg['n_unroll']:<3} epochs={cfg['epochs']} patience={cfg['patience']}")


write("e1_unroll8_long", n_unroll=8,  epochs=250, patience=60)
write("e1_unroll16",     n_unroll=16, epochs=250, patience=60)
write("e1_unroll32",     n_unroll=32, epochs=200, patience=60)
