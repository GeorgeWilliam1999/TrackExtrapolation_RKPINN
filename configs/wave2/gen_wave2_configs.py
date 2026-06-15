#!/usr/bin/env python3
"""Emit the Wave-2 training configs: residual/kick head + range-aware loss +
tuned schedule, on the deployment-weighted corpus. Two families:

  size curve (lambda_pde=0, pure data -> isolates CAPACITY): h32..h384
  lambda sweep at fixed h128 (physical kappa): lambda_pde in {0, 0.01, 0.1}

The 64 KB Allen constant-memory budget is a DEPLOYMENT constraint only: we train
big for the accuracy ceiling, then distil/prune. ~ float32 param budget = 16384;
PINN_v2 [96,96]~10.0k params (fits), [128,128]~17.9k (just over), [256,256]~68.6k,
[384,384]~152k (4-9x over -> distil targets).
"""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
LAB = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"

BASE = dict(
    seed=42,
    data_path=f"{LAB}/data/train_wave2_deploy.npz",
    max_samples=6_000_000,          # > corpus size -> use all (~5.2M)
    train_fraction=0.8, val_fraction=0.1,
    model_type="pinn_v2", activation="tanh", dropout=0.0,
    n_collocation=2,
    # --- residual / kick parametrisation is the DEFAULT (Wave-2) ---
    kick_scaled_head=True,
    pde_scale_mode="fixed_L", pde_ref_length=5161.0,   # UT->T leg length
    # --- range-aware residual HUBER (small absolute per-component scale), balanced x,y(mm) vs tx,ty(rad) ---
    loss="residual_rel", resid_scale_pos=0.05, resid_scale_slope=2.0e-5, resid_alpha=0.0, resid_huber_delta=8.0,
    balance_sign=False, select_metric="utt_median_dx_um",
    # --- tuned optimisation (wave-1 early-stopped at epoch 3-17) ---
    batch_size=4096, epochs=120, learning_rate=7.0e-4, weight_decay=1.0e-4,
    warmup_epochs=12, grad_clip=1.0, patience=40, min_delta=1.0e-7,
    physics_warmup_epochs=15,
    checkpoint_dir=f"{LAB}/trained_models",
    use_mlflow=True, mlflow_experiment_name="gen_3_track_extrapolation",
    device="cuda", num_workers=2, pin_memory=True,
)


def write(name, **over):
    cfg = dict(BASE); cfg.update(over)
    cfg["experiment_name"] = name
    (HERE / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(name, cfg["hidden_dims"], "lam", cfg["lambda_pde"])


# size curve: pure data (lambda=0), capacity ladder
for h in (32, 64, 96, 128, 256, 384):
    write(f"wave2_resid_h{h}", hidden_dims=[h, h], lambda_pde=0.0, lambda_ic=0.0)

# lambda sweep at fixed h128 (h128 lam0 already emitted above)
write("wave2_lam0p01_h128", hidden_dims=[128, 128], lambda_pde=0.01, lambda_ic=0.001)
write("wave2_lam0p1_h128",  hidden_dims=[128, 128], lambda_pde=0.1,  lambda_ic=0.01)

print("\nwrote", len(list(HERE.glob('wave2_*.yaml'))), "configs ->", HERE)
