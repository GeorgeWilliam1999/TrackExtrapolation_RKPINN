#!/usr/bin/env python3
"""Emit the 2026-06-23 architecture x cost-function scan configs.

Width is already swept (wave-2 capacity ladder h32..h384, flat ~3 mm), so this
scan isolates the THREE untested levers below the ~3 mm UT->T floor:

  Block A  depth {2,3} x activation {tanh,silu,gelu,sin}   (width 64, default loss)
  Block B  cost-function variants at fixed [96,96] tanh    (alpha-blend, huber-delta,
           inv-p tail weight [new], log-cosh)

Everything else is held identical to the wave-2 ladder so the numbers compare
directly to wave2_resid_h{32..384}. Baseline B0 = wave2_resid_h96 (reuse, no rerun).
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
    checkpoint_dir=f"{LAB}/trained_models",
    use_mlflow=True, mlflow_experiment_name="gen_3_track_extrapolation",
    device="cuda", num_workers=2, pin_memory=True,
)


def write(name, **over):
    cfg = dict(BASE); cfg.update(over); cfg["experiment_name"] = name
    (HERE / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{name:22s} {cfg['hidden_dims']!s:14s} act={cfg['activation']:5s} "
          f"loss={cfg['loss']} a={cfg['resid_alpha']} d={cfg['resid_huber_delta']} "
          f"w={cfg['resid_weight_mode']}")


# ---- Block A: depth x activation (width 64, default residual_rel loss) ----
for depth, dims in ((2, [64, 64]), (3, [64, 64, 64])):
    for act in ("tanh", "silu", "gelu", "sin"):
        over = dict(hidden_dims=dims, activation=act)
        if act == "sin":
            over["siren_w0"] = 30.0
        write(f"scanA_d{depth}_{act}", **over)

# ---- Block B: cost function at fixed [96,96] tanh (deployable shape) ----
H = [96, 96]
write("scanB_alpha05",  hidden_dims=H, resid_alpha=0.5)
write("scanB_delta2",   hidden_dims=H, resid_huber_delta=2.0)
write("scanB_invp",     hidden_dims=H, resid_weight_mode="inv_p")
write("scanB_logcosh",  hidden_dims=H, loss="log_cosh")

print("\nwrote", len(list(HERE.glob('scan*.yaml'))), "configs ->", HERE)
