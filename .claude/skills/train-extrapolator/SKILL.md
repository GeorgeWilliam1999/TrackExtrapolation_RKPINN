---
name: train-extrapolator
description: Train a gen-3 neural track extrapolator (PINN_v2 or MLP) on the LHCb corpus via HTCondor, on GPU or CPU. Use when asked to train, retrain, launch a training run, sweep, or add a model variant for the Track Extrapolation project.
---

# Train a neural track extrapolator

Working dir: `/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3`.
Python env (PyTorch 2.9 + cu128): `/data/bfys/gscriven/conda/envs/TE/bin/python`.

## 1. Write a config (yaml)

Configs live in `configs/`. A run is fully specified by one yaml; `models/train.py` reads it via `--config`. Key keys (see `configs/pinn_v2_kick_10M.yaml` for a full example):

```yaml
model_type: pinn_v2            # or "mlp"
hidden_dims: [96, 96]
activation: tanh
lambda_pde: 0.1                # PINN_v2 only
lambda_ic: 0.01
n_collocation: 2
physics_warmup_epochs: 10
# --- model-improvement flags (PINN_v2; default off = locked candidate) ---
kick_scaled_head: true         # couple correction to qop*dz magnet kick (q/p-bias fix)
pde_scale_mode: fixed_L        # "legacy" | "fixed_L"
pde_ref_length: 5213.0
# --- data / training ---
data_path: /data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/data/train_10M_gen3.npz
max_samples: 10000000          # subsample of the 10M corpus; loader loads full npz then slices
batch_size: 4096
epochs: 80
learning_rate: 0.0005
weight_decay: 0.0001
warmup_epochs: 5
loss: log_cosh                 # ALWAYS log-cosh; selection is on val_median_dx (Fix L1)
checkpoint_dir: /data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/trained_models
experiment_name: my_run
use_mlflow: true
mlflow_experiment_name: gen_3_track_extrapolation
device: cuda                   # cuda for GPU submit, cpu for CPU submit (see gotcha)
num_workers: 4
pin_memory: true               # true for cuda, false for cpu
```

New PINN_v2 kwargs must also be threaded through `models/train.py::_build_model` (the pinn_v2 branch passes explicit kwargs, not **config) and `models/architectures.py::PINN_v2.__init__`.

## 2. Submit to HTCondor

- **CPU:** `condor_submit condor/train.sub` (4 cores, 8 GB; edit the `queue ... from (...)` list or use a dedicated sub like `condor/train_cpu_kick.sub`). CPU is fine for <=2M samples (a few hours); 10M on CPU is ~1-2 days.
- **GPU:** `condor_submit condor/train_gpu.sub`. Use for the full 10M corpus.

Both call `condor/run_train.sh <config> <tag>`, which activates the TE env and runs `train.py --config`.

## 3. GPU condor gotchas (learned the hard way 2026-06-09)

This Nikhef/stoomboot pool does NOT advertise `CUDACapability` — requiring it matches **0** machines. The correct attribute is `GPUs_Capability` (all GPU nodes are V100-32GB, cap 7.0). **In `train_gpu.sub`: do NOT add a `requirements = (CUDACapability >= 7.0)` line** — just `request_gpus = 1` (condor auto-adds `TARGET.GPUs >= 1`).

Also: the free GPU slots don't have many spare CPUs, so a large request matches 0 ("would match if drained"). **Request small: `request_cpus = 1`, `request_memory = 8 GB`.** GPU nodes share `FileSystemDomain = stoomboot.nikhef.nl`, so `should_transfer_files = NO` + shared `/data` works.

Diagnose a stuck idle job with: `condor_q <id> -better-analyze` (look at which requirement reduces to 0 matched slots) and `condor_status -constraint 'TotalGPUs > 0' -af Machine TotalGPUs State`.

The config's `device` is honoured literally (`torch.device(config["device"])`): `device: cuda` on a CPU node **crashes** at `model.to('cuda')`. Match `device` to the submit.

## 4. Monitor + outputs

```bash
condor_q <cluster>
tail -f condor/logs/<tag>.<cluster>.<proc>.out      # per-epoch median_dx / p95 lines
# MLflow: mlflow ui --backend-store-uri .../experiments/gen_3/mlruns
```
Outputs land in `trained_models/<experiment_name>/`: `best_model.pt` (selected on `val_median_dx`), `config.json`, `normalization.json`, `history.json`, `test_indices.npy`. Watch for late-epoch val rise — best checkpoint is the early minimum, which is correct.

## 5. After training → evaluate

Run the `evaluate-checkpoint` skill (A4 Jacobian gate + UT→T Split-B). A model is only a re-lock candidate if it beats `pinn_v2_small_v1` on Split-B median AND passes A4.
