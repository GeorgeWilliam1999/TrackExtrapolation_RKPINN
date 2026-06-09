---
name: evaluate-checkpoint
description: Evaluate a trained track-extrapolator checkpoint with the A4 Jacobian gate and the UT->T Split-B accuracy eval, comparing to the locked candidate. Use after training a model, or when asked to evaluate/gate/compare a checkpoint.
---

# Evaluate a checkpoint (A4 Jacobian + UT->T Split-B)

Working dir: `/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3`. Env: `/data/bfys/gscriven/conda/envs/TE/bin/python`. These run on CPU (login host has no GPU) in a couple of minutes.

The two gates that decide a re-lock:
1. **A4 Jacobian** — `run_r2_jacobian.py`. fp64-autograd Jacobian vs the cached RK45 reference (`For_Allen/artifacts/phase1a/J_rk4_reference.npy`, `X_a4.npy`). Gate: Frobenius rel-err < 0.05.
2. **UT->T Split-B accuracy** — `run_r7_utt_eval.py`. Masks each model's test split to the UT->T window (z_start ∈ [2300,3000], z_end ∈ [7600,9500], dz>0), reports median/p68/p95/p99 |dx| and the **median-by-|q/p|-quartile** (the q/p bias shows as rising quartiles).

## How to add your run to the eval

Both scripts have a **hardcoded candidate list** in `main()`. Add your run dir, keeping `pinn_v2_small_v1` for comparison:
```python
("pinn_v2_small_v1",   GEN3_ROOT/"trained_models"/"pinn_v2_small_v1"),
("my_run",             GEN3_ROOT/"trained_models"/"my_run"),
```

## CRITICAL: the loader must honour PINN_v2 flags

Both scripts rebuild the model from `config.json`. The `_load_checkpoint` / `_load` pinn_v2 branch MUST pass the kick flags or a kick checkpoint (extra `kick_loggain` param) fails to load:
```python
model = create_model("pinn_v2", hidden_dims=cfg["hidden_dims"], activation=cfg["activation"],
                     dropout=cfg.get("dropout",0.0), lambda_pde=cfg.get("lambda_pde",0.1),
                     lambda_ic=cfg.get("lambda_ic",0.1), n_collocation=cfg.get("n_collocation",2),
                     kick_scaled_head=cfg.get("kick_scaled_head", False),
                     pde_scale_mode=cfg.get("pde_scale_mode", "legacy"),
                     pde_ref_length=cfg.get("pde_ref_length", 5213.0))
```
(Already fixed in the current scripts; preserve it.)

## Run

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY run_r2_jacobian.py     # A4: per-model frob_mean/p95, PASS/FAIL
$PY run_r7_utt_eval.py      # UT->T: median/p95 + by-|q/p|-quartile; writes results/R7_utt_eval_<date>.json
```

## Interpret (the reference numbers to beat)

Locked `pinn_v2_small_v1`: Split-A median 11.7 µm; **UT->T Split-B median 287 µm**; A4 frob 9e-4; by-|q/p|-quartile [175, 219, 383, 487] (rising = the bias). A re-lock candidate must **beat 287 µm on Split-B AND pass A4 (< 0.05)**. Watch the **highest-|q/p| quartile** and **p95** — the 2026-06 kick head halved the median but worsened the low-momentum (high-|q/p|) tail; the median alone is not sufficient.
