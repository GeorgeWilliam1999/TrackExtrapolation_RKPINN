# `configs/`

YAML config per run. One file per run, named with the phase and a short
descriptor: `phase2b_full_10M/seed0.yaml` etc. Every config is loaded by
`scripts/train.py`, `scripts/eval_*.py`, etc., and is reproducible from
its git SHA.

Each config must include:

```yaml
phase: "2b"
purpose: "final"          # one of {calibration, sweep, final, sanity}
seed:
  numpy: 0
  torch: 0
  cuda: 0
  dataloader: 0
  pythonhashseed: 0
data:
  train_sha: "..."        # must match pins/data_manifests/train_10M.sha256
  val_sha:   "..."
  test_sha:  "..."        # test_v1_frozen for gate runs, test_v2 for dev
  splitter_sha: "..."
model:
  family: "nrk4"
  hidden_dims: [64, 64]
  n_rk_steps: 2           # from pins/n_rk_steps_prod.txt (ADR 0007, 2026-05-12)
  corrector_enabled: false
  precision: "fp32"
loss:
  recipe: "sigma_weighted_endpoint+jacobian_reg"
  lambda_jacobian: 0.01
  sigma_velo:  [12.0e-6, 12.0e-6, ...]   # per-coord, units consistent with state
  sigma_ut:    [50.0e-6, 50.0e-6, ...]
optim:
  lr: 1.0e-3
  wd: 1.0e-4
  batch_size: 4096
  epochs: 60
  scheduler: "cosine"
mlflow:
  experiment: "for_allen"
  tags_required: true     # tracking/check_tags.py refuses if any mandatory tag is missing
```

The list of mandatory MLflow tags is enforced by
`src/for_allen/tracking/check_tags.py`, not by this directory.
