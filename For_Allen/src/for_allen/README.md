# `src/for_allen/`

Library code. **No** notebook may import from anywhere else, and no
script under `scripts/` may bypass this layer.

| Module | Responsibility |
|--------|----------------|
| `models/` | `nrk4` family: RHS MLP, the (now-disabled) corrector class, model factory from a config. |
| `integrators/` | `rk4_n_step.py` (fixed step count), `rk4_adaptive.py`. The Phase-1a winner is selected by config. |
| `losses/` | `endpoint_sigma.py` (Fix I), `jacobian_reg.py` (Fix J via `torch.func.jacfwd` against a cached RK45 Jacobian). |
| `data/` | event-grouped `splitter.py`, `dataloader.py`, `manifest_check.py` (refuses to load a corpus whose SHA doesn't match `pins/`). |
| `export/` | `manifest.py` (V3 schema reader/writer), `bin_v3.py` (binary writer + Python mirror loader for round-trip tests). |
| `sanity/` | The 6 cheap regressions from `PLAN.md` §"Per-checkpoint smoke battery". Pure functions over a model + a small input. |
| `eval/` | `stage1.py` (per-cell stratified VELO/UT/SciFi gates with bootstrap CIs), `a4.py` (Frobenius + max-off-diagonal), `bwd_fwd.py`, `bootstrap.py` (BCa). |
| `tracking/` | `check_tags.py` — refuses to start a training run unless every mandatory MLflow tag is present. |

Style: type-hinted, `numpy`/`torch` boundaries explicit, no global state,
no module-level GPU side effects (no `torch.cuda.set_device` outside
`scripts/`). Tests live under `tests/` and run on a 1 k-param toy model.
