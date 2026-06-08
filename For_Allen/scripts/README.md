# `scripts/`

Thin entry-points that load a config, call into `src/for_allen/`, and
write artifacts to `artifacts/<phase>/<run-id>/`. Every script is
guarded — it asserts on environment, git pin, and MLflow tags before
doing anything destructive.

| Script | Phase | Function |
|--------|------:|----------|
| `capture_host.py` | 0 | Generates `pins/host.txt`. |
| `hash_data.py` | 0 | Computes SHA-256 of every data artifact and writes `pins/data_manifests/`. |
| `sanity_check.py` | every | Runs the 6-test smoke battery on a checkpoint (`--toy` mode for Phase 0). |
| `train.py` | 2a, 2b | Loads a config, refuses to start without mandatory MLflow tags, runs training, runs sanity battery at every save. |
| `eval_stage1.py` | 1a, 2a, 2b, 4 | Per-cell stratified VELO/UT/SciFi gates with bootstrap CIs. |
| `eval_a4.py` | 1a, 2b | Frobenius + max-off-diagonal A4 against cached RK45 reference. |
| `eval_bwd_fwd.py` | 1a, 2b | bwd/fwd ratio per cell. |
| `export_bin.py` | 3 | Exports a checkpoint to V3 `.bin` with full manifest stamping. |
| `compare_runs.py` | every | MLflow diff helper: prints metric deltas between two runs with their tags. |

Every script begins with:

```python
from for_allen.tracking.check_tags import assert_required_tags
from for_allen.data.manifest_check import assert_pinned_data
assert_pinned_data(cfg)        # SHA matches pins/data_manifests/
assert_required_tags(cfg)      # MLflow tag set is complete
```

A script that does not perform these two assertions does not get merged.
