# `pins/`

Pinned values that any acceptance claim depends on. Anything in here
is committed to git. Hashes are over the canonical artifact, never the
artifact itself.

| File | Source | Phase |
|------|--------|-------|
| `allen_commit.txt` | `git -C /data/bfys/gscriven/Allen rev-parse HEAD` | 0 |
| `protocol_sha.txt` | git SHA of `docs/reports/gen3_protocol.tex` | 0 |
| `host.txt` | output of `scripts/capture_host.py` | 0, re-run on host change |
| `loader_v3_spec.md` | the manifest schema, locked at end of Phase 1b | 1b |
| `n_rk_steps_prod.txt` | the Phase-1a-winner step count (= `2` as of 2026-05-12, see [ADR 0007](../docs/decisions/0007-phase1a-winner.md)) | 1a |
| `baseline_throughput.txt` | RKN4 throughput + per-track cost from Allen CI job 75148540 (or latest retry); written by hand after reading the CI log. Target for A6 gate in Phase R6. | pre-flight |
| `moore_commit.txt` | Moore git SHA used in Phase 7 | 7 |
| `gitconddb_tag.txt`, `dddb_tag.txt` | conditions tags compatible with the Allen pin | 5 |
| `data_manifests/*.sha256` | one per data artifact (train, val, test_v1_frozen, test_v2_event_grouped, field_map) | 0 |
| `data_manifests/qop_bins.txt` | percentile bin edges for the stratified gates | 0 |
| `env_gpu.yml` | conda env on the GPU host (Phase 5+) | 5 |
