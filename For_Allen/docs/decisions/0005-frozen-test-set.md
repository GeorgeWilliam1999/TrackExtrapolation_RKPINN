# ADR 0005 — The M1 test set is frozen as `test_v1_frozen.npy`

* **Date:** 2026-05-08
* **Status:** accepted
* **Context:** The M1 deep-dive evaluated all stage-1, A4, and ABI gates on the test indices recorded in `test_indices.npy`. To compare future runs against M1 fairly, that exact set must remain untouched. But that set must **not** be used for hyperparameter tuning in Phase 2 — that would be selection-bias contamination, and once a split has been used for HP search it is no longer a clean acceptance set.
* **Decision:** Treat the existing M1 test set as an **immutable, never-touched acceptance set**:
  * Renamed `test_v1_frozen.npy`.
  * SHA-256 pinned in `pins/data_manifests/test_v1_frozen.sha256`.
  * Read-only (file-permission `444`) once pinned.
  * **Used only at gate decisions:** end-of-Phase-1a, end-of-Phase-2b, end-of-Phase-4. Not for HP tuning, not for early-stopping, not for checkpoint selection during training.
  * A separate, event-grouped (per ADR 0004) train/val/test is generated in Phase 0 from the full 10 M corpus and used for all development.
* **Consequences:**
  * Phase 2's "best checkpoint" is selected by the *new* val set, not by `test_v1_frozen.npy`.
  * `test_v1_frozen.npy` is reported on, in full, exactly once per phase boundary (1a → 2b → 4) and recorded in MLflow with the phase tag.
  * If `test_v1_frozen.npy` is found to have been touched (its SHA changes), every PASS claim that referenced it is invalidated.
