# ADR 0004 — Train/val/test splits are event-grouped

* **Date:** 2026-05-08
* **Status:** accepted
* **Context:** The Gen-3 corpus contains many tracks per simulated event. Tracks within an event share:
  * the same magnetic-field sampling along their trajectory (the same field map, but more importantly the same realisation of any noise / multipoles),
  * material-budget realisation,
  * pile-up / occupancy state,
  * trigger / topology bias.
  A track-level random split therefore leaks event-level correlations across train/val/test. The reviewer (audit b.2) flagged this as a leakage risk that potentially contaminates the §12 corrector-ablation conclusion if the M1 split was track-level.
* **Decision:** All splits in this workspace are **event-grouped**:
  * `event_id` is hashed → bucket; tracks inherit their event's bucket.
  * The splitter (`src/for_allen/data/splitter.py`) refuses to operate on a corpus that lacks a per-track `event_id`.
  * The exact split is a function of `(corpus_sha, splitter_sha, seed)` and is reproducible from those three values.
* **Consequences:**
  * The existing M1 `test_indices.npy` is audited at start of Phase 0. If it was track-level, the §12 corrector ablation must be re-run on an event-grouped split before its conclusion is trusted as motivation for ADR 0002.
  * Phase 0 ships a new splitter and produces a new event-grouped train/val/test for Phase 2 development.
  * The frozen M1 set is preserved separately under ADR 0005.
