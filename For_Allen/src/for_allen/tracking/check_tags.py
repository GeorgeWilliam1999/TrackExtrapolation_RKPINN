"""tracking/check_tags.py — refuse-to-start guard for MLflow runs.

Every script that calls mlflow.start_run() must first call
assert_required_tags(cfg). The function inspects the resolved tag set
that the config + environment will produce and raises if any mandatory
tag is missing or empty.

The list of mandatory tags is defined here, in one place, so a new
required tag is a one-line change.
"""
from __future__ import annotations

from typing import Iterable, Mapping

REQUIRED_TAGS: tuple[str, ...] = (
    "git_sha",
    "git_dirty",
    "allen_commit",
    "protocol_sha",
    "data_train_sha",
    "data_val_sha",
    "data_test_sha",
    "splitter_sha",
    "host",
    "gpu_model",
    "cuda_version",
    "torch_version",
    "cudnn_version",
    "seed_numpy",
    "seed_torch",
    "seed_cuda",
    "seed_dataloader",
    "pythonhashseed",
    "n_rk_steps",
    "corrector_enabled",
    "loss_recipe",
    "precision",
    "phase",
    "purpose",
)


class MissingTagError(RuntimeError):
    pass


def assert_required_tags(tags: Mapping[str, object], extra_required: Iterable[str] = ()) -> None:
    """Raise MissingTagError if any mandatory tag is missing or empty.

    `tags` is the dict that will be passed to mlflow.set_tags(). `extra_required`
    can be used by per-phase callers to add phase-specific requirements.
    """
    required = set(REQUIRED_TAGS) | set(extra_required)
    missing = [k for k in required if k not in tags or tags[k] in (None, "")]
    if missing:
        raise MissingTagError(
            "Refusing to start MLflow run; missing mandatory tags: " + ", ".join(sorted(missing))
        )
