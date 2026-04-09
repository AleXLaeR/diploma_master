"""
rocv.py
=======
Rolling-Origin Cross-Validation (ROCV) configuration.

Defines the 4-fold expanding-window boundaries used by ALL model planes
(DDA, MMM, Survival). Every model's run() entry-point must accept
fold_id and derive its temporal split from this module.

Spec reference: docs/comparison/comparison_framework.md — ROCV section
"""

from __future__ import annotations

from typing import TypedDict


class FoldSpec(TypedDict):
    train_end: str       # exclusive upper bound for training data  (date < train_end)
    holdout_start: str   # inclusive lower bound for holdout data   (date >= holdout_start)
    holdout_end: str     # exclusive upper bound for holdout data   (date < holdout_end)


# ---------------------------------------------------------------------------
# Canonical fold definitions (4-fold monthly expanding window)
# ---------------------------------------------------------------------------
# Fold 1: Train Apr–Aug (5 mo) | Holdout Sep–Nov  (3 mo)
# Fold 2: Train Apr–Sep (6 mo) | Holdout Oct–Dec  (3 mo)
# Fold 3: Train Apr–Oct (7 mo) | Holdout Nov–Jan  (3 mo)
# Fold 4: Train Apr–Nov (8 mo) | Holdout Dec–Feb  (3 mo)
#
# All date strings are ISO-8601 in BigQuery-compatible format.
# "exclusive upper bound" means the SQL predicate is: date < bound.
# ---------------------------------------------------------------------------
FOLDS: dict[str, FoldSpec] = {
    "fold_1": {
        "train_end":     "2021-09-01",
        "holdout_start": "2021-09-01",
        "holdout_end":   "2021-12-01",
    },
    "fold_2": {
        "train_end":     "2021-10-01",
        "holdout_start": "2021-10-01",
        "holdout_end":   "2022-01-01",
    },
    "fold_3": {
        "train_end":     "2021-11-01",
        "holdout_start": "2021-11-01",
        "holdout_end":   "2022-02-01",
    },
    "fold_4": {
        "train_end":     "2021-12-01",
        "holdout_start": "2021-12-01",
        "holdout_end":   "2022-03-01",
    },
}

FOLD_IDS: list[str] = list(FOLDS.keys())  # ["fold_1", "fold_2", "fold_3", "fold_4"]


def get_fold(fold_id: str) -> FoldSpec:
    """Return the FoldSpec for the given fold_id. Raises KeyError if invalid."""
    if fold_id not in FOLDS:
        raise KeyError(
            f"Unknown fold_id '{fold_id}'. Valid options: {FOLD_IDS}"
        )
    return FOLDS[fold_id]
