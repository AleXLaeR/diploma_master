"""Evaluation and visualization entrypoints for Phase 3."""

import os
from pathlib import Path
import sys


def run(*args, **kwargs):
    from models.evaluation.runner import run as _run

    return _run(*args, **kwargs)

__all__ = ["run"]

# Allow direct script execution:
#   python models/evaluation/runner.py
# by ensuring project root is importable.
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

if __name__ == "__main__":
    run(
        bq_project=os.getenv("BQ_PROJECT"),
        bq_dataset=os.getenv("BQ_DATASET"),
    )
