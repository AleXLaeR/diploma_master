"""
survival_bdw_new entry point — Airflow callable.
"""
from __future__ import annotations

from models.survival.survival_bdw_new import BDWModel
from models.survival.survival_base_new import execute_survival_model


def run_bdw(
    bq_project: str,
    bq_dataset: str,
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
) -> None:
    """Entry point for Airflow. Called with fold-specific parameters from DAG conf."""
    execute_survival_model(
        bq_project    = bq_project,
        bq_dataset    = bq_dataset,
        model_name    = "BdW",
        model_class   = BDWModel,
        bounds        = [[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]],
        fold_id       = fold_id,
        train_end     = train_end,
        holdout_start = holdout_start,
        holdout_end   = holdout_end,
    )
