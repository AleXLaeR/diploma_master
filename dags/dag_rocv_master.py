"""
Master Orchestrator for Phase 2 ROCV Pipeline.

This DAG dynamically iterates over the FOLDS dictionary and spawns 4 parallel 
instances of `phase2_dda_models`. Each triggered instance will run its fold-specific
DDA algorithms natively, and then cascade down to survival and MMM models by
forwarding the `dag_run.conf` automatically.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.rocv import FOLDS

DAG_DEFAULT_ARGS = {
    "owner": "thesis",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="rocv_master_orchestrator",
    description="Trigger Phase 2 Pipeline for all ROCV folds simultaneously",
    default_args=DAG_DEFAULT_ARGS,
    schedule=None,
    start_date=datetime(2021, 4, 1),
    catchup=False,
    tags=["phase-2", "rocv", "master"],
) as dag:

    for fold_id, fold_boundaries in FOLDS.items():
        if fold_id != "fold_4":
            continue

        conf_payload = {
            "fold_id": fold_id,
            "train_end": fold_boundaries["train_end"],
            "holdout_start": fold_boundaries["holdout_start"],
            "holdout_end": fold_boundaries["holdout_end"],
        }

        # Trigger phase2_dda_models with the conf
        TriggerDagRunOperator(
            task_id=f"trigger_dda_{fold_id}",
            trigger_dag_id="phase2_dda_models",
            wait_for_completion=False,
            conf=conf_payload,
        )
