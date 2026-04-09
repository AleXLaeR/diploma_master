"""
dag_phase2_survival.py
======================
Apache Airflow DAG for Phase 2 — Predictive Modeling Layer (Survival).

ROCV Architecture
-----------------
Triggered automatically by phase2_dda_models with fold conf forwarded.
Reads fold boundaries from dag_run.conf (same keys as DDA DAG).

Spec reference: docs/orchestration_guidelines.md §2
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryInsertJobOperator,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BQ_PROJECT = os.environ.get("BQ_PROJECT", "{{ BQ_PROJECT }}")
BQ_DATASET = os.environ.get("BQ_DATASET", "{{ BQ_DATASET }}")
BQ_CONN_ID = os.environ.get("BQ_CONN_ID", "google_cloud_default")

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

DAG_DEFAULT_ARGS = {
    "owner": "thesis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30)
}

logger = logging.getLogger(__name__)


def _render_sql(filename: str, extra: dict | None = None) -> str:
    """Read a .sql file from the sql/ directory and apply substitution."""
    raw = (SQL_DIR / filename).read_text(encoding="utf-8")
    rendered = raw.replace("{{ project }}", BQ_PROJECT).replace("{{ dataset }}", BQ_DATASET)
    if extra:
        for key, val in extra.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", val)
    return rendered


def _get_fold_conf(kwargs: dict) -> dict:
    """Extract fold boundary params from dag_run.conf (with fold_4 fallback)."""
    conf = kwargs.get("dag_run", {})
    c = conf.conf if hasattr(conf, "conf") and conf.conf else {}
    return {
        "fold_id":       c.get("fold_id",       "fold_4"),
        "train_end":     c.get("train_end",     "2021-12-01"),
        "holdout_start": c.get("holdout_start", "2021-12-01"),
        "holdout_end":   c.get("holdout_end",   "2022-03-01"),
    }


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _run_sbg(**kwargs) -> None:
    """sBG Survival Prediction entry point for Airflow (fold-aware)."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from models.survival.run_sbg_new import run_sbg
    fold = _get_fold_conf(kwargs)
    run_sbg(
        bq_project=BQ_PROJECT,
        bq_dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
    )


def _run_bdw(**kwargs) -> None:
    """BdW Survival Prediction entry point for Airflow (fold-aware)."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from models.survival.run_bdw_new import run_bdw
    fold = _get_fold_conf(kwargs)
    run_bdw(
        bq_project=BQ_PROJECT,
        bq_dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
    )


def _run_survival_baseline(**kwargs) -> None:
    """Survival Baseline regression entry point for Airflow (fold-aware)."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from models.survival.run_baseline_new import run_baseline
    fold = _get_fold_conf(kwargs)
    run_baseline(
        bq_project=BQ_PROJECT,
        bq_dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="phase2_survival_models",
    description="Phase 2 — Customer Base Plane (Survival Models) | One run per ROCV fold",
    default_args=DAG_DEFAULT_ARGS,
    schedule=None,          # Triggered automatically by phase2_dda_models
    start_date=datetime(2021, 4, 1),
    catchup=False,
    tags=["phase-2", "predictive-models", "survival", "rocv"],
    max_active_runs=1
) as dag:
    task_init_survival_model_params = BigQueryInsertJobOperator(
        task_id="init_survival_model_params",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("final/survival/init_survival_model_params.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_init_survival_monetary_params = BigQueryInsertJobOperator(
        task_id="init_survival_monetary_params",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("final/survival/init_survival_monetary_params.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_model_sbg = PythonOperator(
        task_id="model_sbg",
        python_callable=_run_sbg,
    )

    task_model_bdw = PythonOperator(
        task_id="model_bdw",
        python_callable=_run_bdw,
    )

    task_model_survival_baseline = PythonOperator(
        task_id="model_survival_baseline",
        python_callable=_run_survival_baseline,
    )

    table_tasks = [task_init_survival_model_params, task_init_survival_monetary_params]
    model_tasks = [task_model_sbg, task_model_bdw, task_model_survival_baseline]

    for init_task in table_tasks:
        init_task >> model_tasks
