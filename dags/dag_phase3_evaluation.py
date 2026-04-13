"""
dag_phase3_evaluation.py
========================
Apache Airflow DAG for Phase 3 — Consolidation, Evaluation & Visualization.

This DAG is triggered manually AFTER Phase 2 ROCV DAGs complete.
It assumes all predictive models have written their forecasts to
``eval_dda``, ``eval_mmm``, and ``eval_survival`` in BigQuery.

Pipeline topology:
  3 parallel consolidation SQL tasks (fill actual_net_revenue_usd)
      ↓
  task_evaluate_and_visualize (compute metrics + render F1–F9 charts)

Spec references:
    docs/orchestration_guidelines.md §2
    docs/comparison/comparison_framework.md §1–§2
    docs/data/final/consolidation_phase.md
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
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
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: read and template a SQL file
# ---------------------------------------------------------------------------

def _render_sql(filename: str) -> str:
    """Read a .sql file from the sql/ directory and apply substitution."""
    raw = (SQL_DIR / filename).read_text(encoding="utf-8")
    return raw.replace("{{ project }}", BQ_PROJECT).replace("{{ dataset }}", BQ_DATASET)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="phase3_evaluation",
    description="Phase 3 — Consolidation, scoring & thesis visualization",
    default_args=DAG_DEFAULT_ARGS,
    schedule=None,          # Triggered manually after Phase 2 completes
    start_date=datetime(2021, 4, 1),
    catchup=False,
    tags=["phase-3", "evaluation", "visualization"],
) as dag:
    task_init_portfolio_budget_simulation = BigQueryInsertJobOperator(
        task_id="init_portfolio_budget_simulation",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("final/init_portfolio_budget_simulation.sql"),
                "useLegacySql": False,
            }
        },
    )

    # ===================================================================
    # Phase 3.1: Consolidation Layer (Impartial Judge)
    # Three parallel MERGE queries that populate factuals (actual_net_revenue_usd, etc.)
    # in the domain-specific eval tables from the holdout purchases data.
    # Spec: docs/data/final/consolidation_phase.md
    # ===================================================================

    task_consolidate_survival = BigQueryInsertJobOperator(
        task_id="consolidate_survival_factuals",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("consolidation/consolidate_survival.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_consolidate_dda = BigQueryInsertJobOperator(
        task_id="consolidate_dda_factuals",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("consolidation/consolidate_dda.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_consolidate_mmm = BigQueryInsertJobOperator(
        task_id="consolidate_mmm_factuals",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("consolidation/consolidate_mmm.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_consolidate_channel_contribs = BigQueryInsertJobOperator(
        task_id="consolidate_mmm_channel_contribs_factuals",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("consolidation/consolidate_channel_contribs.sql"),
                "useLegacySql": False,
            }
        },
    )

    # ===================================================================
    # Phase 3.2: Evaluation & Visualization (Terminal Task)
    # Fetches the completed comparison table, computes global error
    # metrics and renders thesis-ready F1–F9 charts
    # to the local reports/ directory.
    # Spec: docs/comparison/comparison_framework.md §1–§2
    # ===================================================================

    def _run_evaluate_and_visualize(**kwargs) -> None:
        """Evaluation & visualization entry point for Airflow."""
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from models.evaluation import run
        run(bq_project=BQ_PROJECT, bq_dataset=BQ_DATASET)

    task_evaluate_and_visualize = PythonOperator(
        task_id="evaluate_and_visualize",
        python_callable=_run_evaluate_and_visualize,
    )

    # ===================================================================
    # Dependency wiring:
    # 3 parallel consolidation tasks → terminal evaluation task
    # ===================================================================
    consolidation_tasks = [
        task_consolidate_survival,
        task_consolidate_dda,
        task_consolidate_mmm,
        task_consolidate_channel_contribs,
    ]
    task_init_portfolio_budget_simulation >> consolidation_tasks
    consolidation_tasks >> task_evaluate_and_visualize
