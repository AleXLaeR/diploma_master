"""
Apache Airflow DAG for Phase 2 — Predictive Modeling Layer (Micro/DDA).

ROCV Architecture
-----------------
This DAG runs once per fold. The fold boundary is passed via DAG Run
configuration (dag_run.conf). Trigger it four times — once for each fold
defined in models/rocv.py — either manually or via an orchestrating DAG.

Expected dag_run.conf keys:
    fold_id        : str  — e.g. 'fold_1'
    train_end      : str  — exclusive upper bound for training (YYYY-MM-DD)
    holdout_start  : str  — inclusive lower bound for holdout  (YYYY-MM-DD)
    holdout_end    : str  — exclusive upper bound for holdout  (YYYY-MM-DD)

Execution order (within a fold):
    [init_eval_tables, init_dda_weights]
      → [Baseline_LastClick, Markov_DDA, Shapley_DDA]       (parallel)
        → [trigger_survival_models, trigger_mmm_models]     (parallel)

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
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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
    """
    Extract fold boundary params from dag_run.conf.
    Falls back to fold_4 defaults for manual ad-hoc runs without conf.
    """
    conf = kwargs.get("dag_run", {})
    if hasattr(conf, "conf") and conf.conf:
        c = conf.conf
    else:
        # Fallback: full training split (fold_4) — allows manual test runs
        logger.warning(
            "dag_run.conf not set. Defaulting to fold_4 boundaries. "
            "Pass conf={fold_id, train_end, holdout_start, holdout_end} for ROCV."
        )
        c = {
            "fold_id":       "fold_4",
            "train_end":     "2021-12-01",
            "holdout_start": "2021-12-01",
            "holdout_end":   "2022-03-01",
        }
    return {
        "fold_id":       c.get("fold_id",       "fold_4"),
        "train_end":     c.get("train_end",     "2021-12-01"),
        "holdout_start": c.get("holdout_start", "2021-12-01"),
        "holdout_end":   c.get("holdout_end",   "2022-03-01"),
    }


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _run_baseline_last_click(**kwargs) -> None:
    """
    Last-Click baseline: execute fold-scoped SQL, translate weights to
    weekly holdout forecasts, write to eval_dda.
    """
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from models.attribution.dda_common_new import (
        get_bq_client,
        translate_weights_to_forecasts_ewma,
        write_forecasts_to_bq,
        write_weights_to_bq,
    )

    fold = _get_fold_conf(kwargs)
    client = get_bq_client()

    # Render SQL with fold-specific train_end to prevent leakage
    sql = _render_sql("dda_last_click.sql", extra={"train_end": fold["train_end"]})
    df = client.query(sql).to_dataframe()

    if df.empty:
        logger.error("Last-Click query returned 0 rows — aborting")
        return

    weights = dict(zip(df["media_source"], df["weight"]))
    logger.info("Last-Click weights fold=%s: %s", fold["fold_id"], weights)

    forecast_df = translate_weights_to_forecasts_ewma(
        weights,
        model_name="Baseline_LastClick",
        client=client,
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
        confidence_weight=0,
    )

    write_forecasts_to_bq(
        client, forecast_df, BQ_PROJECT, BQ_DATASET,
        model_name="Baseline_LastClick",
    )
    write_weights_to_bq(
        client, weights, BQ_PROJECT, BQ_DATASET,
        model_name="Baseline_LastClick",
        fold_id=fold["fold_id"],
    )
    logger.info("Baseline Last-Click fold=%s complete ✓", fold["fold_id"])


def _run_markov(**kwargs) -> None:
    """Markov DDA entry point for Airflow (fold-aware)."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from models.attribution.dda_markov_new import run
    fold = _get_fold_conf(kwargs)
    run(
        bq_project=BQ_PROJECT,
        bq_dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
    )


def _run_shapley(**kwargs) -> None:
    """Shapley DDA entry point for Airflow (fold-aware)."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from models.attribution.dda_shapley_new import run
    fold = _get_fold_conf(kwargs)
    run(
        bq_project=BQ_PROJECT,
        bq_dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
    )


with DAG(
    dag_id="phase2_dda_models",
    description="Phase 2 — Micro-Analytical Plane (DDA) + triggers | One run per ROCV fold",
    default_args=DAG_DEFAULT_ARGS,
    schedule=None,
    start_date=datetime(2021, 4, 1),
    catchup=False,
    tags=["phase-2", "init", "predictive-models", "dda", "rocv"],
    max_active_runs=2,
) as dag:

    task_init_eval_tables = BigQueryInsertJobOperator(
        task_id="init_eval_tables",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("final/init_eval_tables.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_init_dda_weights = BigQueryInsertJobOperator(
        task_id="init_dda_weights",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("final/init_dda_weights.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_model_baseline_lc = PythonOperator(
        task_id="model_baseline_last_click",
        python_callable=_run_baseline_last_click,
    )

    task_model_markov = PythonOperator(
        task_id="model_markov_dda",
        python_callable=_run_markov,
    )

    task_model_shapley = PythonOperator(
        task_id="model_shapley_dda",
        python_callable=_run_shapley,
    )

    # Trigger Downstream DAGs (pass fold conf forward)
    trigger_survival_models = TriggerDagRunOperator(
        task_id="trigger_survival_models",
        trigger_dag_id="phase2_survival_models",
        wait_for_completion=False,
        conf={
            "fold_id": "{{ (dag_run.conf or {}).get('fold_id', 'fold_4') }}",
            "train_end": "{{ (dag_run.conf or {}).get('train_end', '2021-12-01') }}",
            "holdout_start": "{{ (dag_run.conf or {}).get('holdout_start', '2021-12-01') }}",
            "holdout_end": "{{ (dag_run.conf or {}).get('holdout_end', '2022-03-01') }}",
        },
    )

    trigger_mmm_models = TriggerDagRunOperator(
        task_id="trigger_mmm_models",
        trigger_dag_id="phase2_mmm_models",
        wait_for_completion=False,
        conf={
            "fold_id": "{{ (dag_run.conf or {}).get('fold_id', 'fold_4') }}",
            "train_end": "{{ (dag_run.conf or {}).get('train_end', '2021-12-01') }}",
            "holdout_start": "{{ (dag_run.conf or {}).get('holdout_start', '2021-12-01') }}",
            "holdout_end": "{{ (dag_run.conf or {}).get('holdout_end', '2022-03-01') }}",
        },
    )

    # Execution Flow
    dda_models = [task_model_baseline_lc, task_model_markov, task_model_shapley]
    downstream_triggers = [trigger_survival_models, trigger_mmm_models]
    initial_tables = [task_init_eval_tables, task_init_dda_weights]

    for create_task in initial_tables:
        create_task >> dda_models
    for trigger in downstream_triggers:
        dda_models >> trigger
