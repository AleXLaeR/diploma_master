"""
dag_phase1_datasets.py
======================
Phase 1 ELT pipeline:
1. Build ROCV folds.
2. Build and populate users_attribution_imputed.
3. Build intermediate datasets.
4. Build model-specific marts.
5. Trigger DDA phase.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

BQ_PROJECT = os.environ.get("BQ_PROJECT", "{{ BQ_PROJECT }}")
BQ_DATASET = os.environ.get("BQ_DATASET", "{{ BQ_DATASET }}")
BQ_CONN_ID = os.environ.get("BQ_CONN_ID", "google_cloud_default")

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

DAG_DEFAULT_ARGS = {
    "owner": "thesis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30)
}


def _render_sql(filename: str) -> str:
    raw = (SQL_DIR / filename).read_text(encoding="utf-8")
    return raw.replace("{{ project }}", BQ_PROJECT).replace("{{ dataset }}", BQ_DATASET)


with DAG(
    dag_id="phase_1_marketing_analytics_pipeline",
    description="Phase 1: fold setup, attribution imputation, and marts",
    default_args=DAG_DEFAULT_ARGS,
    schedule=None,
    start_date=datetime(2021, 4, 1),
    catchup=False,
    tags=["phase-1", "datasets"],
) as dag:
    task_create_rocv_folds = BigQueryInsertJobOperator(
        task_id="create_rocv_folds",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("intermediate/create_rocv_folds.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_copy_users_attribution_base = BigQueryInsertJobOperator(
        task_id="copy_users_attribution_imputed_base",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("copy_users_attribution.sql"),
                "useLegacySql": False,
            }
        },
    )

    def _run_users_attribution_imputation(**kwargs) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.users_attribution_imputation import run

        run(bq_project=BQ_PROJECT, bq_dataset=BQ_DATASET)

    task_impute_users_attribution = PythonOperator(
        task_id="impute_users_attribution",
        python_callable=_run_users_attribution_imputation,
    )

    def _run_generate_touchpoints(**kwargs) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.generate_touchpoints import run

        run(bq_project=BQ_PROJECT, bq_dataset=BQ_DATASET)

    task_generate_touchpoints = PythonOperator(
        task_id="generate_touchpoints_log",
        python_callable=_run_generate_touchpoints,
    )

    task_create_channel_cpc_weights = BigQueryInsertJobOperator(
        task_id="create_channel_cpc_weights",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("intermediate/create_channel_cpc_weights.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_create_channel_spend = BigQueryInsertJobOperator(
        task_id="create_insights_channel_spend",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("intermediate/create_insights_channel_spend.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_create_refund_rate = BigQueryInsertJobOperator(
        task_id="create_refund_rate",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("intermediate/create_refund_rate.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_create_cohorts_retention = BigQueryInsertJobOperator(
        task_id="create_cohorts_retention",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("model_specific/create_cohorts_retention.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_create_attribution_paths = BigQueryInsertJobOperator(
        task_id="create_attribution_paths",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("model_specific/create_attribution_paths.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_create_mmm_timeseries = BigQueryInsertJobOperator(
        task_id="create_mmm_timeseries",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("model_specific/create_mmm_timeseries.sql"),
                "useLegacySql": False,
            }
        },
    )

    task_create_rocv_folds >> task_copy_users_attribution_base >> task_impute_users_attribution
    task_impute_users_attribution >> task_generate_touchpoints
    task_impute_users_attribution >> task_create_cohorts_retention
    task_impute_users_attribution >> task_create_mmm_timeseries
    task_impute_users_attribution >> task_create_refund_rate
    task_generate_touchpoints >> task_create_attribution_paths
    task_generate_touchpoints >> task_create_channel_cpc_weights >> task_create_channel_spend
    task_create_channel_spend >> task_create_mmm_timeseries

