"""
dag_phase2_mmm.py
=================
Apache Airflow DAG for Phase 2 — Predictive Modeling Layer (MMM).

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

# ===================================================================
# Configuration
# ===================================================================
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


def _render_sql(filename: str) -> str:
    """Read a .sql file from the sql/ directory and apply substitution."""
    raw = (SQL_DIR / filename).read_text(encoding="utf-8")
    return raw.replace("{{ project }}", BQ_PROJECT).replace("{{ dataset }}", BQ_DATASET)


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


def _resolve_docker_mount_source() -> str:
    """
    Resolve bind source for DockerOperator mount.

    Priority:
    1) Explicit DOCKER_MOUNT_DIR (recommended).
    2) Existing local path candidates.
    """
    explicit = os.environ.get("DOCKER_MOUNT_DIR")
    if explicit:
        if explicit.startswith("/mnt/") and not Path(explicit).exists():
            # Docker Desktop Linux VM often exposes Windows drives under
            # /run/desktop/mnt/host/<drive>, not /mnt/<drive>.
            parts = explicit.split("/", maxsplit=3)
            # ['', 'mnt', '<drive>', '<rest...>']
            if len(parts) >= 4 and len(parts[2]) == 1:
                mapped = f"/run/desktop/mnt/host/{parts[2]}/{parts[3]}"
                if Path(mapped).exists():
                    logger.info("DOCKER_MOUNT_DIR remapped from %s to %s", explicit, mapped)
                    return mapped
        return explicit

    candidates = [
        _PROJECT_ROOT,
        "/run/desktop/mnt/host/c/Users/Gigabyte/Desktop/diploma",
        "/mnt/c/Users/Gigabyte/Desktop/diploma",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            logger.info("Resolved Docker mount source: %s", candidate)
            return candidate
    logger.warning("No known Docker mount source exists; falling back to %s", _PROJECT_ROOT)
    return _PROJECT_ROOT


# ===================================================================
# Task callables
# ===================================================================

def _run_mmm_ols(**kwargs) -> None:
    """OLS MMM baseline entry point for Airflow (fold-aware)."""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from models.mmm.run_baseline_ols_new import run
    fold = _get_fold_conf(kwargs)
    run(
        bq_project=BQ_PROJECT,
        bq_dataset=BQ_DATASET,
        fold_id=fold["fold_id"],
        train_end=fold["train_end"],
        holdout_start=fold["holdout_start"],
        holdout_end=fold["holdout_end"],
    )


# ===================================================================
# DAG definition
# ===================================================================
with DAG(
    dag_id="phase2_mmm_models",
    description="Phase 2 — Macro-Analytical Plane (MMM Models) | One run per ROCV fold",
    default_args=DAG_DEFAULT_ARGS,
    schedule=None,          # Triggered automatically by phase2_dda_models
    start_date=datetime(2021, 4, 1),
    catchup=False,
    tags=["phase-2", "predictive-models", "mmm", "rocv"],
    max_active_runs=1,
) as dag:
    task_init_mmm_channel_contribs = BigQueryInsertJobOperator(
        task_id="init_mmm_channel_contribs",
        gcp_conn_id=BQ_CONN_ID,
        configuration={
            "query": {
                "query": _render_sql("final/init_mmm_channel_contribs.sql"),
                "useLegacySql": False,
            }
        },
    )

    # ===================================================================
    # MMM Models (Macro-Analytical Plane)
    # Reads from: mmm_timeseries, refund_rates, insights_channel_spend
    # Writes to: eval_mmm
    # ===================================================================

    task_model_mmm_ols = PythonOperator(
        task_id="model_mmm_baseline_ols",
        python_callable=_run_mmm_ols,
    )

    # Bayesian MMM: runs inside an ephemeral PyMC-ready Docker container.
    from airflow.providers.docker.operators.docker import DockerOperator  # noqa: PLC0415
    from docker.types import Mount  # noqa: PLC0415

    HOST_PROJECT_ROOT = _resolve_docker_mount_source()

    # NOTE: DockerOperator does not have a built-in way to read dag_run.conf.
    # The fold boundaries are read from environment variables injected via
    # the `environment` dict — the outer DAG passes them as ROCV_* vars.
    # For simplicity, we inject all 4 fold params from env (set by the
    # orchestrating ROCV wrapper DAG) and fall back to fold_4 defaults.
    _FOLD_ID       = os.environ.get("ROCV_FOLD_ID",       "fold_4")
    _TRAIN_END     = os.environ.get("ROCV_TRAIN_END",     "2021-12-01")
    _HOLDOUT_START = os.environ.get("ROCV_HOLDOUT_START", "2021-12-01")
    _HOLDOUT_END   = os.environ.get("ROCV_HOLDOUT_END",   "2022-03-01")

    _docker_fold_args = (
        f"fold_id='{_FOLD_ID}', "
        f"train_end='{_TRAIN_END}', "
        f"holdout_start='{_HOLDOUT_START}', "
        f"holdout_end='{_HOLDOUT_END}'"
    )

    task_model_mmm_bsts = DockerOperator(
        task_id="model_mmm_bsts",
        image="thesis-pymc-model",
        mounts=[
            Mount(
                source=HOST_PROJECT_ROOT,
                target="/app",
                type="bind",
            )
        ],
        working_dir="/app",
        command=(
            "python -c \""
            "import sys; sys.path.insert(0, '/app'); "
            "from models.mmm.run_bsts_new import run; "
            "run(bq_project='" + BQ_PROJECT + "', "
            "bq_dataset='" + BQ_DATASET + "', "
            "fold_id='{{ dag_run.conf.get('fold_id', 'fold_4') }}', "
            "train_end='{{ dag_run.conf.get('train_end', '2021-12-01') }}', "
            "holdout_start='{{ dag_run.conf.get('holdout_start', '2021-12-01') }}', "
            "holdout_end='{{ dag_run.conf.get('holdout_end', '2022-03-01') }}')\""
        ),
        environment={
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/sa_key.json",
            "BQ_PROJECT": BQ_PROJECT,
            "BQ_DATASET": BQ_DATASET,
            "MMM_REPORTS_DIR": "/app/reports",
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "HOME": "/tmp",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        },
        user="0:0",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
    )

    task_model_mmm_bayesian = DockerOperator(
        task_id="model_mmm_bayesian",
        image="thesis-pymc-model",
        mounts=[
            Mount(
                source=HOST_PROJECT_ROOT,
                target="/app",
                type="bind",
            )
        ],
        working_dir="/app",
        command=(
            "python -c \""
            "import sys; sys.path.insert(0, '/app'); "
            "from models.mmm.run_bayesian_regression_new import run; "
            "run(bq_project='" + BQ_PROJECT + "', "
            "bq_dataset='" + BQ_DATASET + "', "
            "fold_id='{{ dag_run.conf.get('fold_id', 'fold_4') }}', "
            "train_end='{{ dag_run.conf.get('train_end', '2021-12-01') }}', "
            "holdout_start='{{ dag_run.conf.get('holdout_start', '2021-12-01') }}', "
            "holdout_end='{{ dag_run.conf.get('holdout_end', '2022-03-01') }}')\""
        ),
        environment={
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/sa_key.json",
            "BQ_PROJECT": BQ_PROJECT,
            "BQ_DATASET": BQ_DATASET,
            "MMM_REPORTS_DIR": "/app/reports",
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "HOME": "/tmp",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        },
        user="0:0",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
    )

    for task in [task_model_mmm_ols, task_model_mmm_bsts, task_model_mmm_bayesian]:
        task_init_mmm_channel_contribs >> task
