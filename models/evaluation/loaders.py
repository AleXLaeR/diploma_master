"""BigQuery and local-artifact loaders used by the evaluation package."""

from __future__ import annotations

from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery

from models.attribution.dda_common_new import fetch_attribution_paths
from models.rocv import get_fold

logger = logging.getLogger(__name__)


def get_bq_client() -> bigquery.Client:
    logger.info("Creating BigQuery client for evaluation loaders.")
    return bigquery.Client()


def run_query(client: bigquery.Client, sql: str, *, label: str) -> pd.DataFrame:
    start = time.perf_counter()
    logger.info("BQ query start: %s", label)
    df = client.query(sql).to_dataframe()
    elapsed = time.perf_counter() - start
    logger.info("BQ query done: %s rows=%d elapsed=%.2fs", label, len(df), elapsed)
    return df


def _fold_filter_sql(folds: list[str] | None, column: str = "fold_id") -> str:
    if not folds:
        return ""
    quoted = ",".join(f"'{fold}'" for fold in folds)
    return f" AND {column} IN ({quoted})"


def load_eval_dda(
    client: bigquery.Client,
    project: str,
    dataset: str,
    folds: list[str] | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{project}.{dataset}.eval_dda`
        WHERE TRUE {_fold_filter_sql(folds)}
        ORDER BY fold_id, model_name, forecast_period
    """
    return run_query(client, sql, label="load_eval_dda")


def load_eval_mmm(
    client: bigquery.Client,
    project: str,
    dataset: str,
    folds: list[str] | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{project}.{dataset}.eval_mmm`
        WHERE TRUE {_fold_filter_sql(folds)}
        ORDER BY fold_id, model_name, segment, forecast_period
    """
    return run_query(client, sql, label="load_eval_mmm")


def load_eval_survival(
    client: bigquery.Client,
    project: str,
    dataset: str,
    folds: list[str] | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{project}.{dataset}.eval_survival`
        WHERE TRUE {_fold_filter_sql(folds)}
        ORDER BY fold_id, model_name, segment, rebill_period_t, forecast_period
    """
    return run_query(client, sql, label="load_eval_survival")


def load_dda_weights(
    client: bigquery.Client,
    project: str,
    dataset: str,
    folds: list[str] | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{project}.{dataset}.dda_weights`
        WHERE TRUE {_fold_filter_sql(folds)}
        ORDER BY fold_id, model_name, media_source
    """
    return run_query(client, sql, label="load_dda_weights")


def load_mmm_channel_contribs(
    client: bigquery.Client,
    project: str,
    dataset: str,
    folds: list[str] | None = None,
) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{project}.{dataset}.mmm_channel_contribs`
        WHERE TRUE {_fold_filter_sql(folds)}
        ORDER BY fold_id, model_name, segment
    """
    return run_query(client, sql, label="load_mmm_channel_contribs")


def load_mmm_timeseries_fold(
    client: bigquery.Client,
    project: str,
    dataset: str,
    fold_id: str,
) -> pd.DataFrame:
    sql = f"""
        SELECT
            date_week,
            macro_region,
            total_net_revenue_usd
        FROM `{project}.{dataset}.mmm_timeseries`
        WHERE fold_id = '{fold_id}'
        ORDER BY date_week, macro_region
    """
    return run_query(client, sql, label=f"load_mmm_timeseries_fold[{fold_id}]")


def load_attribution_paths_for_fold(
    client: bigquery.Client,
    project: str,
    dataset: str,
    fold_id: str,
) -> pd.DataFrame:
    """Recompute fold-bounded attribution path view using authoritative ROCV boundaries."""
    fold = get_fold(fold_id)
    logger.info(
        "Loading attribution paths for fold=%s (train_end=%s).",
        fold_id,
        fold["train_end"],
    )
    df = fetch_attribution_paths(
        client=client,
        project=project,
        dataset=dataset,
        train_end=fold["train_end"],
    )
    logger.info("Loaded attribution paths for fold=%s rows=%d", fold_id, len(df))
    return df


def get_reports_root(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "reports"


def load_posterior_trace(
    model_name: str,
    fold_id: str,
    reports_root: Path,
) -> object | None:
    safe = model_name.replace(" ", "_")
    path = reports_root / f"posterior_trace_{safe}_{fold_id}.nc"
    if not path.exists():
        logger.warning("Posterior trace not found: %s", path)
        return None
    logger.info("Loading posterior trace (xr.Dataset): %s", path)
    try:
        import xarray as xr  # noqa: PLC0415
        return xr.open_dataset(str(path))
    except Exception as exc:  # pragma: no cover
        logger.warning("xarray failed reading trace %s: %s — falling back to ArviZ", path, exc)
        try:
            import arviz as az  # noqa: PLC0415
            return az.from_netcdf(str(path))
        except Exception as exc2:
            logger.warning("ArviZ also failed: %s", exc2)
            return None



def load_posterior_predictive(
    model_name: str,
    fold_id: str,
    reports_root: Path,
) -> np.ndarray | None:
    safe = model_name.replace(" ", "_")
    path = reports_root / f"posterior_predictive_{safe}_{fold_id}.npy"
    if not path.exists():
        logger.warning("Posterior predictive not found: %s", path)
        return None
    logger.info("Loading posterior predictive: %s", path)
    return np.load(path)


def load_ols_coeffs(
    fold_id: str,
    reports_root: Path,
) -> pd.DataFrame | None:
    path = reports_root / f"ols_channel_coeffs_{fold_id}.csv"
    if not path.exists():
        logger.warning("OLS coefficients artifact not found: %s", path)
        return None
    logger.info("Loading OLS coefficients artifact: %s", path)
    return pd.read_csv(path)
