"""BigQuery persistence helpers for MMM v2 (task #9)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from google.cloud import bigquery

from models.mmm.common_new import CHANNEL_TO_ACTUAL_COLUMN, CHANNEL_TO_INCR_COLUMN

logger = logging.getLogger(__name__)

EVAL_MMM_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_name", "STRING"),
    bigquery.SchemaField("forecast_period", "DATE"),
    bigquery.SchemaField("segment", "STRING"),
    bigquery.SchemaField("expected_net_revenue_usd", "FLOAT64"),
    bigquery.SchemaField("actual_net_revenue_usd", "FLOAT64"),
    bigquery.SchemaField("base_sales_intercept", "FLOAT64"),
    bigquery.SchemaField("mean_saturation_point", "FLOAT64"),
    bigquery.SchemaField("prior_source", "STRING"),
    bigquery.SchemaField("confidence_weight", "INT64"),
]

MMM_CHANNEL_CONTRIBS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("segment", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("incr_gads_search", "FLOAT64"),
    bigquery.SchemaField("incr_gads_youtube", "FLOAT64"),
    bigquery.SchemaField("incr_gads_discover", "FLOAT64"),
    bigquery.SchemaField("incr_metads_inst", "FLOAT64"),
    bigquery.SchemaField("incr_metads_fb", "FLOAT64"),
    bigquery.SchemaField("incr_tiktok", "FLOAT64"),
    bigquery.SchemaField("actual_contrib_gads_search", "FLOAT64"),
    bigquery.SchemaField("actual_contrib_gads_youtube", "FLOAT64"),
    bigquery.SchemaField("actual_contrib_gads_discover", "FLOAT64"),
    bigquery.SchemaField("actual_contrib_metads_inst", "FLOAT64"),
    bigquery.SchemaField("actual_contrib_metads_fb", "FLOAT64"),
    bigquery.SchemaField("actual_contrib_tiktok", "FLOAT64"),
]


def build_eval_mmm_row(
    fold_id: str,
    model_name: str,
    forecast_period: Any,
    segment: str,
    expected_net_revenue_usd: float,
    base_sales_intercept: float | None,
    mean_saturation_point: float | None,
    prior_source: str | None,
    confidence_weight: int,
) -> dict[str, Any]:
    """Build one schema-aligned eval_mmm row."""
    return {
        "fold_id": fold_id,
        "model_name": model_name,
        "forecast_period": pd.Timestamp(forecast_period).date(),
        "segment": segment,
        "expected_net_revenue_usd": float(expected_net_revenue_usd),
        "actual_net_revenue_usd": None,
        "base_sales_intercept": (
            float(base_sales_intercept) if base_sales_intercept is not None else None
        ),
        "mean_saturation_point": (
            float(mean_saturation_point) if mean_saturation_point is not None else None
        ),
        "prior_source": prior_source,
        "confidence_weight": int(confidence_weight),
    }


def _prepare_dataframe(df: pd.DataFrame, schema: list[bigquery.SchemaField]) -> pd.DataFrame:
    """Add missing columns and order dataframe to schema."""
    ordered_cols = [field.name for field in schema]
    prepared = df.copy()
    for column in ordered_cols:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[ordered_cols]


def write_eval_mmm_to_bq(
    client: bigquery.Client,
    project: str,
    dataset: str,
    model_name: str,
    fold_id: str,
    eval_rows: pd.DataFrame,
) -> None:
    """Fold+model scoped idempotent write into eval_mmm."""
    table_fqn = f"{project}.{dataset}.eval_mmm"

    delete_sql = f"""
        DELETE FROM `{table_fqn}`
        WHERE model_name = '{model_name}'
          AND fold_id = '{fold_id}'
    """
    client.query(delete_sql).result()

    if eval_rows.empty:
        logger.warning("No eval_mmm rows to write for %s fold=%s.", model_name, fold_id)
        return

    upload_df = _prepare_dataframe(eval_rows, EVAL_MMM_SCHEMA)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=EVAL_MMM_SCHEMA,
    )
    job = client.load_table_from_dataframe(upload_df, table_fqn, job_config=job_config)
    job.result()
    logger.info("Wrote %d eval_mmm rows for %s fold=%s.", len(upload_df), model_name, fold_id)


def build_channel_contrib_row(
    fold_id: str,
    model_name: str,
    segment: str,
    shares_by_channel: dict[str, float],
) -> dict[str, Any]:
    """Build one schema-aligned mmm_channel_contribs row."""
    row: dict[str, Any] = {
        "fold_id": fold_id,
        "model_name": model_name,
        "segment": segment,
    }

    for channel, incr_col in CHANNEL_TO_INCR_COLUMN.items():
        row[incr_col] = float(shares_by_channel.get(channel, 0.0))

    for actual_col in CHANNEL_TO_ACTUAL_COLUMN.values():
        row[actual_col] = None

    return row


def write_mmm_channel_contribs_to_bq(
    client: bigquery.Client,
    project: str,
    dataset: str,
    model_name: str,
    fold_id: str,
    contrib_rows: pd.DataFrame,
) -> None:
    """Fold+model scoped idempotent write into mmm_channel_contribs."""
    table_fqn = f"{project}.{dataset}.mmm_channel_contribs"

    delete_sql = f"""
        DELETE FROM `{table_fqn}`
        WHERE model_name = '{model_name}'
          AND fold_id = '{fold_id}'
    """
    client.query(delete_sql).result()

    if contrib_rows.empty:
        logger.warning(
            "No mmm_channel_contribs rows to write for %s fold=%s.", model_name, fold_id,
        )
        return

    upload_df = _prepare_dataframe(contrib_rows, MMM_CHANNEL_CONTRIBS_SCHEMA)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=MMM_CHANNEL_CONTRIBS_SCHEMA,
    )
    job = client.load_table_from_dataframe(upload_df, table_fqn, job_config=job_config)
    job.result()
    logger.info(
        "Wrote %d mmm_channel_contribs rows for %s fold=%s.",
        len(upload_df),
        model_name,
        fold_id,
    )
