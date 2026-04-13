"""Output-contract and factual-readiness checks for evaluation."""

from __future__ import annotations

from typing import Iterable
import logging

import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

REQUIRED_TABLE_COLUMNS: dict[str, set[str]] = {
    "eval_dda": {
        "fold_id",
        "model_name",
        "forecast_period",
        "expected_conversions",
        "actual_conversions",
        "expected_cac_usd",
        "actual_cac_usd",
    },
    "eval_mmm": {
        "fold_id",
        "model_name",
        "forecast_period",
        "segment",
        "expected_net_revenue_usd",
        "actual_net_revenue_usd",
    },
    "eval_survival": {
        "fold_id",
        "model_name",
        "forecast_period",
        "segment",
        "rebill_period_t",
        "expected_active_users",
        "actual_active_users",
        "expected_ltv_usd",
        "actual_ltv_usd",
    },
    "dda_weights": {"fold_id", "model_name", "media_source", "weight"},
    "mmm_channel_contribs": {
        "fold_id",
        "model_name",
        "segment",
        "incr_gads_search",
        "incr_gads_youtube",
        "incr_gads_discover",
        "incr_metads_inst",
        "incr_metads_fb",
        "incr_tiktok",
    },
}


def validate_table_contracts(
    client: bigquery.Client,
    project: str,
    dataset: str,
) -> None:
    """Ensure required tables and columns are present before evaluation runs."""
    logger.info("Validating output-contract table schemas in `%s.%s`.", project, dataset)
    table_names = sorted(REQUIRED_TABLE_COLUMNS.keys())
    table_filter = ",".join(f"'{name}'" for name in table_names)
    sql = f"""
        SELECT
            table_name,
            column_name
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name IN ({table_filter})
    """
    df = client.query(sql).to_dataframe()
    if df.empty:
        raise ValueError(
            f"No output-contract tables found in `{project}.{dataset}`. "
            "Run Phase 2 + table init tasks before evaluation."
        )

    observed: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        observed.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))

    missing_tables = [name for name in table_names if name not in observed]
    if missing_tables:
        raise ValueError(
            "Missing required output tables: " + ", ".join(sorted(missing_tables))
        )

    missing_columns: list[str] = []
    for table_name, required_columns in REQUIRED_TABLE_COLUMNS.items():
        absent = sorted(required_columns - observed.get(table_name, set()))
        if absent:
            missing_columns.append(f"{table_name}: {', '.join(absent)}")
    if missing_columns:
        raise ValueError(
            "Output-contract mismatch detected. Missing columns -> "
            + " | ".join(missing_columns)
        )
    logger.info("Output-contract schema validation passed.")


def _assert_not_null(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str],
    label: str,
    folds: list[str],
) -> None:
    if frame.empty:
        raise ValueError(f"{label} is empty for folds={folds}. Consolidation is incomplete.")
    if "fold_id" in frame.columns:
        frame = frame[frame["fold_id"].isin(folds)].copy()
    if frame.empty:
        raise ValueError(f"{label} has no rows for folds={folds}.")

    violations = []
    for column in columns:
        if column not in frame.columns:
            violations.append(f"missing column `{column}`")
            continue
        null_count = int(frame[column].isna().sum())
        if null_count > 0:
            violations.append(f"`{column}` contains {null_count} NULL rows")
    if violations:
        raise ValueError(
            f"Factual readiness check failed for {label}: " + "; ".join(violations)
        )


def validate_factual_readiness(
    eval_dda: pd.DataFrame,
    eval_mmm: pd.DataFrame,
    eval_survival: pd.DataFrame,
    folds: list[str],
) -> None:
    """Fail-fast when consolidation has not populated actual_* columns."""
    logger.info("Validating factual readiness (actual_* non-null) for folds=%s", folds)
    _assert_not_null(
        eval_dda,
        columns=["actual_conversions", "actual_cac_usd"],
        label="eval_dda",
        folds=folds,
    )
    _assert_not_null(
        eval_mmm,
        columns=["actual_net_revenue_usd"],
        label="eval_mmm",
        folds=folds,
    )
    _assert_not_null(
        eval_survival,
        columns=["actual_active_users", "actual_ltv_usd"],
        label="eval_survival",
        folds=folds,
    )
    logger.info("Factual readiness validation passed.")
