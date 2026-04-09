"""
Shared helpers for DDA models using EWMA-smoothed static holdout projections.

Spec reference:
    docs/algorithms/dda_models.md §3
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

_ZERO_SPEND_CHANNELS = ("legacy_untracked", "organic")
_EWMA_ALPHA = 0.3


def get_bq_client() -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    return bigquery.Client()


def _run_query(client: bigquery.Client, sql: str) -> pd.DataFrame:
    """Execute SQL and return a pandas DataFrame."""
    return client.query(sql).to_dataframe()


def fetch_attribution_paths(
    client: bigquery.Client,
    project: str,
    dataset: str,
    train_end: str,
) -> pd.DataFrame:
    """
    Load fold-bounded paths from touchpoints and attribution marts.

    Revenue is read directly from attribution_paths (no raw recomputation).
    """
    sql = f"""
        WITH journeys AS (
            SELECT
                user_id,
                STRING_AGG(media_source, ' > ' ORDER BY created_at ASC) AS journey
            FROM `{project}.{dataset}.touchpoints_log`
            WHERE created_at < '{train_end}'
            GROUP BY user_id
        ),
        train_converters AS (
            SELECT DISTINCT
                user_id
            FROM `{project}.{dataset}.purchases`
            WHERE
                rebill_number = 0
                AND order_status IN ('approved', 'settled_ok', 'refunded')
                AND CAST(order_date AS DATE) < '{train_end}'
        )
        SELECT
            j.user_id,
            j.journey,
            CASE WHEN tc.user_id IS NOT NULL THEN TRUE ELSE FALSE END AS is_converted,
            CASE
                WHEN tc.user_id IS NOT NULL THEN COALESCE(ap.conversion_value_usd, 0.0)
                ELSE 0.0
            END AS conversion_value_usd
        FROM journeys AS j
        LEFT JOIN train_converters AS tc
            ON j.user_id = tc.user_id
        LEFT JOIN `{project}.{dataset}.attribution_paths` AS ap
            ON j.user_id = ap.user_id
    """
    df = _run_query(client, sql)
    logger.info("Fetched %d attribution paths bounded by %s.", len(df), train_end)
    return df


def fetch_training_weekly_paid_conversions(
    client: bigquery.Client,
    project: str,
    dataset: str,
    train_end: str,
    fold_id: str,
) -> pd.DataFrame:
    """
    Weekly paid conversion totals for training window (Step 1 of translation).
    """
    sql = f"""
        SELECT
            DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY)) AS week_start,
            COUNT(DISTINCT p.user_id) AS paid_conversions
        FROM `{project}.{dataset}.purchases` AS p
        LEFT JOIN `{project}.{dataset}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
            AND ua_imputed.fold_id = '{fold_id}'
            AND ua_imputed.is_synthetic = FALSE
        LEFT JOIN `{project}.{dataset}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        WHERE
            p.rebill_number = 0
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
            AND CAST(p.order_date AS DATE) < '{train_end}'
            AND COALESCE(ua_imputed.media_source, ua_raw.media_source)
                NOT IN ('legacy_untracked', 'organic')
        GROUP BY week_start
        ORDER BY week_start
    """
    df = _run_query(client, sql)
    if df.empty:
        return pd.DataFrame(columns=["week_start", "paid_conversions"])
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    df["paid_conversions"] = pd.to_numeric(df["paid_conversions"], errors="coerce").fillna(0.0)
    return df


def fetch_training_weekly_spend(
    client: bigquery.Client,
    project: str,
    dataset: str,
    train_end: str,
    fold_id: str,
) -> pd.DataFrame:
    """
    Weekly channel spend in training window (Step 1 of translation).
    """
    sql = f"""
        SELECT
            DATE_TRUNC(date, WEEK(MONDAY)) AS week_start,
            media_source,
            SUM(alloc_spend_in_usd) AS weekly_spend
        FROM `{project}.{dataset}.insights_channel_spend`
        WHERE
            fold_id = '{fold_id}'
            AND date < '{train_end}'
            AND media_source NOT IN ('legacy_untracked', 'organic')
        GROUP BY week_start, media_source
        ORDER BY week_start, media_source
    """
    df = _run_query(client, sql)
    if df.empty:
        return pd.DataFrame(columns=["week_start", "media_source", "weekly_spend"])
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    df["weekly_spend"] = pd.to_numeric(df["weekly_spend"], errors="coerce").fillna(0.0)
    return df


def fetch_holdout_weekly_spend(
    client: bigquery.Client,
    project: str,
    dataset: str,
    holdout_start: str,
    holdout_end: str,
    fold_id: str,
) -> pd.DataFrame:
    """
    Per-channel weekly spend in holdout window.
    """
    sql = f"""
        SELECT
            DATE_TRUNC(date, WEEK(MONDAY)) AS week_start,
            media_source,
            SUM(alloc_spend_in_usd) AS weekly_spend
        FROM `{project}.{dataset}.insights_channel_spend`
        WHERE
            fold_id = '{fold_id}'
            AND date >= '{holdout_start}'
            AND date < '{holdout_end}'
            AND media_source NOT IN ('legacy_untracked', 'organic')
        GROUP BY week_start, media_source
        ORDER BY week_start, media_source
    """
    df = _run_query(client, sql)
    if df.empty:
        return pd.DataFrame(columns=["week_start", "media_source", "weekly_spend"])
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    df["weekly_spend"] = pd.to_numeric(df["weekly_spend"], errors="coerce").fillna(0.0)
    return df


def normalise_paid_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Remove zero-spend channels and renormalize weights to 1.0.
    """
    paid_weights = {
        channel: float(weight)
        for channel, weight in weights.items()
        if channel not in _ZERO_SPEND_CHANNELS
    }
    total = float(sum(paid_weights.values()))
    if total <= 0:
        return {}
    return {channel: value / total for channel, value in paid_weights.items()}


def _ewma_last(values: Iterable[float], alpha: float = _EWMA_ALPHA) -> float:
    cleaned = [float(v) for v in values if pd.notna(v) and np.isfinite(v)]
    if not cleaned:
        return 0.0
    series = pd.Series(cleaned, dtype=float)
    return float(series.ewm(alpha=alpha, adjust=False).mean().iloc[-1])


def compute_channel_ewma_projection(
    weekly_paid_conversions: pd.DataFrame,
    weekly_channel_spend: pd.DataFrame,
    normalised_weights: dict[str, float],
    alpha: float = _EWMA_ALPHA,
) -> dict[str, float]:
    """
    Build per-channel neutral EWMA(CAC) from train-period weekly series.

    Uses an equal-split (1/N_channels) conversion denominator — deliberately
    independent of any model's attribution weights — to avoid the circular
    collapse where CAC[c] ∝ 1/W_c and spend/CAC ∝ W_c, making the aggregate
    invariant to the weight distribution.

    Step 1 (neutral baseline):
      neutral_conversions[c,w] = total_paid_conversions[w] / N_channels
      neutral_CAC[c,w]         = spend[c,w] / neutral_conversions[c,w]
    Step 2:
      ewma_neutral_cac[c] = EWMA(neutral_CAC[c,*], alpha)

    Returns: {channel: ewma_neutral_cac}
    """
    if weekly_paid_conversions.empty or not normalised_weights:
        return {}

    n_channels = len(normalised_weights)
    spend_lookup: dict[tuple[Any, Any], float] = {}
    for _, row in weekly_channel_spend.iterrows():
        spend_lookup[(row["week_start"], row["media_source"])] = float(row["weekly_spend"])

    channel_cac_series: dict[str, list[float]] = {c: [] for c in normalised_weights}

    for _, conv_row in weekly_paid_conversions.iterrows():
        week = conv_row["week_start"]
        paid_conversions = float(conv_row["paid_conversions"])
        neutral_conv_per_channel = paid_conversions / n_channels if n_channels > 0 else 0.0
        for channel in normalised_weights:
            spend = spend_lookup.get((week, channel), 0.0)
            cac = (
                np.nan
                if neutral_conv_per_channel <= 0
                else float(spend / neutral_conv_per_channel)
            )
            channel_cac_series[channel].append(cac)

    return {
        channel: _ewma_last(series, alpha=alpha)
        for channel, series in channel_cac_series.items()
    }


def build_holdout_weeks(holdout_start: str, holdout_end: str) -> list[pd.Timestamp]:
    """
    Return sorted Monday week-start dates spanning [holdout_start, holdout_end).
    """
    start_date = pd.Timestamp(holdout_start).date()
    end_date_exclusive = pd.Timestamp(holdout_end).date()
    if start_date >= end_date_exclusive:
        return []

    days = pd.date_range(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date_exclusive) - pd.Timedelta(days=1),
        freq="D",
    )
    week_starts = sorted({(day - pd.Timedelta(days=day.weekday())).normalize() for day in days})
    return week_starts


def translate_weights_to_forecasts_ewma(
    weights: dict[str, float],
    model_name: str,
    client: bigquery.Client,
    project: str,
    dataset: str,
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
    confidence_weight: int = 0,
) -> pd.DataFrame:
    """
    Apply DDA EWMA-smoothed holdout projection with weight-as-quality-signal.

    Channel CACs are estimated using a neutral equal-split baseline (1/N_channels),
    independent of model weights. Model weights then blend these neutral CACs into
    a single model-specific aggregate CAC:

      model_aggregate_CAC   = Σ_c W_c * ewma_neutral_cac[c]
      expected_conversions[w] = total_holdout_spend[w] / model_aggregate_CAC
      expected_cac_usd[w]     = model_aggregate_CAC  (constant per model/fold)

    W_c is the direct, linear driver of all outputs: high weight on cheap channels
    → lower aggregate CAC → more predicted conversions.
    """
    normalised = normalise_paid_weights(weights)
    if not normalised:
        logger.warning("%s: paid weights empty after normalization.", model_name)
        return pd.DataFrame(
            columns=[
                "fold_id",
                "model_name",
                "forecast_period",
                "expected_conversions",
                "actual_conversions",
                "expected_cac_usd",
                "actual_cac_usd",
                "confidence_weight",
            ]
        )

    weekly_conversions = fetch_training_weekly_paid_conversions(
        client=client,
        project=project,
        dataset=dataset,
        train_end=train_end,
        fold_id=fold_id,
    )
    weekly_spend = fetch_training_weekly_spend(
        client=client,
        project=project,
        dataset=dataset,
        train_end=train_end,
        fold_id=fold_id,
    )
    # {channel: ewma_neutral_cac} — model-agnostic, equal-split baseline
    neutral_cac_per_channel = compute_channel_ewma_projection(
        weekly_paid_conversions=weekly_conversions,
        weekly_channel_spend=weekly_spend,
        normalised_weights=normalised,
        alpha=_EWMA_ALPHA,
    )
    if not neutral_cac_per_channel:
        logger.warning("%s: no neutral CAC estimates, skipping projection.", model_name)
        return pd.DataFrame(
            columns=[
                "fold_id", "model_name", "forecast_period",
                "expected_conversions", "actual_conversions",
                "expected_cac_usd", "actual_cac_usd", "confidence_weight",
            ]
        )

    # model_aggregate_CAC = Σ_c W_c * ewma_neutral_cac[c]
    # W_c is the direct, linear driver — high weight on cheap channels → lower CAC.
    model_aggregate_cac = sum(
        normalised[ch] * neutral_cac_per_channel[ch]
        for ch in normalised
        if ch in neutral_cac_per_channel and neutral_cac_per_channel[ch] > 0
    )
    if model_aggregate_cac <= 0:
        logger.warning("%s: model_aggregate_cac is zero, skipping projection.", model_name)
        return pd.DataFrame(
            columns=[
                "fold_id", "model_name", "forecast_period",
                "expected_conversions", "actual_conversions",
                "expected_cac_usd", "actual_cac_usd", "confidence_weight",
            ]
        )

    holdout_spend_df = fetch_holdout_weekly_spend(
        client=client,
        project=project,
        dataset=dataset,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        fold_id=fold_id,
    )
    # Aggregate to total spend per week (all paid channels summed)
    holdout_total_spend: dict[Any, float] = {}
    for _, row in holdout_spend_df.iterrows():
        week_key = row["week_start"]
        holdout_total_spend[week_key] = holdout_total_spend.get(week_key, 0.0) + float(
            row["weekly_spend"]
        )

    holdout_weeks = build_holdout_weeks(holdout_start=holdout_start, holdout_end=holdout_end)
    rows = []
    for week_start in holdout_weeks:
        week_date = week_start.date()
        total_spend = holdout_total_spend.get(week_date, 0.0)
        # expected_conversions = total_holdout_spend / model_aggregate_CAC
        expected_conversions = total_spend / model_aggregate_cac if total_spend > 0 else 0.0
        rows.append(
            {
                "fold_id": fold_id,
                "model_name": model_name,
                "forecast_period": week_date,
                "expected_conversions": float(round(expected_conversions, 2)),
                "actual_conversions": None,
                "expected_cac_usd": float(round(model_aggregate_cac, 2)),
                "actual_cac_usd": None,
                "confidence_weight": int(confidence_weight),
            }
        )
    return pd.DataFrame(rows)


def write_forecasts_to_bq(
    client: bigquery.Client,
    forecast_df: pd.DataFrame,
    project: str,
    dataset: str,
    model_name: str,
    target_table: str = "eval_dda",
) -> None:
    """
    Fold-scoped idempotent write into eval table.
    """
    table_fqn = f"{project}.{dataset}.{target_table}"

    fold_id_val = (
        forecast_df["fold_id"].iloc[0]
        if "fold_id" in forecast_df.columns and not forecast_df.empty
        else None
    )
    if fold_id_val:
        delete_sql = f"""
            DELETE FROM `{table_fqn}`
            WHERE model_name = '{model_name}' AND fold_id = '{fold_id_val}'
        """
    else:
        delete_sql = f"""
            DELETE FROM `{table_fqn}`
            WHERE model_name = '{model_name}'
        """
    client.query(delete_sql).result()

    if forecast_df.empty:
        logger.warning("No forecast rows for %s.", model_name)
        return

    schemas = {
        "eval_dda": [
            bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_name", "STRING"),
            bigquery.SchemaField("forecast_period", "DATE"),
            bigquery.SchemaField("expected_conversions", "FLOAT64"),
            bigquery.SchemaField("actual_conversions", "INT64"),
            bigquery.SchemaField("expected_cac_usd", "FLOAT64"),
            bigquery.SchemaField("actual_cac_usd", "FLOAT64"),
            bigquery.SchemaField("confidence_weight", "INT64"),
        ],
        "eval_mmm": [
            bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_name", "STRING"),
            bigquery.SchemaField("forecast_period", "DATE"),
            bigquery.SchemaField("segment", "STRING"),
            bigquery.SchemaField("expected_net_revenue_usd", "FLOAT64"),
            bigquery.SchemaField("actual_net_revenue_usd", "FLOAT64"),
            bigquery.SchemaField("crps_score", "FLOAT64"),
            bigquery.SchemaField("confidence_weight", "INT64"),
            bigquery.SchemaField("prior_source", "STRING"),
        ],
        "eval_survival": [
            bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_name", "STRING"),
            bigquery.SchemaField("forecast_period", "DATE"),
            bigquery.SchemaField("segment", "STRING"),
            bigquery.SchemaField("rebill_period_t", "INT64"),
            bigquery.SchemaField("expected_active_users", "FLOAT64"),
            bigquery.SchemaField("actual_active_users", "INT64"),
            bigquery.SchemaField("confidence_weight", "FLOAT64"),
        ],
    }
    schema = schemas.get(target_table)
    if schema:
        target_columns = [field.name for field in schema]
        upload_df = forecast_df.copy()
        for column in target_columns:
            if column not in upload_df.columns:
                upload_df[column] = None
        upload_df = upload_df[target_columns]
    else:
        upload_df = forecast_df.copy()

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
    )
    job = client.load_table_from_dataframe(upload_df, table_fqn, job_config=job_config)
    job.result()
    logger.info("%s: wrote %d rows to %s.", model_name, len(upload_df), table_fqn)


def write_weights_to_bq(
    client: bigquery.Client,
    weights: dict[str, float],
    project: str,
    dataset: str,
    model_name: str,
    fold_id: str,
) -> None:
    """
    Fold-scoped idempotent write into dda_weights.
    """
    table_fqn = f"{project}.{dataset}.dda_weights"
    if not weights:
        logger.warning("No weights to write for %s.", model_name)
        return

    delete_sql = f"""
        DELETE FROM `{table_fqn}`
        WHERE model_name = '{model_name}' AND fold_id = '{fold_id}'
    """
    client.query(delete_sql).result()

    rows = [
        {
            "fold_id": fold_id,
            "model_name": model_name,
            "media_source": channel,
            "weight": round(float(weight), 6),
        }
        for channel, weight in weights.items()
    ]
    df = pd.DataFrame(rows)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_name", "STRING"),
            bigquery.SchemaField("media_source", "STRING"),
            bigquery.SchemaField("weight", "FLOAT64"),
        ],
    )
    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_config)
    job.result()
    logger.info("%s fold=%s: wrote %d weights.", model_name, fold_id, len(df))
