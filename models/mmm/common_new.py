"""Shared utilities for MMM v2 implementation (task #9)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

MIN_DENSITY_RATIO: float = 0.30
MIN_WEEKLY_REVENUE_USD: float = 150.0
EPSILON: float = 1e-8

SPEND_COLUMNS: list[str] = [
    "spend_gads_search",
    "spend_gads_youtube",
    "spend_gads_discover",
    "spend_metads_inst",
    "spend_metads_fb",
    "spend_tiktok",
]

CHANNEL_NAMES: list[str] = [
    "gads:search",
    "gads:youtube",
    "gads:discover",
    "metads:inst",
    "metads:fb",
    "tiktok",
]

SPEND_TO_CHANNEL: dict[str, str] = dict(zip(SPEND_COLUMNS, CHANNEL_NAMES, strict=True))
CHANNEL_TO_SPEND: dict[str, str] = {v: k for k, v in SPEND_TO_CHANNEL.items()}

CHANNEL_TO_INCR_COLUMN: dict[str, str] = {
    "gads:search": "incr_gads_search",
    "gads:youtube": "incr_gads_youtube",
    "gads:discover": "incr_gads_discover",
    "metads:inst": "incr_metads_inst",
    "metads:fb": "incr_metads_fb",
    "tiktok": "incr_tiktok",
}

CHANNEL_TO_ACTUAL_COLUMN: dict[str, str] = {
    "gads:search": "actual_contrib_gads_search",
    "gads:youtube": "actual_contrib_gads_youtube",
    "gads:discover": "actual_contrib_gads_discover",
    "metads:inst": "actual_contrib_metads_inst",
    "metads:fb": "actual_contrib_metads_fb",
    "tiktok": "actual_contrib_tiktok",
}

EXOG_COLUMNS: list[str] = [
    "fourier_sin_q1",
    "fourier_cos_q1",
    "revenue_anomaly_score",
    "is_structural_peak",
    "is_sep_nov_trough",
]

EXOG_PRIORS: dict[str, tuple[float, float]] = {
    "fourier_sin_q1": (0.0, 1.0),
    "fourier_cos_q1": (0.0, 1.0),
    "revenue_anomaly_score": (0.0, 0.5),
    "is_structural_peak": (0.5, 0.75),
    "is_sep_nov_trough": (-0.5, 0.75),
}

FUNNEL_TIERS: dict[str, str] = {
    "tiktok": "top",
    "gads:discover": "top",
    "gads:youtube": "mid",
    "metads:inst": "mid",
    "gads:search": "bottom",
    "metads:fb": "bottom",
}

ALPHA_MEAN_BY_TIER: dict[str, float] = {
    "top": 0.75,
    "mid": 0.50,
    "bottom": 0.25,
}


def get_bq_client() -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    return bigquery.Client()


def segment_for_region(region: str) -> str:
    """Return output-contract segment key for a region/global label."""
    return f"Total_Macro_{region}"


def load_mmm_data(
    client: bigquery.Client,
    project: str,
    dataset: str,
    fold_id: str,
    train_end: str | None = None,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> pd.DataFrame:
    """Load fold-bounded rows from mmm_timeseries."""
    if holdout_start and not holdout_end:
        msg = "holdout_end must be provided when holdout_start is set"
        raise ValueError(msg)

    where_clauses = [f"fold_id = '{fold_id}'"]
    if train_end:
        where_clauses.append(f"date_week < '{train_end}'")
    if holdout_start and holdout_end:
        where_clauses.append(f"date_week >= '{holdout_start}'")
        where_clauses.append(f"date_week < '{holdout_end}'")

    sql = f"""
        SELECT
            date_week,
            macro_region,
            total_net_revenue_usd,
            {', '.join(SPEND_COLUMNS)},
            {', '.join(EXOG_COLUMNS)}
        FROM `{project}.{dataset}.mmm_timeseries`
        WHERE {' AND '.join(where_clauses)}
        ORDER BY macro_region, date_week
    """

    df = client.query(sql).to_dataframe()
    if df.empty:
        return df

    df["date_week"] = pd.to_datetime(df["date_week"])
    for column in ["total_net_revenue_usd", *SPEND_COLUMNS, *EXOG_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df


def fetch_shapley_weights(
    client: bigquery.Client,
    project: str,
    dataset: str,
    fold_id: str,
) -> dict[str, float] | None:
    """Read fold-specific Shapley weights for Scenario B priors."""
    sql = f"""
        SELECT media_source, weight
        FROM `{project}.{dataset}.dda_weights`
        WHERE model_name = 'Shapley_DDA'
          AND fold_id = '{fold_id}'
    """
    try:
        df = client.query(sql).to_dataframe()
    except Exception as exc:  # pragma: no cover - network/credential path
        logger.warning("Shapley weight query failed for fold=%s: %s", fold_id, exc)
        return None

    if df.empty:
        logger.warning("No Shapley_DDA rows found for fold=%s.", fold_id)
        return None

    raw_weights = {
        str(row["media_source"]): float(row["weight"])
        for _, row in df.iterrows()
        if str(row["media_source"]) in CHANNEL_NAMES
    }
    total = float(sum(max(value, 0.0) for value in raw_weights.values()))
    if total <= 0.0:
        logger.warning("Fold=%s Shapley weights are non-positive; fallback to heuristic.", fold_id)
        return None

    normalized: dict[str, float] = {}
    for channel in CHANNEL_NAMES:
        normalized[channel] = max(raw_weights.get(channel, 0.0), 0.0) / total

    return normalized


def adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """Geometric carryover transform A[t] = S[t] + alpha * A[t-1]."""
    if not 0.0 <= alpha < 1.0:
        msg = f"AdStock alpha must be in [0, 1), got {alpha}"
        raise ValueError(msg)
    if spend.size == 0:
        return np.array([], dtype=float)

    output = np.empty_like(spend, dtype=float)
    output[0] = float(spend[0])
    for idx in range(1, spend.shape[0]):
        output[idx] = float(spend[idx]) + alpha * output[idx - 1]
    return output


def apply_minmax_scaler(series: np.ndarray, lo: float, hi: float, eps: float = EPSILON) -> np.ndarray:
    """Apply training-window min-max scaling and epsilon clipping."""
    if hi - lo < 1e-12:
        return np.full_like(series, eps, dtype=float)
    return np.clip((series - lo) / (hi - lo), eps, 1.0)


def fit_spend_scalers(train_spend: np.ndarray, alphas: Sequence[float]) -> list[tuple[float, float]]:
    """Fit (min,max) scaler bounds per channel using training spend only."""
    n_channels = train_spend.shape[1]
    if len(alphas) != n_channels:
        msg = "Length of alphas must equal number of spend channels"
        raise ValueError(msg)

    scalers: list[tuple[float, float]] = []
    for channel_idx in range(n_channels):
        adstocked = adstock(train_spend[:, channel_idx], float(alphas[channel_idx]))
        scalers.append((float(adstocked.min()), float(adstocked.max())))
    return scalers


def transform_spend_with_scalers(
    spend: np.ndarray,
    alphas: Sequence[float],
    scalers: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Apply AdStock and train-window min-max scaling to spend matrix."""
    n_channels = spend.shape[1]
    if len(alphas) != n_channels or len(scalers) != n_channels:
        msg = "alphas and scalers must align with spend channel count"
        raise ValueError(msg)

    output = np.empty_like(spend, dtype=float)
    for channel_idx in range(n_channels):
        adstocked = adstock(spend[:, channel_idx], float(alphas[channel_idx]))
        lo, hi = scalers[channel_idx]
        output[:, channel_idx] = apply_minmax_scaler(adstocked, lo, hi)
    return output


def hill(x: np.ndarray, k: np.ndarray | float, s: np.ndarray | float) -> np.ndarray:
    """Hill saturation function for normalized inputs."""
    return 1.0 / (1.0 + (k / x) ** s)


def check_fallback_needed(
    train_df: pd.DataFrame,
    min_density_ratio: float = MIN_DENSITY_RATIO,
    min_weekly_revenue: float = MIN_WEEKLY_REVENUE_USD,
) -> bool:
    """Return True when region-level fit should fallback to global."""
    if train_df.empty:
        return True

    mean_weekly_revenue = float(train_df["total_net_revenue_usd"].mean())
    if mean_weekly_revenue < min_weekly_revenue:
        logger.warning(
            "Fallback trigger: mean weekly revenue %.2f < %.2f",
            mean_weekly_revenue,
            min_weekly_revenue,
        )
        return True

    for column in SPEND_COLUMNS:
        density = float((train_df[column] > 0.0).mean())
        if density < min_density_ratio:
            logger.warning(
                "Fallback trigger: channel %s density %.2f%% < %.2f%%",
                column,
                density * 100.0,
                min_density_ratio * 100.0,
            )
            return True

    return False


def aggregate_to_global(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regional MMM data to a single global series."""
    if df.empty:
        return df.copy()

    aggregations: dict[str, str] = {
        "total_net_revenue_usd": "sum",
        **{column: "sum" for column in SPEND_COLUMNS},
        **{column: "first" for column in EXOG_COLUMNS},
    }

    out = (
        df.groupby("date_week", as_index=False)
        .agg(aggregations)
        .sort_values("date_week")
        .reset_index(drop=True)
    )
    out["macro_region"] = "Global"
    return out[["date_week", "macro_region", "total_net_revenue_usd", *SPEND_COLUMNS, *EXOG_COLUMNS]]


def compute_training_revenue_shares(train_df: pd.DataFrame, regions: Sequence[str]) -> dict[str, float]:
    """Compute training-period revenue shares for region disaggregation."""
    region_revenue = (
        train_df.groupby("macro_region")["total_net_revenue_usd"]
        .sum()
        .reindex(list(regions), fill_value=0.0)
    )
    total = float(region_revenue.sum())
    if total <= 0.0:
        if not regions:
            return {}
        uniform = 1.0 / float(len(regions))
        return {region: uniform for region in regions}
    return {region: float(value / total) for region, value in region_revenue.items()}


def disaggregate_global_eval_rows(
    global_eval_rows: pd.DataFrame,
    train_df: pd.DataFrame,
    target_regions: Sequence[str],
) -> pd.DataFrame:
    """Disaggregate Global eval rows back to region segments via revenue share."""
    if global_eval_rows.empty or not target_regions:
        return pd.DataFrame(columns=global_eval_rows.columns)

    shares = compute_training_revenue_shares(train_df, target_regions)
    rows: list[dict] = []

    for _, row in global_eval_rows.iterrows():
        for region in target_regions:
            share = float(shares.get(region, 0.0))
            rows.append(
                {
                    "fold_id": row["fold_id"],
                    "model_name": row["model_name"],
                    "forecast_period": row["forecast_period"],
                    "segment": segment_for_region(region),
                    "expected_net_revenue_usd": float(row["expected_net_revenue_usd"]) * share,
                    "actual_net_revenue_usd": None,
                    "base_sales_intercept": (
                        float(row["base_sales_intercept"]) * share
                        if pd.notna(row["base_sales_intercept"])
                        else None
                    ),
                    "mean_saturation_point": (
                        float(row["mean_saturation_point"])
                        if pd.notna(row["mean_saturation_point"])
                        else None
                    ),
                    "prior_source": row["prior_source"],
                    "confidence_weight": -1,
                }
            )

    return pd.DataFrame(rows)


def get_reports_dir() -> Path:
    """Return a writable reports directory with safe fallback."""
    candidates: list[Path] = []

    env_dir = os.environ.get("MMM_REPORTS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(Path(__file__).resolve().parents[2] / "reports")
    candidates.append(Path("/tmp/mmm_reports"))

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".mmm_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            logger.info("Using MMM reports directory: %s", candidate)
            return candidate
        except Exception:  # pragma: no cover - filesystem/env specific
            continue

    # Last resort: let upstream error show explicit path.
    fallback = Path(__file__).resolve().parents[2] / "reports"
    logger.warning(
        "No writable reports directory found among %s. Using %s and letting write errors surface.",
        [str(path) for path in candidates],
        fallback,
    )
    return fallback


def get_trace_artifact_path(model_name: str, fold_id: str) -> Path:
    """Return standardized posterior trace artifact path."""
    safe_model_name = model_name.replace(" ", "_")
    return get_reports_dir() / f"posterior_trace_{safe_model_name}_{fold_id}.nc"


def get_ppc_artifact_path(model_name: str, fold_id: str) -> Path:
    """Return standardized posterior predictive artifact path."""
    safe_model_name = model_name.replace(" ", "_")
    return get_reports_dir() / f"posterior_predictive_{safe_model_name}_{fold_id}.npy"


def get_ols_coeff_artifact_path(fold_id: str, extension: str = "csv") -> Path:
    """Return standardized OLS coefficient artifact path."""
    ext = extension.lstrip(".")
    return get_reports_dir() / f"ols_channel_coeffs_{fold_id}.{ext}"
