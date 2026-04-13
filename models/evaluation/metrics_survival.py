"""Survival-specific aggregations and error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dedupe_survival_rows(eval_survival: pd.DataFrame) -> pd.DataFrame:
    """
    Dedupe repeated weekly rows mapping to the same rebill_period_t.

    Aggregation key follows spec intent:
    (fold_id, model_name, segment, rebill_period_t).
    """
    if eval_survival.empty:
        return eval_survival.copy()

    grouped = (
        eval_survival.groupby(
            ["fold_id", "model_name", "segment", "rebill_period_t"],
            as_index=False,
        )
        .agg(
            expected_active_users=("expected_active_users", "mean"),
            actual_active_users=("actual_active_users", "mean"),
            expected_ltv_usd=("expected_ltv_usd", "mean"),
            actual_ltv_usd=("actual_ltv_usd", "mean"),
            n_rows=("forecast_period", "count"),
        )
        .sort_values(["fold_id", "model_name", "segment", "rebill_period_t"])
        .reset_index(drop=True)
    )
    return grouped


def compute_rmse_by_horizon(
    dedup_survival: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = ("Baseline_Survival", "BdW"),
) -> pd.DataFrame:
    """Compute fold-level RMSE over expected vs actual active users by horizon."""
    if dedup_survival.empty:
        return pd.DataFrame(columns=["fold_id", "model_name", "rebill_period_t", "rmse", "cohort_count"])

    subset = dedup_survival[dedup_survival["model_name"].isin(model_names)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["fold_id", "model_name", "rebill_period_t", "rmse", "cohort_count"])

    rows: list[dict[str, object]] = []
    for (fold_id, model_name, horizon), group in subset.groupby(
        ["fold_id", "model_name", "rebill_period_t"]
    ):
        expected = group["expected_active_users"].astype(float)
        actual = group["actual_active_users"].astype(float)
        rmse = float(np.sqrt(np.mean((expected - actual) ** 2)))
        rows.append(
            {
                "fold_id": fold_id,
                "model_name": model_name,
                "rebill_period_t": int(horizon),
                "rmse": rmse,
                "cohort_count": int(group["segment"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fold_id", "model_name", "rebill_period_t"]
    ).reset_index(drop=True)


def build_ltv_curves_table(
    dedup_survival: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = ("Baseline_Survival", "BdW"),
) -> pd.DataFrame:
    """
    Return LTV curve rows for F9, restricted to cohorts with D90 evidence.

    D90 proxy in weekly tables: at least 3 distinct rebill periods observed.
    """
    if dedup_survival.empty:
        return pd.DataFrame()
    subset = dedup_survival[dedup_survival["model_name"].isin(model_names)].copy()
    if subset.empty:
        return pd.DataFrame()

    eligible_segments = (
        subset.groupby(["fold_id", "segment"], as_index=False)
        .agg(max_rebill_t=("rebill_period_t", "max"))
    )
    eligible = eligible_segments[eligible_segments["max_rebill_t"] >= 3][["fold_id", "segment"]]
    if eligible.empty:
        return pd.DataFrame(columns=subset.columns)
    merged = subset.merge(eligible, on=["fold_id", "segment"], how="inner")
    return merged.sort_values(["fold_id", "segment", "model_name", "rebill_period_t"])

