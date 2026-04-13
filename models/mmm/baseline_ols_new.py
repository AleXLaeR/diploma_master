"""Baseline OLS MMM rewrite for task #9."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

from models.mmm.channel_contribs_new import (
    build_contrib_rows,
    replicate_global_shares_to_regions,
    shares_from_weekly_contribs,
)
from models.mmm.common_new import (
    EXOG_COLUMNS,
    SPEND_COLUMNS,
    aggregate_to_global,
    check_fallback_needed,
    disaggregate_global_eval_rows,
    get_bq_client,
    get_ols_coeff_artifact_path,
    load_mmm_data,
    segment_for_region,
)
from models.mmm.persistence_new import (
    build_eval_mmm_row,
    write_eval_mmm_to_bq,
    write_mmm_channel_contribs_to_bq,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "Baseline_MMM_Reg"
_ALL_REGRESSORS = [*SPEND_COLUMNS, *EXOG_COLUMNS]


def _stl_decompose(series: pd.Series, period: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply STL decomposition and return trend/seasonal/residual arrays."""
    result = STL(series, period=period, robust=True).fit()
    return result.trend.values, result.seasonal.values, result.resid.values


def _fit_region(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    region_label: str,
    fold_id: str,
    confidence_weight: int,
) -> tuple[list[dict[str, Any]], dict[str, float], pd.DataFrame]:
    """Fit OLS on STL residuals and return eval rows + modeled channel shares."""
    y_train = train_df["total_net_revenue_usd"].astype(float).reset_index(drop=True)
    x_train_raw = train_df[_ALL_REGRESSORS].astype(float).reset_index(drop=True)
    x_holdout_raw = holdout_df[_ALL_REGRESSORS].astype(float).reset_index(drop=True)
    holdout_dates = holdout_df["date_week"].reset_index(drop=True)

    spend_min = x_train_raw[SPEND_COLUMNS].min()
    spend_max = x_train_raw[SPEND_COLUMNS].max()
    spend_range = (spend_max - spend_min).replace(0.0, 1e-8)

    x_train = x_train_raw.copy()
    x_holdout = x_holdout_raw.copy()
    x_train[SPEND_COLUMNS] = (x_train_raw[SPEND_COLUMNS] - spend_min) / spend_range
    x_holdout[SPEND_COLUMNS] = (x_holdout_raw[SPEND_COLUMNS] - spend_min) / spend_range

    t_train = len(y_train)
    stl_period = min(13, max(3, t_train // 2))

    try:
        trend, seasonal, residual = _stl_decompose(y_train, period=stl_period)
    except Exception as exc:  # pragma: no cover - statsmodels edge path
        logger.warning("[%s] STL failed (%s); fallback to level OLS.", region_label, exc)
        trend = np.zeros(t_train, dtype=float)
        seasonal = np.zeros(t_train, dtype=float)
        residual = y_train.to_numpy(dtype=float)

    x_train_const = sm.add_constant(x_train.values, prepend=True, has_constant="add")
    model = sm.OLS(residual, x_train_const)
    result = model.fit()

    n_holdout = len(holdout_df)
    trend_mean = float(np.mean(trend[max(0, t_train - 8):]))
    trend_forecast = np.full(n_holdout, trend_mean, dtype=float)

    seasonal_tail = seasonal[-stl_period:] if stl_period <= len(seasonal) else seasonal
    seasonal_forecast = np.array(
        [float(seasonal_tail[idx % len(seasonal_tail)]) for idx in range(n_holdout)],
        dtype=float,
    )

    params = np.asarray(result.params, dtype=float)
    intercept = float(params[0])
    spend_betas = params[1 : 1 + len(SPEND_COLUMNS)]
    exog_betas = params[1 + len(SPEND_COLUMNS) :]

    exog_holdout = x_holdout[EXOG_COLUMNS].to_numpy(dtype=float)
    exog_contrib = np.dot(exog_holdout, exog_betas)

    # Operational definition required by task #9:
    # base_sales_intercept_t = trend_t + seasonal_t + intercept + exog_contrib_t
    base_sales_intercept = trend_forecast + seasonal_forecast + intercept + exog_contrib

    spend_holdout = x_holdout[SPEND_COLUMNS].to_numpy(dtype=float)
    spend_contrib_raw = spend_holdout * spend_betas[np.newaxis, :]
    media_weekly_contrib = np.maximum(spend_contrib_raw, 0.0)

    forecast = np.maximum(base_sales_intercept + spend_contrib_raw.sum(axis=1), 0.0)

    eval_rows: list[dict[str, Any]] = []
    segment = segment_for_region(region_label)
    for idx, date_week in enumerate(holdout_dates):
        eval_rows.append(
            build_eval_mmm_row(
                fold_id=fold_id,
                model_name=MODEL_NAME,
                forecast_period=date_week,
                segment=segment,
                expected_net_revenue_usd=float(forecast[idx]),
                base_sales_intercept=float(base_sales_intercept[idx]),
                mean_saturation_point=None,
                prior_source=None,
                confidence_weight=confidence_weight,
            )
        )

    weekly_contribs = pd.DataFrame(
        {
            "gads:search": media_weekly_contrib[:, 0],
            "gads:youtube": media_weekly_contrib[:, 1],
            "gads:discover": media_weekly_contrib[:, 2],
            "metads:inst": media_weekly_contrib[:, 3],
            "metads:fb": media_weekly_contrib[:, 4],
            "tiktok": media_weekly_contrib[:, 5],
        }
    )
    shares = shares_from_weekly_contribs(
        weekly_contribs=weekly_contribs,
        fold_id=fold_id,
        model_name=MODEL_NAME,
        segment=segment,
    )

    coeff_rows = []
    regressor_names = ["const", *_ALL_REGRESSORS]
    for regressor_name, coefficient, p_value in zip(
        regressor_names,
        result.params,
        result.pvalues,
        strict=True,
    ):
        coeff_rows.append(
            {
                "fold_id": fold_id,
                "model_name": MODEL_NAME,
                "segment": segment,
                "fit_region": region_label,
                "confidence_weight": confidence_weight,
                "regressor": regressor_name,
                "coefficient": float(coefficient),
                "p_value": float(p_value),
            }
        )

    return eval_rows, shares, pd.DataFrame(coeff_rows)


def run(
    bq_project: str,
    bq_dataset: str,
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
) -> None:
    """Run baseline OLS MMM with region->global fallback and contract-aligned writes."""
    client = get_bq_client()

    train_all = load_mmm_data(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
        train_end=train_end,
    )
    holdout_all = load_mmm_data(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )

    if train_all.empty:
        logger.error("%s fold=%s: empty training data, skipping.", MODEL_NAME, fold_id)
        return

    eval_rows: list[dict[str, Any]] = []
    shares_by_segment: dict[str, dict[str, float]] = {}
    coeff_artifacts: list[pd.DataFrame] = []

    fallback_regions: list[str] = []
    all_regions = sorted(train_all["macro_region"].dropna().unique().tolist())

    for region in all_regions:
        tr = train_all[train_all["macro_region"] == region].sort_values("date_week")
        ho = holdout_all[holdout_all["macro_region"] == region].sort_values("date_week")

        if ho.empty:
            continue

        if check_fallback_needed(tr):
            fallback_regions.append(region)
            continue

        region_eval_rows, region_shares, coeff_df = _fit_region(
            train_df=tr,
            holdout_df=ho,
            region_label=region,
            fold_id=fold_id,
            confidence_weight=0,
        )
        eval_rows.extend(region_eval_rows)
        shares_by_segment[segment_for_region(region)] = region_shares
        coeff_artifacts.append(coeff_df)

    uncovered = [region for region in fallback_regions if segment_for_region(region) not in shares_by_segment]
    if uncovered:
        global_eval_rows, global_shares, global_coeff_df = _fit_region(
            train_df=aggregate_to_global(train_all),
            holdout_df=aggregate_to_global(holdout_all),
            region_label="Global",
            fold_id=fold_id,
            confidence_weight=-1,
        )
        coeff_artifacts.append(global_coeff_df)

        disagg_eval = disaggregate_global_eval_rows(
            global_eval_rows=pd.DataFrame(global_eval_rows),
            train_df=train_all,
            target_regions=uncovered,
        )
        eval_rows.extend(disagg_eval.to_dict("records"))

        shares_by_segment.update(
            replicate_global_shares_to_regions(
                global_shares=global_shares,
                regions=uncovered,
            )
        )

    if not eval_rows:
        logger.error("%s fold=%s: no rows generated.", MODEL_NAME, fold_id)
        return

    coeff_path = get_ols_coeff_artifact_path(fold_id, extension="csv")
    if coeff_artifacts:
        pd.concat(coeff_artifacts, ignore_index=True).to_csv(coeff_path, index=False)
        logger.info("Saved OLS coefficients to %s", coeff_path)

    eval_df = pd.DataFrame(eval_rows)
    write_eval_mmm_to_bq(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        model_name=MODEL_NAME,
        fold_id=fold_id,
        eval_rows=eval_df,
    )

    contrib_rows = build_contrib_rows(
        fold_id=fold_id,
        model_name=MODEL_NAME,
        shares_by_segment=shares_by_segment,
    )
    write_mmm_channel_contribs_to_bq(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        model_name=MODEL_NAME,
        fold_id=fold_id,
        contrib_rows=pd.DataFrame(contrib_rows),
    )

    logger.info(
        "%s fold=%s complete. eval_rows=%d contrib_rows=%d",
        MODEL_NAME,
        fold_id,
        len(eval_df),
        len(contrib_rows),
    )
