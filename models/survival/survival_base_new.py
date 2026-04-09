"""
survival_base_new.py
====================
Abstract LTVModel base class + full pipeline orchestrator.
Spec reference: docs/algorithms/survival_models.md

Changes vs. deprecated models/survival_base.py
-----------------------------------------------
- Correct BQ column names: `subscription_type` / `macro_region`
  (table uses these, not `product_id` / `country_group`).
- N_initial fetched from `purchases WHERE rebill_number = 0`
  (cohorts_retention starts at rebill_number=1, no t=0 row).
- Fallback 1 (confidence_weight=-1): Pooled-MLE + Weighted Disaggregation.
  Segment: `{week}_{sub_type}_MACRO_GLOBAL`.
- Fallback 2 (confidence_weight=-2): Monthly aggregation for subscription_type.
  Segment: `{YYYY-MM}_{sub_type}_ALL_SUB_`.
- Boundary-hit (confidence_weight=-0.5): any fitted param at lower/upper bound.
- GammaGamma monetary model feeds expected_ltv_usd in eval_survival.
- actual_ltv_usd is intentionally NULL; populated by consolidate_survival.sql.
- survival_model_params table persisted per fold for BdW only.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMALL_COHORT_THRESHOLD: int = 50   # accept lower-bound solutions below this


# ===========================================================================
# Abstract Base Class
# ===========================================================================

class LTVModel(ABC):
    """Abstract base for probability-based survival models (sBG, BdW, Baseline)."""

    def __init__(self, bounds: List[List[float]]) -> None:
        self.bounds = np.array(bounds)
        self.params: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @abstractmethod
    def predicted_survival(self, t: int) -> float:
        """Return S(t) — probability of surviving past period t."""

    @abstractmethod
    def log_likelihood_multi_cohort(
        self, params: np.ndarray, data: List[List[int]]
    ) -> float:
        """Compute the *negative* log-likelihood across all cohorts (to minimise)."""

    # ------------------------------------------------------------------
    def get_initial_guess(self) -> np.ndarray:
        """Default starting point: all parameters = 1.0."""
        return np.ones(len(self.bounds))

    # ------------------------------------------------------------------
    def optimize(
        self,
        data: List[List[int]],
        n_restarts: int = 5,
        initial_users: int = 0,
    ) -> bool:
        """
        Fit MLE parameters via L-BFGS-B with multiple random restarts.

        Parameters
        ----------
        data            : Each inner list is one cohort vector [N0, N1, …, N_T_obs].
        n_restarts      : Number of random start points (rescues ~20-30% of failures).
        initial_users   : Real user count.  If < SMALL_COHORT_THRESHOLD, lower-bound
                          solutions are accepted (homogeneous churn is plausible).
        """
        if not data or all(len(c) < 2 for c in data):
            return False

        # Spec §2.5 mandatory: T_obs ≤ 1 is underdetermined for sBG (2 params,
        # 1 transition) and BdW (3 params, 1 transition). Route to fallback.
        max_t_obs = max(len(c) - 1 for c in data)
        if max_t_obs < 2:
            return False

        accept_lower = initial_users < SMALL_COHORT_THRESHOLD
        best_res = None
        best_ll  = np.inf

        rng = np.random.default_rng(seed=42)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        candidates = [self.get_initial_guess()]
        for _ in range(n_restarts - 1):
            candidates.append(rng.uniform(lo + 1e-3, np.minimum(hi - 1e-3, 5.0)))

        for x0 in candidates:
            res = minimize(
                self.log_likelihood_multi_cohort,
                x0,
                args=(data,),
                bounds=self.bounds,
                method="L-BFGS-B",
            )
            if not res.success:
                continue
            at_lower = np.isclose(res.x, lo, rtol=1e-3, atol=1e-4)
            if at_lower.any() and not accept_lower:
                continue
            if res.fun < best_ll:
                best_ll  = res.fun
                best_res = res

        if best_res is None:
            logger.warning("optimize: all restarts failed (initial_users=%d).", initial_users)
            return False

        self.params = best_res.x
        return True

    # ------------------------------------------------------------------
    def hits_boundary(self) -> bool:
        """Return True if any fitted param is at its lower or upper bound."""
        if self.params is None:
            return False
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return bool(
            np.isclose(self.params, lo, rtol=1e-3, atol=1e-4).any()
            or np.isclose(self.params, hi, rtol=1e-3, atol=1e-4).any()
        )


# ===========================================================================
# ISO-week / date helpers
# ===========================================================================

def parse_acquisition_week(val: Any) -> datetime:
    """
    Convert a BQ DATE value (or string) representing the acquisition_week
    to a Python datetime at midnight Monday.
    """
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.to_datetime(val).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    s = str(val).strip()
    try:
        if "-W" in s:
            parts = s.split("-W")
            return datetime.fromisocalendar(int(parts[0]), int(parts[1]), 1)
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        logger.warning("Cannot parse acquisition_week '%s'; defaulting to 2021-04-01.", val)
        return datetime(2021, 4, 1)


def acq_week_iso_str(acq_dt: datetime) -> str:
    """Return 'YYYY-Www' string for a Monday-aligned acquisition datetime."""
    iso_yr, iso_wk, _ = acq_dt.isocalendar()
    return f"{iso_yr}-W{iso_wk:02d}"


def acq_month_str(acq_dt: datetime) -> str:
    """Return 'YYYY-MM' string for Fallback 2 monthly aggregation."""
    return acq_dt.strftime("%Y-%m")


# ===========================================================================
# Output contract translation
# ===========================================================================

def translate_survival_to_output(
    predictions: List[Dict[str, Any]],
    model_name: str,
    fold_id: str,
    gg_models: dict,          # {sub_type: GammaGammaModel}
    refund_rates: dict,       # {sub_type: float}
    holdout_weeks: List[pd.Timestamp],
) -> pd.DataFrame:
    """
    Convert per-cohort prediction dicts into the eval_survival output contract.

    Computes:
        expected_active_users = N_initial × S(t_target)
        expected_ltv_usd      = N_initial × Σ_{t=1}^{T_horizon} S(t) × E[M] × (1 − refund_rate)
            where T_horizon is number of weeks left from forecast_period to holdout_end.

    actual_active_users and actual_ltv_usd are left NULL (filled by consolidation SQL).
    """
    if not predictions:
        return pd.DataFrame()

    from models.survival.gamma_gamma_new import GammaGammaModel  # noqa: avoid circular

    rows = []
    for pred in predictions:
        sub_type  = pred["subscription_type"]
        seg       = pred["segment"]
        n0        = float(pred["N_initial"])
        s_t       = float(np.clip(pred["S_t_target"], 0.0, 1.0))
        t_target  = int(pred["t_target"])
        model_obj = pred["model_obj"]            # fitted LTVModel instance
        forecast_period = pred["forecast_period"]

        expected_active = n0 * s_t

        # Gamma-Gamma LTV: Σ_{t=1}^{T_h} S(t) × E[M] × (1 − refund_rate)
        gg: Optional[GammaGammaModel] = gg_models.get(sub_type)
        em = gg.expected_monetary_value() if gg is not None else 0.0
        rr = float(refund_rates.get(sub_type, 0.0))

        # Horizon: from t_target onward until end of holdout
        # T_horizon = number of remaining holdout steps from t_target
        # Approximate as (max holdout rebill - t_target + 1)
        max_holdout_t = pred.get("max_holdout_t", t_target)
        ltv_horizon_sum = 0.0
        if em > 0 and model_obj is not None:
            for tt in range(t_target, max_holdout_t + 1):
                ltv_horizon_sum += model_obj.predicted_survival(tt)
        expected_ltv = n0 * ltv_horizon_sum * em * (1.0 - rr)

        rows.append({
            "fold_id":               fold_id,
            "model_name":            model_name,
            "forecast_period":       forecast_period,
            "segment":               seg,
            "rebill_period_t":       t_target,
            "expected_active_users": round(expected_active, 4),
            "actual_active_users":   None,
            "expected_ltv_usd":      round(expected_ltv, 4),
            "actual_ltv_usd":        None,
            "confidence_weight":     float(pred["confidence_weight"]),
        })

    df = pd.DataFrame(rows)
    df["forecast_period"] = pd.to_datetime(df["forecast_period"]).dt.date

    group_cols = [
        "fold_id", "model_name", "forecast_period",
        "segment", "rebill_period_t", "confidence_weight",
    ]
    df_grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            expected_active_users=("expected_active_users", "sum"),
            expected_ltv_usd=("expected_ltv_usd", "sum"),
        )
    )
    df_grouped["actual_active_users"] = None
    df_grouped["actual_ltv_usd"]      = None

    return df_grouped[group_cols + [
        "expected_active_users", "actual_active_users",
        "expected_ltv_usd", "actual_ltv_usd",
    ]]


# ===========================================================================
# Persist survival_model_params
# ===========================================================================

def persist_model_params(
    client: bigquery.Client,
    params_rows: List[Dict[str, Any]],
    fold_id: str,
    project: str,
    dataset: str,
) -> None:
    """Idempotent write of (fold_id, segment, alpha, beta, c)."""
    table_fqn = f"{project}.{dataset}.survival_model_params"
    client.query(
        f"DELETE FROM `{table_fqn}` WHERE fold_id='{fold_id}'"
    ).result()

    if not params_rows:
        return

    df = pd.DataFrame(params_rows)
    schema = [
        bigquery.SchemaField("fold_id",     "STRING",  mode="REQUIRED"),
        bigquery.SchemaField("segment",     "STRING",  mode="REQUIRED"),
        bigquery.SchemaField("alpha",       "FLOAT64"),
        bigquery.SchemaField("beta",        "FLOAT64"),
        bigquery.SchemaField("c",           "FLOAT64"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
    )
    client.load_table_from_dataframe(df, table_fqn, job_config=job_config).result()
    logger.info("Persisted %d param rows for fold=%s", len(df), fold_id)


# ===========================================================================
# BQ write helper (eval_survival — updated schema includes LTV columns)
# ===========================================================================

_EVAL_SURVIVAL_SCHEMA = [
    bigquery.SchemaField("fold_id",               "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("model_name",             "STRING"),
    bigquery.SchemaField("forecast_period",        "DATE"),
    bigquery.SchemaField("segment",                "STRING"),
    bigquery.SchemaField("rebill_period_t",        "INT64"),
    bigquery.SchemaField("expected_active_users",  "FLOAT64"),
    bigquery.SchemaField("actual_active_users",    "INT64"),
    bigquery.SchemaField("expected_ltv_usd",       "FLOAT64"),
    bigquery.SchemaField("actual_ltv_usd",         "FLOAT64"),
    bigquery.SchemaField("confidence_weight",      "FLOAT64"),
]


def write_eval_survival_to_bq(
    client: bigquery.Client,
    df: pd.DataFrame,
    project: str,
    dataset: str,
    model_name: str,
    fold_id: str,
) -> None:
    """Fold-scoped idempotent write to eval_survival."""
    table_fqn = f"{project}.{dataset}.eval_survival"
    client.query(
        f"DELETE FROM `{table_fqn}` WHERE model_name='{model_name}' AND fold_id='{fold_id}'"
    ).result()

    if df.empty:
        logger.warning("%s fold=%s: no eval_survival rows to write.", model_name, fold_id)
        return

    target_cols = [f.name for f in _EVAL_SURVIVAL_SCHEMA]
    upload_df = df.copy()
    for col in target_cols:
        if col not in upload_df.columns:
            upload_df[col] = None
    upload_df = upload_df[target_cols]

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=_EVAL_SURVIVAL_SCHEMA,
    )
    client.load_table_from_dataframe(upload_df, table_fqn, job_config=job_config).result()
    logger.info("%s fold=%s: wrote %d rows to eval_survival.", model_name, fold_id, len(upload_df))


# ===========================================================================
# Main pipeline orchestrator
# ===========================================================================

def execute_survival_model(
    bq_project: str,
    bq_dataset: str,
    model_name: str,
    model_class: type,
    bounds: List[List[float]],
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
) -> None:
    """
    Full pipeline for one model class (sBG, BdW, or Baseline_Survival) for
    a single ROCV fold.

    Steps
    -----
    1. Load cohorts_retention (training window only).
    2. Load N_initial per cohort from purchases (rebill_number=0).
    3. Load product durations.
    4. Load refund_rates for LTV calculation.
    5. Fit Gamma-Gamma monetary model per subscription_type.
    6. Hierarchical MLE with Pooled-MLE disaggregation fallbacks.
    7. Map holdout weeks to discrete rebill periods (trial exclusion).
    8. Build eval_survival rows with expected_ltv_usd.
    9. Persist BdW-only parameter tables and write eval_survival.
    """
    from models.survival.gamma_gamma_new import (
        fetch_monetary_data,
        fit_gamma_gamma_models,
        persist_monetary_params,
    )

    client = bigquery.Client()

    # ------------------------------------------------------------------ 1.
    sql_retention = f"""
        SELECT
            cohort_id,
            acquisition_week,
            subscription_type,
            macro_region,
            rebill_number,
            active_users_at_t,
            churned_users_at_t
        FROM `{bq_project}.{bq_dataset}.cohorts_retention`
        WHERE acquisition_week < '{train_end}'
        ORDER BY acquisition_week, subscription_type, macro_region, rebill_number
    """
    df = client.query(sql_retention).to_dataframe()
    if df.empty:
        logger.error("%s fold=%s: no cohorts_retention data. Aborting.", model_name, fold_id)
        return

    # Normalize DATE-like column for safe comparisons in fallback filters.
    df["acquisition_week"]  = pd.to_datetime(df["acquisition_week"]).dt.date
    df["rebill_number"]     = df["rebill_number"].astype(int)
    df["active_users_at_t"] = df["active_users_at_t"].astype(int)

    # ------------------------------------------------------------------ 2.
    # N_initial (t=0 trial subscribers) are NOT in cohorts_retention.
    # Reconstruct from purchases WHERE rebill_number = 0.
    sql_n0 = f"""
        SELECT
            p.product_id AS subscription_type,
            COALESCE(c.region, 'ROW') AS macro_region,
            DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY)) AS acquisition_week,
            COUNT(DISTINCT p.user_id) AS n_initial
        FROM `{bq_project}.{bq_dataset}.purchases` AS p
        LEFT JOIN `{bq_project}.{bq_dataset}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
            AND ua_imputed.fold_id = '{fold_id}'
            AND ua_imputed.is_synthetic = FALSE
        LEFT JOIN `{bq_project}.{bq_dataset}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        LEFT JOIN `{bq_project}.{bq_dataset}.countries` AS c
            ON COALESCE(ua_imputed.country_code, ua_raw.country_code) = c.country_code
        WHERE
            p.rebill_number = 0
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
            AND CAST(p.order_date AS DATE) < '{train_end}'
        GROUP BY subscription_type, macro_region, acquisition_week
    """
    df_n0 = client.query(sql_n0).to_dataframe()
    df_n0["acquisition_week"] = pd.to_datetime(df_n0["acquisition_week"]).dt.date
    # Build lookup: (subscription_type, macro_region, acquisition_week) -> n_initial
    n0_lookup: Dict[Tuple, int] = {
        (row["subscription_type"], row["macro_region"], row["acquisition_week"]): int(row["n_initial"])
        for _, row in df_n0.iterrows()
    }

    # ------------------------------------------------------------------ 3.
    sql_durations = f"""
        SELECT
            product_id,
            MAX(rebill_duration)  AS duration_days,
            MAX(trial_duration)   AS trial_duration_days
        FROM `{bq_project}.{bq_dataset}.purchases`
        GROUP BY product_id
    """
    df_dur = client.query(sql_durations).to_dataframe()
    duration_map: Dict[str, float] = dict(zip(df_dur["product_id"], df_dur["duration_days"]))
    trial_map:    Dict[str, float] = dict(zip(df_dur["product_id"], df_dur["trial_duration_days"]))

    # ------------------------------------------------------------------ 4.
    sql_refunds = f"""
        SELECT sub_type, refund_rate
        FROM `{bq_project}.{bq_dataset}.refund_rates`
        WHERE fold_id = '{fold_id}'
    """
    df_rr = client.query(sql_refunds).to_dataframe()
    refund_rates: Dict[str, float] = dict(zip(df_rr["sub_type"], df_rr["refund_rate"].astype(float)))

    # ------------------------------------------------------------------ 5.
    df_monetary = fetch_monetary_data(client, bq_project, bq_dataset, train_end)
    expected_sub_types = sorted(df["subscription_type"].dropna().astype(str).unique().tolist())
    gg_models = fit_gamma_gamma_models(
        df_monetary,
        fold_id,
        expected_subscription_types=expected_sub_types,
    )

    # ------------------------------------------------------------------ 6. Holdout week anchors
    holdout_weeks = pd.date_range(
        start=holdout_start, end=holdout_end, freq="W-MON", inclusive="left"
    )

    # ------------------------------------------------------------------ 7. Fallback caches
    # (acq_week, sub_type) -> (success, model.params)
    fb1_cache: Dict[Tuple, Tuple[bool, Optional[np.ndarray]]]  = {}
    # sub_type + month -> (success, model.params)
    fb2_cache: Dict[Tuple, Tuple[bool, Optional[np.ndarray]]] = {}

    predictions: List[Dict[str, Any]] = []
    params_by_segment: Dict[str, Dict[str, Any]] = {}
    segment_to_sub_type: Dict[str, str] = {}

    # Iterate over target granularity: acquisition_week × subscription_type × macro_region
    grouped_target = df.groupby(["acquisition_week", "subscription_type", "macro_region"])

    for (acq_week_raw, sub_type, macro_region), group in grouped_target:
        group = group.sort_values("rebill_number")

        acq_week_date = (
            acq_week_raw.date() if hasattr(acq_week_raw, "date")
            else pd.to_datetime(acq_week_raw).date()
        )
        acq_dt        = parse_acquisition_week(acq_week_raw)
        acq_week_str  = acq_week_iso_str(acq_dt)
        acq_month_key = acq_month_str(acq_dt)

        # rebill vector: [N0, N1, N2, …] — N0 from purchases lookup
        counts_from_t1 = group["active_users_at_t"].tolist()
        n0_key = (sub_type, macro_region, acq_week_date)
        n_initial = n0_lookup.get(n0_key, counts_from_t1[0] if counts_from_t1 else 0)
        cohort_vector = [n_initial] + counts_from_t1   # [N_0, N_1, ..., N_T]

        t_max = group["rebill_number"].max()

        # --------  Target Granularity  (confidence_weight = 0) --------
        model = model_class(bounds)
        success = False
        weight  = 0

        # Guard: spec §2.5 — skip MLE if initial_users < 10 or t_max < 2
        if n_initial >= 10 and t_max >= 2:
            success = model.optimize([cohort_vector], initial_users=n_initial)
            if success:
                weight       = 0
                segment_name = f"{acq_week_str}_{sub_type}_{macro_region}"
                # Boundary-hit check (spec note 4)
                if model.hits_boundary():
                    weight = -0.5

        # --------  Fallback 1: Pooled-MLE + Weighted Disaggregation  --------
        if not success:
            weight = -1
            segment_name = f"{acq_week_str}_{sub_type}_MACRO_GLOBAL"
            fb1_key = (acq_week_raw, sub_type)

            if fb1_key in fb1_cache:
                fb1_ok, cached_params = fb1_cache[fb1_key]
                model.params = cached_params
                success = fb1_ok
            else:
                # Aggregate all regions for this (week, sub_type) into a single vector
                sub_df = df[
                    (df["acquisition_week"] == acq_week_raw)
                    & (df["subscription_type"] == sub_type)
                ]
                pooled_vec: List[int] = []
                pooled_n0 = 0
                if not sub_df.empty:
                    region_groups = sub_df.groupby("macro_region")
                    # Sum active_users_at_t across all regions per rebill_number
                    pooled_by_t = sub_df.groupby("rebill_number")["active_users_at_t"].sum()
                    pooled_t1_plus = [int(v) for v in pooled_by_t.sort_index().values]
                    for region_key in sub_df["macro_region"].unique():
                        rg = sub_df[sub_df["macro_region"] == region_key]
                        acq_w_date = (
                            acq_week_raw.date() if hasattr(acq_week_raw, "date")
                            else pd.to_datetime(acq_week_raw).date()
                        )
                        pooled_n0 += n0_lookup.get((sub_type, region_key, acq_w_date), 0)
                    pooled_vec = [pooled_n0] + pooled_t1_plus

                fb1_guard_t = max(len(pooled_vec) - 1, 0)
                if pooled_n0 >= 10 and fb1_guard_t >= 2 and len(pooled_vec) >= 3:
                    fb1_model = model_class(bounds)
                    success = fb1_model.optimize([pooled_vec], initial_users=pooled_n0)
                    if success:
                        model.params = fb1_model.params
                else:
                    success = False
                fb1_cache[fb1_key] = (success, model.params)

        # --------  Fallback 2: Monthly aggregation  --------
        if not success:
            weight = -2
            segment_name = f"{acq_month_key}_{sub_type}_ALL_SUB_"
            fb2_key = (acq_month_key, sub_type)

            if fb2_key in fb2_cache:
                fb2_ok, cached_params = fb2_cache[fb2_key]
                model.params = cached_params
                success = fb2_ok
            else:
                # Group all weeks in this YYYY-MM for this sub_type
                month_start = pd.to_datetime(acq_month_key + "-01").date()
                if acq_dt.month == 12:
                    month_end = datetime(acq_dt.year + 1, 1, 1).date()
                else:
                    month_end = datetime(acq_dt.year, acq_dt.month + 1, 1).date()

                month_df = df[
                    (df["subscription_type"] == sub_type)
                    & (df["acquisition_week"] >= month_start)
                    & (df["acquisition_week"] < month_end)
                ]
                fb2_n0 = 0
                fb2_by_t: Dict[int, int] = {}
                for region_rk in month_df["macro_region"].unique():
                    for aw in month_df["acquisition_week"].unique():
                        aw_date = pd.to_datetime(aw).date() if not isinstance(aw, date) else aw
                        fb2_n0 += n0_lookup.get((sub_type, region_rk, aw_date), 0)
                for _, mrow in month_df.iterrows():
                    rb = int(mrow["rebill_number"])
                    fb2_by_t[rb] = fb2_by_t.get(rb, 0) + int(mrow["active_users_at_t"])

                if fb2_by_t:
                    sorted_t = sorted(fb2_by_t.keys())
                    fb2_vec = [fb2_n0] + [fb2_by_t[t] for t in sorted_t]
                    fb2_guard_t = len(fb2_vec) - 1
                    if fb2_n0 >= 10 and fb2_guard_t >= 2:
                        fb2_model = model_class(bounds)
                        success = fb2_model.optimize([fb2_vec], initial_users=fb2_n0)
                        if success:
                            model.params = fb2_model.params
                    else:
                        success = False
                else:
                    success = False
                fb2_cache[fb2_key] = (success, model.params)

        # Skip if all fallbacks exhausted
        if not success or model.params is None:
            logger.warning(
                "%s: total failure for (%s, %s, %s). Skipping.",
                model_name, acq_week_str, sub_type, macro_region,
            )
            continue

        # --------  Persist model params  --------
        p_row = {
            "fold_id":    fold_id,
            "segment":    segment_name,
            "alpha":      float(model.params[0]) if len(model.params) > 0 else None,
            "beta":       float(model.params[1]) if len(model.params) > 1 else None,
            "c":          float(model.params[2]) if len(model.params) > 2 else 1.0,
        }
        existing_row = params_by_segment.get(segment_name)
        if existing_row is not None:
            changed = any(
                not np.isclose(
                    float(existing_row.get(k, np.nan)),
                    float(p_row.get(k, np.nan)),
                    rtol=1e-9,
                    atol=1e-12,
                    equal_nan=True,
                )
                for k in ("alpha", "beta", "c")
            )
            if changed:
                logger.warning(
                    "%s fold=%s segment=%s encountered different params across rows; "
                    "keeping latest values.",
                    model_name,
                    fold_id,
                    segment_name,
                )
        params_by_segment[segment_name] = p_row
        segment_to_sub_type[segment_name] = str(sub_type)

        # --------  Determine max holdout t for LTV horizon  --------
        duration_days = float(duration_map.get(sub_type, 30))
        trial_days    = float(trial_map.get(sub_type, 7))

        holdout_end_dt = pd.to_datetime(holdout_end).to_pydatetime()
        last_holdout_diff = (holdout_end_dt - acq_dt).days
        max_holdout_t = max(
            1,
            math.ceil((last_holdout_diff - trial_days) / duration_days),
        )

        # --------  Map each holdout week to t_target  --------
        for holdout_ts in holdout_weeks:
            holdout_date_obj = holdout_ts.to_pydatetime()
            diff_days = (holdout_date_obj - acq_dt).days
            # spec §3.1 step 2
            t_target = math.ceil((diff_days - trial_days) / duration_days)
            if t_target < 1:
                continue   # still within trial window

            s_t = model.predicted_survival(t_target)

            predictions.append({
                "fold_id":           fold_id,
                "segment":           segment_name,
                "subscription_type": sub_type,
                "S_t_target":        float(np.clip(s_t, 0.0, 1.0)),
                "t_target":          t_target,
                "N_initial":         float(n_initial),
                "forecast_period":   holdout_ts.strftime("%Y-%m-%d"),
                "confidence_weight": float(weight),
                "model_obj":         model,
                "max_holdout_t":     max_holdout_t,
            })

    # ------------------------------------------------------------------ 8.
    forecast_df = translate_survival_to_output(
        predictions  = predictions,
        model_name   = model_name,
        fold_id      = fold_id,
        gg_models    = gg_models,
        refund_rates = refund_rates,
        holdout_weeks= holdout_weeks,
    )

    # ------------------------------------------------------------------ 9.
    params_rows = list(params_by_segment.values())

    # Contract: survival_model_params is persisted only for BdW.
    if model_name == "BdW":
        # Persist Gamma-Gamma monetary params with contract-aligned segment mapping.
        persist_monetary_params(
            client,
            gg_models,
            fold_id,
            bq_project,
            bq_dataset,
            segment_to_sub_type=segment_to_sub_type,
        )
        persist_model_params(
            client, params_rows, fold_id, bq_project, bq_dataset
        )
    write_eval_survival_to_bq(
        client, forecast_df, bq_project, bq_dataset, model_name, fold_id
    )
    logger.info(
        "%s fold=%s complete ✓ (%d eval rows, %d param rows)",
        model_name, fold_id, len(forecast_df), len(params_rows),
    )
