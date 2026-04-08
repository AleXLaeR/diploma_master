# Context & Objective
This document defines the model-specific data marts used to project the raw/augmented dataset into acceptable intrinsic formats. These tables serve as the direct inputs (X, y) for the statistical algorithms.

# 1. `cohorts_retention` table (for survival analysis)
Flattened survival analysis matrix. Tracks the exact decay of a specific cohort over discrete billing cycles.
Data Volume: ~80-90k rows (3 subscription types x 17 macro-regions x 48 weeks x ~8-10 rebills x 4 folds).
Doesn't require a `fold_id` attribute, as cohort lifetime can be directly observed within fold boundaries.
Partitioned in BigQuery MONTHLY by `acquisition_week` column.

Schema:
- cohort_id: template format - '2021_Week-40_SUB_MONTHLY_NA_US'.
- acquisition_week (Date): First day of the cohort's acquisition week.
- subscription_type: SUB_WEEKLY, SUB_MONTHLY, SUB_3_MONTH.
- macro_region: 17-region mapping.
- rebill_number (Int): discrete time period $t$ ($t=0$ = trial).
- active_users_at_t (Int): users who successfully paid this specific rebill.
- churned_users_at_t (Int): users who cancelled exactly at this rebill period (max_rebill = $t$).

Data Preview:
cohort_id | acquisition_week | subscription_type | country_group | rebill_number | active_users_at_t | churned_users_at_t
2021_Week-32_SUB_MONTHLY_NA_US | 2021-08-02 | SUB_MONTHLY | NA_US | 0 | 1500 | 250 |
2021_Week-32_SUB_MONTHLY_NA_US | 2021-08-02 | SUB_MONTHLY | NA_US | 1 | 1250 | 400 |


# 2. `mmm_timeseries` table (for MMM)
Denormalized, weekly time-series integrating marketing spend, baseline revenue, and exogenous factors.
Data Volume: 3264 rows (48 weeks x 17 macro-regions x 4 folds).
Partitioned in BigQuery MONTHLY by `date_week` column.

Schema:
- fold_id: ROCV fold.
- date_week (Date): first day of the week (Monday).
- macro_region: 17-region macro-group label.
- total_net_revenue_usd (Float): total revenue (approved + settled_ok) minus refunds.
- spend_gads_search, spend_gads_youtube, spend_gads_discover, spend_metads_inst, spend_metads_fb, spend_tiktok (Float): per-channel ad spend (joined by fold_id).
- exog var #1: revenue_anomaly_score (Float): per-region 8-week rolling z-score (microeconomic factor).
- exog var #2, 3: fourier_cos_q1, fourier_sin_q1 (Float): quarterly harmonic capturing seasonality.
- exog var #4, 5: is_sep_nov_trough, is_structural_peak (Boolean): structural change indicators derived from z-score.


# 3. `attribution_paths` table (for DDA)
Sequences of advertising touches leading to either a terminal success (Conversion) or failure (Null).
Data Volume: a single journey per customer aggregation from `touchpoints_log`.

Schema:
- user_id.
- journey: chronological channel sequence (e.g. 'tiktok > metads:fb > gads:search').
- is_converted (Boolean): whether the journey ended in a conversion.
- conversion_value_usd (Float): net acquisition value (initial purchase - refunds).

Data Preview:
user_id | journey | is_converted | conversion_value_usd
<uuid> | tiktok > metads:fb > gads:search | true | 49.99 |
<uuid> | metads:inst > gads:youtube | false | 0.00 |
