Supplementary to: __docs/data/model_specific_marts.md__

# Context & Objective
This document outlines the exact algorithmic steps, heuristics and transformations required to convert the augmented DWH inputs into the final model-specific data marts.


# 1. Generating `cohorts_retention`
Objective: create a flattened retention matrix tracking active and churned users per discrete rebill period.

1. **Country Group Mapping**: map individual country codes into macro-regional groups (join with `countries` table).
2. **Define the Base (Cohort Origin)**: query the `purchases` table and filter for `rebill_number = 0`. This gives the initial list of unique users, their `order_date` (-> `acquisition_week`), their `product_id` (-> `subscription_type`), `country_group` (-> `macro_region`).
3. **Track rebills**: join base users back to purchases (`rebill_number >= 0`).
4. **Find max rebill**: find each user's maximum `rebill_number` to derive churn point.
5. **Aggregate**: for every rebill_number $t$ within the cohort:
    - `active_users_at_t`: distinct count of user_ids who successfully paid order $t$.
    - `churned_users_at_t`: distinct count of user_ids where $t$ equals **their** `MAX(rebill_number)`.

Note: a user's churn point is implicitly defined by their maximum observed `rebill_number`. If it's 2, the pipeline automatically tallies them as `churned_users_at_t` for period 2.


# 2. Generating `attribution_paths`
Objective: flatten chronological events into a single path string per user.

1. **Aggregate**: `user_id` inside `touchpoints_log`, order the touchpoints for the user by `created_at` ascending.
2. **Concatenate**: join the `media_source` values using template "$1 > $2 > ... > $n".
3. **Join Financials**: left join with the `purchases` table where `rebill_number = 0` (only acquisition events).
    - If a match exists: `is_converted = TRUE`, and `conversion_value_usd = SUM(order_amount_in_usd - COALESCE(refund_amount_in_usd, 0))` for that user's initial purchase.
    - If no match (churners): `is_converted = FALSE`, `conversion_value_usd = 0.0`.


# 3. Generating `mmm_timeseries`
Objective: create a weekly macro-level time series integrating revenue, spend, and exogenous factors.

1. **Base Grain**: generate a continuous timeline of training weeks (-> `date_week`) from `purchases` cross-joined with `users_attribution_imputed` and `countries`.
2. **Aggregate Revenue**: sum the net revenue from purchases for that `date_week` and `country_group` (including `refunded`, `settled_ok` statuses to capture deductions).
3. **Aggregate Spend**: join the multi-tenant `insights_channel_spend` table on `fold_id` (AS CTE queried from `rocv_folds`) and `date_week`.
4. **Calculate Exogenous Factors**.
