# Context & Objective
This document defines the final process before the cross-model comparison. To prevent "data leakage" (models accidentally seeing the future), all predictive algorithms output blind forecasts into their respective domain tables, leaving the `actual_*` columns `null`. The objective of this process is to act as the impartial "Judge". A separate pipeline will execute 3 decoupled SQL operations to calculate the ground truth specific to each domain, and update the respective evaluation tables.

# 1. Input Data
- Predictions: `eval_dda`, `eval_survival`, `eval_mmm` (target tables with populated `expected_*` but empty `actual_*`).
- Ground Truth: augmented `purchases` table (transaction ledger used by the modeling pipeline; holdout weeks only).
- Mapping: `users_attribution_imputed` (to filter out non-paid attributions) and fold-scoped `insights_channel_spend`.

# 2. Domain-Specific Factual Calculation Logic
Because the models operate on entirely different analytical planes, this process must calculate the factual reality using tailored logic for each domain table.

**Universal Financial Rule: for all calculations below, factual net revenue is defined strictly as `SUM(order_amount_in_usd - COALESCE(refund_amount_in_usd, 0))` for orders with status IN ('approved', 'settled_ok', 'refunded').**
Note: if a model predicted volume for a `segment`, but the factual query returns NULL (e.g., 0 purchases happened that week for that segment), the SQL must use `COALESCE(factual_value, 0.0)` to ensure metrics calculate correctly against a zero-revenue reality.


## 2.1. Survival Analysis Plane (`eval_survival`)
Domain Focus: Customer Retention / Recurring Revenue.

SQL Logic:
- Filter: WHERE `rebill_number > 0` AND `order_status IN ('approved', 'settled_ok', 'refunded')`.
- Grouping: group the purchases by `segment`, derived `rebill_period_t`, and explicitly matched by `fold_id`.
- Matching: update the `actual_active_users` and `actual_lifetime_value_usd` columns where the `segment`, `fold_id`, and `rebill_period_t` are matching.


## 2.2. Data-Driven Attribution Plane (`eval_dda`)
Domain Focus: New Customer Acquisition Efficiency.

SQL Logic:
- Filter: WHERE `rebill_number = 0` AND `order_status IN ('approved', 'settled_ok', 'refunded')`.
- Exclusion: any non-paid attributions.
- Grouping: group total paid conversions and total holdout block spend by `fold_id` and `forecast_period` (week), with spend filtered by matching `insights_channel_spend.fold_id`.
- Matching: update `actual_conversions` and calculate `actual_cac_usd = Total Factual Spend / Total Factual Conversions` per group.


## 2.3. Media Mix Modeling Plane (`eval_mmm` and `mmm_channel_contribs`)
Domain Focus: Total Macro-Level Revenue.

SQL Logic:
- Filter: include all transactions where `order_date` falls within the `forecast_period`.
- Grouping: group by `fold_id`, `segment` and `forecast_period`.
- Matching: update `actual_net_revenue_usd` column within that exact fold context (within `eval_mmm` table). Also calculate `actual_contrib_*` columns from empirical reality by joining `touchpoints_log` and `insights_channel_spend` (within `mmm_channel_contribs` table).
