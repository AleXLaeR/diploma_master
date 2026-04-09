# Feature Context Summary

## 2026-04-09 — Task 7 Completed (Survival Models Rewrite)
- Created `models/survival/` package with 9 new files (sBG, BdW, Baseline, Gamma-Gamma, base orchestrator, 3 entry-point shims).
- **Critical BQ fix**: corrected column names from `product_id`/`country_group` → `subscription_type`/`macro_region` across all queries.
- **N₀ reconstruction**: `cohorts_retention` has no `rebill_number=0` row; initial cohort sizes fetched from `purchases WHERE rebill_number=0` grouped by (sub_type, macro_region, acq_week).
- **Fallback hierarchy updated**: FB1 now uses Pooled-MLE (pool all regions for week×sub_type, fit once, disaggregate by N₀). FB2 uses monthly aggregation. Segment names corrected to `_MACRO_GLOBAL` / `_ALL_SUB_`. Boundary-hit produces `confidence_weight=-0.5`.
- **T_obs guard**: enforced in both orchestrator (skip MLE when t_max < 2) and `LTVModel.optimize()` (hard-reject underdetermined systems).
- **Gamma-Gamma LTV** (NEW): fits (p,q,γ) per sub_type via MLE; E[M]=q·γ/(q−1); uses `refund_rates` table for net-LTV; persists to `survival_monetary_params`. `expected_ltv_usd` populated in `eval_survival`; `actual_ltv_usd` left NULL for consolidation SQL.
- Updated `eval_survival` BQ schema to include `expected_ltv_usd`/`actual_ltv_usd` columns.
- Updated `dags/dag_phase2_survival.py` imports to new module paths.
- 47/47 automated tests pass (`tests/test_survival_new.py`).

## 2026-04-08 — Task 4 Completed (Consolidation SQL Rewrite for Updated Output Contract)
- Rewrote consolidation SQL for all three evaluation planes to align with updated output contract requirements:
  - `sql/consolidation/consolidate_dda.sql`
  - `sql/consolidation/consolidate_mmm.sql`
  - `sql/consolidation/consolidate_survival.sql`
- DDA consolidation now updates `actual_conversions` and `actual_cac_usd` from holdout factuals, with paid-only attribution filtering and holdout-safe fallback joins (`users_attribution_imputed` + `users_attribution`).
- MMM consolidation now resolves holdout factual `actual_net_revenue_usd` at country, macro-region, and global segment levels, and zero-fills unmatched predicted rows.
- Survival consolidation now updates both `actual_active_users` and `actual_ltv_usd`, with fallback-aware segment matching and cumulative holdout net-revenue translation by `rebill_period_t`.
- Expanded automated validation in:
  - `tests/test_consolidation.py` (contract fields, rendering, target-row-driven zero-fill behavior).
- Ran `pytest -q`: final result `28 passed`.

## 2026-04-08 — Task 3 Completed (Net-Revenue Deduplication + Holdout Attribution Fallback)
- Removed duplicate refund/net-revenue adjustments from MMM model translation layer so forecasts are persisted directly as net values from `mmm_timeseries`.
- Updated `models/dda_common.py` to stop re-summing purchase net revenue where `attribution_paths.conversion_value_usd` already defines net acquisition value.
- Patched flagged downstream SQL to support holdout converters missing fold-scoped imputed attribution:
  - `sql/intermediate/create_channel_cpc_weights.sql`
  - `sql/intermediate/create_insights_channel_spend.sql`
  - `sql/consolidation/consolidate_dda.sql`
- Implemented fallback pattern with raw attribution joins and coalesced keys:
  - `COALESCE(ua_imputed.country_code, ua_raw.country_code)`
  - `COALESCE(ua_imputed.media_source, ua_raw.media_source)`
- Added automated coverage in `tests/test_task3_net_revenue_and_fallbacks.py` for:
  - no second MMM refund netting,
  - DDA revenue source using `attribution_paths`,
  - presence of required SQL fallback joins.
- Ran `pytest -q`: final result `27 passed` after mid-verification fixes to existing guard test path scanning/allowlist behavior.

## 2026-04-08 — Task 2 Completed (`touchpoints_log` Reimplementation, v2)
- Rewrote `scripts/generate_touchpoints.py` from scratch per spec §1 of `intermediate_datasets_implementation.md`.
- v2 fixes 5 critical bugs: (1) join raw `users_attribution` instead of fold-scoped imputed → 16,997 legacy_untracked (not 47k); (2) remove `order_status` filter → 113,801 converters (not 108k); (3) UUID v4 churn IDs; (4) conv_rate from 1st-rebill retention (26.3%, not 90%); (5) multiplicative jitter instead of additive.
- Pure-logic `build_touchpoints_log()` separated from BQ I/O for testability.
- 17 automated pytest tests (`tests/test_generate_touchpoints.py`), all pass.
- BQ output: 800,570 rows, 433,307 users (113.8k converters + 319.5k churn). Path distribution: 35.7%/34.3%/19.9%/10.1% (spec: 35/30/25/10).


## 2026-04-08 — Task 1.2 Completed (Imputed Attribution Migration + DAG Rewire)
- Migrated downstream non-model SQL/script joins from raw `users_attribution` to `users_attribution_imputed`.
- Added fold-safe join conditions where fold scope exists (`ua.fold_id = folds.fold_id`), and `is_synthetic = FALSE` where applicable.
- Reworked `dags/dag_phase1_datasets.py` to include explicit attribution-imputation pipeline stage:
  `create_rocv_folds -> copy_users_attribution_imputed_base -> impute_users_attribution -> downstream marts`.
- Deleted deprecated `dags/dag_phase0_augmentation.py`.
- Updated docs to declare `users_attribution_imputed` as the required attribution source for subsequent queries.
- Added regression test guarding against new raw-table references outside approved exceptions.

## 2026-04-08 — Task 1.1 Completed (`users_attribution_imputed`)
- Implemented fold-aware region-proportional `users_attribution` imputation in `pipelines/users_attribution_imputation.py`.
- Added strict anti-leakage behavior by filtering source rows to each fold's training window from `rocv_folds`.
- Enforced pair eligibility (`country_code`, `media_source` must already exist in raw fold data).
- Enforced conservative density-floor uplift with per-country cap at regional median.
- Added deterministic synthetic row generation and table write to BigQuery `users_attribution_imputed`.
- Added pytest coverage for floor/cap logic, fold leakage prevention, and media eligibility.
- Executed augmentation and verified output with SQL checks (leakage=0, invalid synthetic pairs=0, cap violations=0).
