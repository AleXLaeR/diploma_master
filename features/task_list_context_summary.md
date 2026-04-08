# Feature Context Summary

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
