# Feature Context Summary

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
