Subsystem: Orchestration & Pipeline Architecture
Tool: Apache Airflow

# Context & Objective
This document defines the active orchestration blueprint for the experiment.
The Airflow DAG layer is responsible for:
- building intermediate/model marts in BigQuery;
- running fold-aware predictive models for DDA, Survival, and MMM;
- consolidating holdout factuals and executing evaluation/visualization.

The architectural rule is compute isolation:
- heavy joins/aggregations stay in BigQuery (`BigQueryInsertJobOperator`);
- mathematical modeling runs in Python / Docker tasks (`PythonOperator`, `DockerOperator`).

# Active DAG Topology

## Phase 0
- `phase_0_dataset_augmentation`
- Produces augmented source tables (`augmented_purchases`, `augmented_users_attribution`).

## Phase 1
- `phase_1_marketing_analytics_pipeline`
- Builds intermediate marts: `insights_channel_spend`, `refund_rate`, `cohorts_retention`, `attribution_paths`, `mmm_timeseries`.

## Phase 2 (ROCV)
- `rocv_master_orchestrator` triggers one `phase2_dda_models` run per fold (`fold_1`..`fold_4`).
- `phase2_dda_models` runs DDA models and forwards fold boundaries downstream.
- `phase2_survival_models` and `phase2_mmm_models` run with the same fold parameters.
- All predictive outputs are written into domain tables: `eval_dda`, `eval_survival`, `eval_mmm`.

## Phase 3
- `phase3_evaluation`
- Runs three parallel consolidation SQL tasks:
  - `consolidate_survival_factuals`
  - `consolidate_dda_factuals`
  - `consolidate_mmm_factuals`
- Then runs `evaluate_and_visualize` (Python) as terminal step.

# Core Engineering Standards

## 1. Fold Integrity
- Every predictive write must be tagged with `fold_id`.
- Consolidation joins must match by `fold_id` where applicable.
- Evaluation/visualization must use `fold_id` from eval tables, not inferred dates.

## 2. Idempotency
- SQL marts: `CREATE OR REPLACE TABLE` for full rebuild tables.
- Model outputs: fold-scoped delete-then-append for a given `(model_name, fold_id)`.
- Consolidation: `MERGE` updates only factual columns.

## 3. Anti-Leakage
- Models write only `expected_*` values; `actual_*` remains null until consolidation.
- Training windows must respect canonical ROCV boundaries from `models/rocv.py`.

## 4. Fallback Traceability
- Hierarchical fallback must be explicit through `confidence_weight`.
- Scenario-B Bayesian rows must declare `prior_source` (`shapley_dda` or `heuristic_fallback`).

