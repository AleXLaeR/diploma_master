# Context & Objective
This document defines where to locate the exact data schemas required for the experimental pipeline. It contains references to the structure of raw data, its instrinsic flaws and ways to augment it, reverse-engineered intermediate views (to solve Last-Click and aggregate-budget biases), down to the model-specific Data Marts, and finally to strict output contracts for cross-model comparison.

1. Raw initial data schema: **./initial/initial_schematics.md**
    - Initial Discoveries & Insights: **./initial/initial_data_discoveries.md**
    - Augmentation spec addressing problems in the raw dataset: **./initial/initial_dataset_augmentation.md**
2. Augmented views built upon raw data: **./intermediate_datasets.md**
    - Implementation specification: **./intermediate_datasets_implementation.md**
3. Model-specific data aggregation views schema: **./model_specific_marts.md**
    - Implementation specification: **./model_specific_marts_implementation.md**
4. Final output schema for cross-model comparison: **./final/output_contract.md**
    - Consolidation phase spec describing how to consolidate the actual holdout data with model forecasts for every scenario: **./final/consolidation_phase.md**

__AI Agent Instruction: you MUST adhere strictly to the column names, data types, and logical constraints defined in the referenced documents. ALL DATA TYPES SHOULD BE INFERRED AS **(String)** UNLESS SPECIFIED OTHERWISE.__
__AI Agent Instruction: for all downstream queries and joins, use `users_attribution_imputed` (fold-aware) instead of raw `users_attribution`.__
