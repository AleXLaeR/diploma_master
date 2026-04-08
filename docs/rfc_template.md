Document Type: AI Agent Implementation Plan (RFC)

**Do not write execution code yet.** Whenever you are asked to implement a feature, algorithm, or DAG task for this project, you must first output a complete RFC using the exact template below. I will review your proposed logic. You may only proceed to write the actual Python or SQL code once I approve this RFC. You must strictly adhere to the overarching rules defined in __docs/data/*.md__ and __docs/orchestration_guidelines.md__.

[RFC-XXX]: [Task or Algorithm Name]
# Context & Objective.
Subsystem: [e.g., Data Engineering, Survival Analysis, MMM, Orchestration]
Target Artifacts: [e.g., sbg_model.py + bdw_model.py, dm_mmm_timeseries.sql, dag_phase_2.py]
Objective: [One paragraph explaining exactly what this implementation achieves and how it serves the broader thesis experiment.]

# 1. Architectural Design & Logic.
Mathematical/Algorithmic Approach: [Briefly describe the core logic. If implementing a model, cite the specific objective function or MCMC sampling strategy. If SQL, describe the grain and aggregation strategy.]

Input Dependencies: * Table/View: [project.dataset.table_name]
Required Columns: [col_1, col_2]
Output Destination: * Table/File: [project.dataset.table_name] or [/path/to/artifact]

# 2. Step-by-Step Implementation Plan.
Step 1: [e.g., Establish the BigQuery client connection using BigQueryInsertJobOperator]
Step 2: [e.g., Pivot the flattened cohort data into a 2D matrix suitable for SciPy]
Step 3: [e.g., Run Nelder-Mead optimization. Catch ConvergenceWarning]
Step 4: [e.g., Append output to final_forecasts_comparison]

# 3. Edge Cases, Imputations & Fallbacks.
Handling Sparsity: [How will this code handle sparse matrices or insufficient data? Specify the exact Dimensionality Reduction / Fallback Hierarchy]
Data Anomalies: [How does this code respect the legacy_untracked or Left Truncation (order_date >= '2021-04-01') rules?]
Known Limitations: [What are the known limitations of this implementation? What errors may it fail on? E.g., "What happens when BigQuery returns 0 rows for a country-week?]

# 4. Idempotency & State Management.
Rerunnability: [Explain exactly how this code can be triggered multiple times without duplicating rows or breaking state. E.g., "I will use a DELETE FROM `eval_mmm` WHERE model_name = 'X' before inserting new predictions."]

# 5. Validation, Sanity Checks and Automated Testing.
Success Criteria: [How will we mathematically or programmatically know this execution succeeded? E.g., "The output dataframe must have exactly 12 rows per country, and no negative revenue values."]
Test before the storm: [How will we test this implementation before deploying it to production? E.g., "I will run this code in a local Docker container with a subset of the data."]
Automated Testing: [How will we test this implementation before deploying it to production? E.g., "I will perform a dry run with only small subset of data to track the changes in the output table."]

# 6. Areas of Influence.
This implementation will influence the following areas of the experiment:
[e.g., Data Quality, Model Accuracy, Pipeline Efficiency, etc.]
...And will change the following files:
[e.g., dags/dag_phase_2.py, models/sbg_model.py, etc.]