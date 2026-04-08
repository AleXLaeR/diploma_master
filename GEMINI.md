This document establishes a Constitution & Project Manifest for AI Assistant consumers

## High-Level Description
This repository contains the practical implementation (Chapter 3) of an experiment conducted for the Master's thesis in Software Engineering field. The thesis is cross-disciplinary, touching upon SE, Digital Marketing and Marketing Analytics domains. Its topic is: "Research of programmatic approaches for automating marketing activity performance analysis based on statistical models".
As a business subject of synthesized experimental scenario, a real B2C enterprise was chosen. Its actual marketing activity was recorded for 11 consecutive months (from 2021-04-01 to 2022-02-28). This company operates by a subscription-based model, offering digital educational app to customers in WEEKLY, MONTHLY, or 3_MONTH packages. Customers have an option of purchasing a trial period for every product at a discounted price.

The experiment is conducted on the semi-synthesized data, which will be generated based on the real business data and business logic. However, to further proof the main thesis statement (probabilistic >> deterministic), the initial data is deliberately made "noisy" with missing values, outliers, missing statistical masses and other imperfections. This is to generate "business uncertainty" to the models, which is a common case in real-world marketing analytics - and show that simplistic heurestic models (like current deterministic strategy) are not robust to such conditions.

## Prior assumptions and What-to-Improve
Currently this business has a deterministic analysis strategy, consisting of Last-Click rule-based attribution for tactical insights about channel performance, Linear multivariate Regression to estimate justified media-mix budget allocations for the near future, and fitted decay curves for customer retention analysis. Chapter 3 aims to improve on this simplified strategy, by evaluating & comparing the compositional viability of inherently different statistical approaches / analytical planes (attribution modeling, media-mix modeling, survival analysis) to proof superiority of probabilistic, data-driven methods over the current deterministic strategy. The comparison would be performed on both business (KPIs like CLV, CAC, ROAS) and programmatic (e.g., out-of-sample WAPE/Bias or in-sample R^2) grounding.

- **CRITICAL**: When working with the data, always remember that the initial dataset should be **logically split onto two parts**: Training (used by models to establish patterns in data) that includes first 8 months of observation (2021-04-01 to 2021-11-30, 35 weeks), and Validation (used to compare the forecasted curves against the factual performance) that includes the remaining 3 months (2021-12-01 to 2022-02-28, 13 weeks). This training/validation split however IS NOT optimistic; severe sparsity of input data and shortness of holdout period requires to rely on rolling-origin backtesting as the only viable option for evaluating the models' performance.

## Scope of Research
Core Paradigm: strictly statistical and probabilistic modeling.
Exclusion Zone: ML or black-box AI are **STRICTLY PROHIBITED**.

The research synthesizes three complementary analytical planes operating at different granularities to form a holistic marketing analytics pipeline (Micro + Customer Base $\approx$ Macro):
- Micro-level: Data-Driven Attribution (DDA) for CAC optimization and per-channel acquisition efficiency.
    - Domain: new customer acquisition revenue (`rebill_number = 0`, paid channels only).
- Macro-level: Media-Mix Modeling (MMM) for strategic budget ceilings and marginal saturation modeling.
    - Domain: total system-wide revenue (all transactions).
- Customer Base: Survival Analysis (econometric frailty models like sBG) for CRR estimation and granular LTV prediction.
    - Domain: recurring subscription revenue (`rebill_number > 0`).

Primary Objective: to prove that a synthesized, multi-plane probabilistic pipeline—where each model solves varying granularities of the customer journey—is capable of extracting superior actionable insights and defining optimal Portfolio ROAS, outperforming naive uncoupled deterministic heuristics (Last-Click + OLS) currently used by the business subject.

## Technical Stack
- Data Warehouse (DWH): Google BigQuery, used as a single source-of-truth for the experiment's data.
- __google-cloud-bigquery__ package for interacting with the BigQuery.
- Language: Python 3.10+, UV as the package manager.
- Task Orchestration (ETL/ELT): Apache Airflow (running locally via Docker Compose).
- Probabilistic Programming (MMM): PyMC for Bayesian inference and posterior sampling.
- `bigquery_toolbox` MCP for YOUR BigQuery interactions.

## Context Navigation Map
__Agent Instruction: Do not attempt to guess specific implementations. Always refer to the specialized manifest files listed below for detailed instructions on data schemas, algorithms & mathematical formulas, and overall pipeline logic.__

### Reference Index for Raw/Transformed Data Schematics: **docs/data/index.md**.
- Contains the Initial DWH schemas (purchases, users_attribution, insights, countries) provided by business observations.
- Defines the Reverse-Engineering augmentation logic for reconstructing multi-touch client journeys and dynamically allocating macro-budgets into intermediate views.
- Defines the strict final Output Contracts that all algorithms must populate for in-domain & cross-plane comparison.

### Reference Index for Algorithm Specifications: **docs/algorithms/index.md**.
__Each of the 3 aforementioned approaches (DDA, MMM, Survival) is represented in two different algorithms to determine the most applicable one.__

### Engineering Guidelines
- **docs/orchestration_guidelines.md**: defines task execution flow, rules for DAG composition, task idempotency, parallel execution logic.
- **docs/comparison/comparison_framework.md**: defines metrics and procedures for comparing the viability of three approaches. Includes the reasoning behind the shift from optimistic dataset split towards advanced ROCV methodology.

---
## General Instructions
 - When in **planning** mode, always generate impl. plans (as per __docs/rfc_template.md__), evaluate what project areas it certainly will influence, and what edge cases it may introduce to the overall experiment flow.
 - Every major codebase change MUST be reflected in the relevant documentation (e.g., new param boundaries for MLE-based survival model optimizer should be documented in __docs/algorithms/survival_models.md__).
 - After every completed & agreed-upon implementation of a feature, manually compact the current context window into a centralized summary file within the **features** folder and prepend the summary to it (create if not exists).