# Context & Objective
This document defines where to locate the exact algorithm specifications required to implement statistical models as part of the experiment, describing corresponding intrinsic details. It contains references to the attribution modeling algorithms (Baseline Last Click, Markov Chains, Shapley Value), MMM (Baseline Regression, Bayesian Non-Linear Regression, BSTS), Survival Analysis (Baseline exponential, BdW, sBG).

1. Data-Driven Attribution (DDA): **./dda_models.md**
2. Media-Mix Modeling (MMM): **./mmm_models.md**
3. Survival Analysis (Retention Projection): **./survival_models.md**

__IMPORTANT NOTE: to prevent "data leakage" (models accidentally seeing the future), ALL predictive algorithms (sBG, BdW, Markov, Shapley, MMM) output blind financial forecasts into `eval_*` tables, leaving the `actual_*` columns `null` initially. An external consolidation process will query the raw holdout dataset, calculate the factual net revenue specific to each model's analytical domain, and update these tables with the ground truth before proceeding with the models' accuracy evaluations.__