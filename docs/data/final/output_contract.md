# Context & Objective
This document defines the exact data structure for the target Output Contract required for the experiment's Evaluation phase.
**CRITICAL Execution Concept**: The evaluation logic operates on a 4-Fold Monthly Expanding Window ROCV. Therefore, each metrics table MUST include a `fold_id` parameter to isolate validation metrics per expanding window.


# 1. `eval_dda` table
**Objective**: evaluate the accuracy of predicted acquisition volume and Marginal CAC.
Schema:
- fold_id: ROCV fold.
- model_name: 'Markov_DDA' or 'Shapley_DDA' or 'Baseline_LastClick'.
- forecast_period: WEEKLY (First day of holdout week).
- expected_conversions (Float): forecasted paid conversion volume per holdout week, computed from holdout paid spend divided by model-specific aggregate CAC.
- actual_conversions (Int): actual observed paid conversions.
- expected_cac_usd (Float): model-specific aggregate CAC (constant within a fold for that model), derived from EWMA-smoothed neutral channel CAC blended by attribution weights.
- actual_cac_usd (Float): factual historical spend divided by factual conversions.
- confidence_weight (Int): confidence marker for fallback-aware analysis.


# 2. `dda_weights` table
Persistent storage for calculated attribution weights from DDA models.
Data Volume: ~72 rows (3 models x 6 paid channels x 4 folds).

Schema:
- fold_id.
- model_name: Last-Click, Markov_DDA, or Shapley_DDA.
- media_source.
- weight (Float): normalized attr weight.

Data Preview:
fold_id | model_name | media_source | weight
fold_1 | Markov_DDA | tiktok | 0.125 |
fold_1 | Shapley_DDA | gads:search | 0.340 |


# 3. `eval_survival` table
**Objective**: evaluate the accuracy of retention-projecting survival models.
Schema:
- fold_id: ROCV fold.
- model_name: 'Baseline_Survival', 'sBG', 'BdW'.
- forecast_period: WEEKLY (mapped calendar week).
- segment: cohort ID (e.g. `2022-W01_SUB_WEEKLY_NA_US`).
- rebill_period_t (Int): specific discrete renewal cycle ($t \geq 1$) being evaluated.
- expected_active_users (Float): cohort start size multiplied by model's predicted Survival Rate $S(t)$.
- actual_active_users (Int): actual users who successfully paid their rebill at period $t$.
- expected_ltv_usd (Float): expected net revenue from the cohort over their entire lifetime.
- actual_ltv_usd (Float): actual net revenue from the cohort over their entire lifetime.
- confidence_weight (Float): 0 for full confidence, negative for fallbacks/boundary-hit.


# 4. `eval_mmm` table
**Objective**: evaluate Macro-level saturation capacity and total incremental revenue prediction.
Schema:
- fold_id: ROCV fold.
- model_name: 'Baseline_MMM_Reg' or 'MMM_Bayesian_Heuristic', 'MMM_BSTS_Heuristic', 'MMM_Bayesian_DDA', 'MMM_BSTS_DDA'.
- forecast_period: WEEKLY.
- segment: macro-region - single row across all subscription types. **Must be UNIQUE per fold.**
- expected_net_revenue_usd (Float): incremental revenue (incl. long-term trend).
- actual_net_revenue_usd (Float): actual net revenue.
- base_sales_intercept (Float): revenue from organic/baseline drivers (no marketing impact, incl. trend).
- mean_saturation_point (Float): derived half-saturation point parameter $K$ from the Hill function, averaged or extracted from the Bayesian posterior.
- confidence_weight (Int): 0 for full confidence, -1 for fallback.
- prior_source: for Bayesian models only - records whether Scenario B used DDA-derived priors (`'shapley_dda'`) or fell back to heuristic priors (`'heuristic_fallback'`). NULL for OLS. This field is critical for ensuring Scenario A and Scenario B rows are genuinely comparable across all regions.

**Important Note**: if a region's Scenario B had no Shapley data and fell back, its `eval_mmm` rows are effectively equivalent to Scenario A and should be flagged accordingly in analysis.


# 5. `mmm_channel_contribs` table
Persistent storage for calculated channel contributions from MMM models.

Schema:
- model_name.
- fold_id: ROCV fold.
- segment: a fold-unique ID that maps to the `eval_mmm.segment`.
- incr_gads_search, incr_gads_youtube, incr_gads_discover, incr_metads_inst, incr_metads_fb, incr_tiktok (Float): actual revenue weights per paid channel, normalized to 1.0.
- actual_contrib_gads_search, actual_contrib_gads_youtube, actual_contrib_gads_discover, actual_contrib_metads_inst, actual_contrib_metads_fb, actual_contrib_tiktok (Float): actual revenue contributions per paid channel.

Data Preview:
model_name | fold_id | segment | incr_gads_search | incr_gads_youtube | incr_gads_discover | incr_metads_inst | incr_metads_fb | incr_tiktok
Baseline_MMM_Reg | fold_1 | 2021_Week-32_Total_EU_SOUTH | 0.125 | 0.250 | 0.125 | 0.125 | 0.250 | 0.125 |


# 6. `survival_model_params` table
Persists fitted parameters for the BdW survival model (and sBG structurally, since sBG is BdW with $c=1$) to enable post-hoc analysis and full distribution tracking.

Schema:
- fold_id: ROCV fold.
- segment: mapping to the cohort ID evaluated (or fallback descriptor).
- alpha (Float): fitted $\alpha$ parameter of the Beta-Geometric distribution.
- beta (Float): fitted $\beta$ parameter.
- c (Float): fitted duration-dependence shape parameter $c$ (for sBG, this is 1.0).


# 7. `survival_monetary_params` table
Persists fitted parameters for the Gamma-Gamma monetary model to enable post-hoc analysis.
Schema:
- fold_id: ROCV fold.
- segment: mapping to the cohort ID evaluated (or fallback descriptor).
- p (Float): fitted $p$ parameter of the Gamma-Gamma distribution.
- q (Float): fitted $q$ parameter.
- gamma (Float): fitted $\gamma$ parameter.
- expected_arpu (Float): expected ARPU.


# 8. `portfolio_budget_simulation` table
Stores the output of the F10 Cross-Domain Portfolio mROAS budget optimization.
Note: details can be found in __docs/comparison/portfolio_optimization.md__.

Schema:
- fold_id: ROCV fold.
- budget_scenario_usd (Float): total constrained budget input for the simulation (e.g., $100k, $250k, $500k).
- allocation_strategy: `'Deterministic'` (Last-Click + OLS) or `'Probabilistic'` (Shapley + BSTS + sBG).
- channel: `tiktok`, `gads:search`, etc.
- allocated_spend_usd (Float): budget distributed to this channel by the simulation algorithm.
- expected_acquisitions (Float): expected new users from this spend.
- expected_ltv_revenue_usd (Float): estimated lifetime value (Gamma-Gamma + Survival) generated by these acquisitions.
- marginal_roas (Float): return on the last dollar spent in this channel at this allocation level.
