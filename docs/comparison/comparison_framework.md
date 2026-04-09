Related doc (visualization strategy and implementation details): **./comparison_visualization.md**.

This document outlines the evaluation framework for the thesis, designed to definitively compare deterministic heuristic strategies (Last-Click, OLS, Exponential decay) against probabilistic data-driven pipelines (Shapley/Markov, BSTS, sBG) across three granular analytical planes.

The core philosophy of this framework: rather than deploying dozens of generic ML metrics, we use **domain-specific programmatic metrics** and **actionable business KPIs** that translate directly into clear visual proofs of the probabilistic approach's superiority.
---

# Execution Methodology: Rolling-Origin Cross-Validation (ROCV).
Given the severe sparsity of data and the relatively short total observation period (11 months: April 2021 to February 2022), a single static Train/Holdout split is vulnerable to seasonality bias (e.g., models overfitting or getting "lucky" during the Q4 holiday spike).
To rigorously stress-test the probabilistic models against baseline heuristics, this framework relies on a **4-Fold Monthly Expanding Window** architecture:
- **Fold 1**: Train (Apr – Aug, 5 months) | Holdout (Sep – Nov, 3 months)
- **Fold 2**: Train (Apr – Sep, 6 months) | Holdout (Oct – Dec, 3 months)
- **Fold 3**: Train (Apr – Oct, 7 months) | Holdout (Nov – Jan, 3 months)
- **Fold 4**: Train (Apr – Nov, 8 months) | Holdout (Dec – Feb, 3 months)

**Why exactly 4 folds and a monthly shift (versus weekly)?**
1. **Holdout Length Constraint (3 Months)**: retention plane requires at least a 3-month holdout to visually demonstrate the divergence (the "crossing point") between naive exponential estimations and heterogeneous sBG curves long-term.
2. **Training Sparsity Window**: MMM regressors need a sufficient cold-start history to establish baseline saturation curves. 5 months is the absolute minimum viable training window to start fitting OLS/BSTS reliably.
3. **Execution Redundancy**: shifting the origin on a weekly basis would create ~16 overlapping folds. Because MCMC and Markov models are computationally expensive, and consecutive weeks are highly autocorrelated, weekly shifting provides diminishing statistical returns.

**Known Design Limitations (must be acknowledged in the thesis):**
1. **Holdout Overlap:** with a 3-month holdout and a 1-month expansion step, consecutive folds share 2 months of holdout data (folds 3 and 4 both contain December 2021 and January 2022). Averaged metrics across all 4 folds are therefore not 4 independent observations - the effective degrees of freedom are lower. **Mitigation:** Report metrics _per fold_ in addition to the aggregate average, so reviewers can see the individual trajectories rather than a misleadingly smooth mean.

2. **Q4 Asymmetric Load:** revenue data contains a structural spike (Jan 2022: $365k vs. Oct/Nov avg: $139k). This spike falls exclusively in the holdout windows of Folds 3 and 4, creating a structurally harder test for the later folds. **Mitigation:** Decompose metric reporting into two performance regimes - **"Stable Period" (Folds 1–2)** and **"Regime Change Period" (Folds 3–4)** - to make the models' calibration under distributional shift a distinct, explicit evaluation axis. A probabilistic model that maintains calibrated credible intervals during the Q4 spike is a stronger thesis argument than one that simply achieves a lower average WAPE.
---

# 1. Cross-Domain Synthesis.
This evaluates the holistic pipeline and demonstrates how combining the planes leads to superior decision-making. No unified mathematical error metric exists across domains, so we focus on synthesized outputs.

## Business KPIs (Out-of-Sample).
- **Portfolio Marginal ROAS (mROAS)**
  - **What it measures**: predicted aggregate enterprise revenue given a specific media budget allocation.
  - **How it compares**: simulates budget allocation based on Deterministic strategy vs Probabilistic strategy.
  - **Visual Proof**: **Figure 10** comparing total expected yield under identical budgets.

---
# 2. Micro-Level: Data-Driven Attribution (DDA).
- **Models**: Last-Click (Deterministic baseline) vs. Shapley Value & Markov Chains (Probabilistic).
- **Challenge**: No strict counterfactual truth exists for individual touchpoint attribution.

## Programmatic Metrics (In-Sample).
- **Bootstrap Stability (Coefficient of Variation)**
  - **What it measures**: how brittle the attribution weights are when the underlying path distribution shifts slightly.
  - **How it compares**: Bootstrap resampling (100-200 iterations per fold). Measure the variance (CV) of channel credit shares. Last-click will remain falsely rigid.
  - **Visual Proof**: **Figure 1** highlighting Top-Funnel variance.


## Out-of-Sample: WAPE - Validation of Non-Forecastability.
**Hypothesis**: attribution models are retrospective instruments and are not designed to forecast. All three DDA models (Last-Click, Markov, Shapley) should converge to a similar out-of-sample error magnitude when static training-period CAC assumptions are projected into the holdout period.

**Purpose of this measurement**: WAPE scores do NOT rank the attribution models against each other — they all perform equivalently poorly at forecasting for the same structural reason (CAC non-stationarity). The purpose is to **confirm the theoretical boundary** of attribution modeling: it is a retrospective instrument that cannot substitute for forward-looking MMM. This section is a designed hypothesis test, not a performance comparison. The high WAPE across all three models is the finding, not a failure.

The WAPE formulas are completely standard:
$$\text{WAPE}_{\text{conv}} = \frac{\sum_w |\text{actual}_w - \text{expected}_w|}{\sum_w \text{actual}_w}$$

$$\text{WAPE}_{\text{cac}} = \frac{\sum_w |\text{actual_cac}_w - \text{model_cac}|}{\sum_w \text{actual_cac}_w}$$

Both are computable, per model per fold.

**Reporting note**: Present the three models' WAPE values side-by-side with a single annotation: _"All DDA models converge to a similar error magnitude under holdout projection, confirming that CAC stationarity does not hold across the 3-month horizon. Attribution credit distribution (Markov vs. Shapley vs. Last-Click) is immaterial to out-of-sample revenue projection; DDA's analytical value lies exclusively in retrospective channel diagnostics and cross-plane MMM prior calibration."_

**Important Note:** must be presented as two separate figures, which are currently excluded from the Figure manifest:
- Conversions plot: 3 time-varying model lines (driven by three distinct model_aggregate_cacs) vs actual: models diverge in predicted volume based on their weight beliefs, but all miss the actual dynamics badly → proves non-stationarity.
- CAC plot: 3 horizontal lines vs actual CAC trajectory: visually the most powerful illustration of a frozen, non-adaptive model. The punchline is that all three horizontal lines cluster near each other (proving convergence), while actual CAC fluctuates. You caption it: "Attribution-derived CAC forecasts collapse to a model-invariant mean regardless of weight methodology, confirming DDA is a retrospective instrument."


## Cross-Plane Validation (Out-of-Sample Surrogate).
- **Rank Concordance with MMM (Spearman $\rho$)** <- **DEMOTED: present as an in-text table, not a standalone figure.**
  - **Rationale for demotion:** with 6 channels, Spearman `ρ` can only take a small number of discrete values. DDA weights barely drift across folds (max observed range $\approx$ 0.003 per channel), so the `ρ`-per-fold trajectory has negligible within-series variance and would appear as a near-flat line — not a compelling standalone figure. The point is still valid and should be reported, but as a compact table.

  **How to compute and present in the thesis text:**
  1. **Extract DDA channel ranks:** for each fold, query `dda_weights` for `model_name IN ('Baseline_LastClick', 'Shapley_DDA')`. Rank the 6 paid channels by `weight` descending (rank 1 = highest weight) for each model separately.
  2. **Extract MMM iROAS ranks:** for each fold, query `mmm_channel_contribs` for the best-performing Bayesian model (`MMM_BSTS_DDA` preferred, fall back to `MMM_Bayesian_DDA`). The `incr_{channel}` columns contain normalized incremental revenue weights. Rank the 6 channels descending.
  3. **Compute Spearman ρ:** for each fold and each DDA model, compute `scipy.stats.spearmanr(dda_ranks, mmm_iroas_ranks).statistic`. This yields 2 ρ values per fold (Last-Click vs MMM, Shapley vs MMM).
  4. **Present as a table** in the thesis:

  | Fold     | Last-Click ρ vs MMM | Shapley_DDA ρ vs MMM |
  | -------- | ------------------- | -------------------- |
  | fold_1   | …                   | …                    |
  | fold_2   | …                   | …                    |
  | fold_3   | …                   | …                    |
  | fold_4   | …                   | …                    |
  | **Mean** | …                   | …                    |

  **Annotation to include:** _"Shapley consistently achieves higher rank concordance with MMM's incremental ROAS estimates, confirming that coalition-based attribution better captures systemic channel efficiency than Last-Click's terminal-touch credit assignment. Note: with N=6 channels, Spearman ρ has limited resolution; these values serve as directional confirmation rather than a precise measurement."_

---
# 3. Macro-Level: Media-Mix Modeling (MMM).
## Programmatic Metrics:
- **In-Sample: WAIC**.
  - Used strictly _within_ the probabilistic models to justify structural specifications (BSTS vs standard Bayesian Regression).
  - Must be reported during the Bayesian model execution.
- **Out-of-Sample: WAPE**.
  - Provides a scale-free programmatic error comparison on a 3-month rolling holdout. Reported per fold, not just globally, to visualize stability. Shows both the mean difference and the fold-level variance - critical with only 4 folds. If BSTS is better on average but worse on one specific fold, you want that visible, not averaged away.
  - Presented as part of **Figure 4**.

## Probabilistic Superiority Metrics (Out-of-Sample).
- **Interval Coverage Rate**
  - **What it measures**: Percentage of held-out actuals that fall within the model's credible interval.
  - **How it compares**: Deterministic OLS offers no native probabilistic bounds. A well-calibrated Bayesian model will track close to the nominal %. Y-axis = nominal coverage level (50%, 80%, 90%, 95%), x-axis = actual empirical coverage across holdout folds. BSTS traces a line near the diagonal. OLS doesn't exist on this chart - annotate it with a note saying the deterministic model produces no intervals. The absence is the argument.
  - **Visual Proof**: **Figure 5** tracking the diagonal (empirical vs nominal rate).

- **Posterior Predictive Time-Series (Shaded Confidence Interval)**
  - **What to plot**: X-axis = timeline (e.g., weeks); Y-axis = target variable. Plot observed actuals, mean posterior predictive, and 95% credible band shaded region, split by a vertical line denoting the train/holdout boundary.
  - **The story**: Translates abstract Bayesian uncertainty into a tangible business visual. Where deterministic OLS produces a rigid point estimate that fails silently during regime shifts (e.g., Q4), the Bayesian credible band dynamically encompasses the actuals, proving the model successfully quantifies the increased volatility.
  - **Build notes**: Use `ax.fill_between` for the 95% band. Ensure a vertical line splits Train vs Holdout. Contrast against the deterministic point forecast directly if possible.
  - **Data requirement**: Observed actuals (`mmm_timeseries`) + posterior predictive traces.
  - **Implementation note**: Best showcased on Fold 4 to visually demonstrate uncertainty calibration explicitly during the Q4 regime shift.
  - **Visual Proof**: **Figure 6** displaying the time-series with the shaded posterior predictive intervals.

## Business KPIs.
- **Systematic Bias (WBIAS) per fold**.
  - **What it measures**: does the model systematically over or under-predict during trend shifts?
  - **How it compares**: OLS/STL will show systematic directional bias (positive in growth, negative post-peak) while BSTS should hover near zero across folds. The signed nature makes it a different story from WAPE - it shows what kind of error each model makes, not just how much. Two panels, same x-axis, tells a complete story.
  - Presented as part of **Figure 4**.

---
# 4. Customer Base: Survival Analysis (Retention).

## Programmatic Metrics (Out-of-Sample)
- **Holdout RMSE by Horizon Age (early vs late)**.
  - **What it measures**: Prediction error trajectory.
  - **How it compares**: X-axis = cohort age (months since acquisition), y-axis = RMSE. Two lines. They'll cross - exponential fits early periods comparably, then diverges at longer horizons as the survivor-pool heterogeneity accumulates. That crossing point is your thesis in one plot.

## Business KPIs (Out-of-Sample).
- **LTV Extrapolation Bias (D90 & D180)**.
  - **What it measures**: error in cohort cumulative revenue over 3-6 months.
  - **How it compares**: exponential curve systematically overestimates late-stage retention, directly inflating the predicted LTV.
  - **Visual Proof**: **Figure 7** (Empirical vs. Exponential vs. BdW) showing divergence at the long tail.

---
# A few things worth flagging explicitly about this set:
- WAPE and WBIAS share the same fold structure and should be presented as a two-panel figure - same x-axis, stacked vertically. That way the reader sees both how much error OLS makes and which direction it consistently errs in, without two separate narrative blocks.
- The calibration plot is the second most powerful figure in the thesis for stated goal. The argument it makes is categorical, not scalar: the probabilistic model can be evaluated on calibration; the deterministic model cannot exist on the same axes. Frame it that way in your caption - not "BSTS performs better" but "OLS STL produces no uncertainty estimates and therefore cannot be evaluated on this dimension." That's a stronger claim than any numerical comparison.
- The RMSE-by-cohort-age figure will likely be your most visually compelling result if the data behaves as expected - two lines that start close together and diverge as cohort age increases. The crossing point, if it exists, becomes a natural focal point for your discussion of why the exponential curve's homogeneity assumption breaks down at longer horizons.