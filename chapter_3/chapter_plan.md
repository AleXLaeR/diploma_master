# Chapter 3 Skeleton — Practical Experiment, Architecture, and Results

## Summary
- Build Chapter 3 as a proof narrative, not as a code diary: the chapter should show that on noisy real subscription-business data, a probabilistic multi-plane analytics pipeline produces more reliable and decision-useful results than the deterministic heuristics currently used by the business subject.
- Keep the logic linear: research claim → comparison design → business scenario → initial data and its flaws → data architecture and feature engineering → output contracts → comparison framework and hypotheses → DAG architecture → model specifications → figure-by-figure results → further research.
- Treat `users_attribution_imputed` and `touchpoints_log` as initial business-provided analytical inputs in the chapter narrative, per your instruction, even though the repo materializes them operationally.

## Chapter Skeleton
### 3.1. Research Goal and What Exactly the Experiment Proves
- Open with 3 paragraphs.
- Paragraph 1: define the business problem. The company currently relies on deterministic heuristics for acquisition attribution, media planning, and retention projection; the experiment tests whether these heuristics fail under real business uncertainty.
- Paragraph 2: define the experimental object. One real B2C subscription business is observed from 2021-04-01 to 2022-02-28; the pipeline covers three analytical planes: micro-level attribution, macro-level MMM, and customer-base retention/LTV.
- Paragraph 3: define the proof claim precisely. The thesis is not “probabilistic models always predict larger profits”; it is that probabilistic models produce more stable attribution structure, lower out-of-sample forecast error, materially lower long-tail retention/LTV distortion, and calibrated uncertainty, while deterministic models produce brittle point estimates and false confidence.

### 3.2. Comparison Design and Why These Models Were Chosen
- Explain that comparison is domain-scoped by analytical plane and split into in-sample diagnostics and out-of-sample validation; do not list concrete metrics yet.
- State the 4-fold monthly expanding-window ROCV design and the two reporting regimes: stable period (`fold_1–fold_2`) and regime-change period (`fold_3–fold_4`).
- DDA comparison: `Baseline_LastClick` vs `Markov_DDA` vs `Shapley_DDA`.
- DDA exclusion logic: do not include extra heuristic rules like first-click/linear/time-decay because they are the same deterministic family; do not include causal/uplift/ML attribution because there is no randomized counterfactual design and black-box ML is out of scope.
- MMM comparison: `Baseline_MMM_Reg` vs `MMM_Bayesian_Heuristic` / `MMM_Bayesian_DDA` / `MMM_BSTS_Heuristic` / `MMM_BSTS_DDA`.
- MMM exclusion logic: do not use Prophet/ARIMA as main competitors because the task is not generic forecasting but marketing-response estimation with carryover, saturation, exogenous controls, and interpretable priors; do not use Robyn or other black-box frameworks because the thesis explicitly restricts itself to transparent statistical models.
- Survival comparison: `Baseline_Survival` vs `sBG` vs `BdW`.
- Survival exclusion logic: do not use Cox PH because the setting is discrete contractual rebilling, cohort-aggregated, sparse, and aimed at renewal-curve/LTV projection rather than hazard-ratio interpretation; do not use Pareto/NBD-like BTYD models because they assume non-contractual continuous-time behavior and are invalid for subscriptions.
- Add one paragraph clarifying that the final results section should focus on the strongest contrasts: Last-Click vs Shapley/Markov, OLS vs BSTS_DDA, and Baseline vs BdW; `sBG` and plain Bayesian MMM stay in the model-spec and comparison tables as intermediate benchmarks.

### 3.3. Business Subject and Experimental Scenario
- Introduce the business subject as a real subscription-based digital educational product with `SUB_WEEKLY`, `SUB_MONTHLY`, and `SUB_3_MONTH` offers plus discounted trial periods.
- Explain why this scenario is analytically demanding: acquisition depends on paid traffic, revenue is seasonal and shock-prone, and business value depends on downstream retention rather than only on initial conversion.
- State the observed scale early: about 240k transactions, 114k subscriptions, 96k attributed users, 68k spend rows, 225+ countries mapped into macro-regions.

### 3.4. Initial Data and Why It Was Deliberately Left Imperfect
- Describe the initial inputs in separate short subsections: `purchases`, `users_attribution`, `insights`, `countries`, `users_attribution_imputed`, `touchpoints_log`.
- For `purchases`, highlight the real business ledger nature, trials, refunds, rebills, the April–September monthly-only phase, and the observed monthly revenue instability including the January 2022 spike.
- For `users_attribution`, emphasize that it is last-click only, heavily concentrated in Facebook (~85%), geographically sparse, and affected by the ATT-related orphaned-transaction gap.
- For `insights`, note sparse low-spend countries but sufficient global density for MMM.
- For `countries`, explain that it is the dimensional mapping needed to roll country data into macro-regions.
- For `users_attribution_imputed`, present it as an as-is attribution source used in the experiment to restore analytical coverage over sparse country/channel cells without rewriting business reality.
- For `touchpoints_log`, present it as the as-is multi-touch interaction log used by attribution and contribution reconstruction.
- Explicitly cite the observations from [initial_data_discoveries.md](/C:/Users/Gigabyte/Desktop/diploma/docs/data/initial/initial_data_discoveries.md): sparse geo coverage, orphaned transactions, unrealistic raw KPI picture, revenue regime shift, and limited useful rebill depth.
- Close this section with the methodological point: the data was mostly left unchanged to show that deterministic heuristics cannot absorb real business uncertainty, while probabilistic models can at least express and manage it honestly.

### 3.5. Data Architecture, Feature Engineering, and Output Contracts
- Explain the data flow as: initial inputs → intermediate analytical views → model-specific marts → fold-tagged evaluation tables.
- Briefly describe engineered analytical tables: `insights_channel_spend`, `channel_cpc_weights`, `rocv_folds`, `refund_rates`, `attribution_paths`, `mmm_timeseries`, `cohorts_retention`.
- DDA feature engineering: journeys, absorbing success/null states, coalition building, paid-channel normalization, EWMA-smoothed CAC translation.
- MMM feature engineering: weekly regional aggregation, min-max spend normalization, AdStock, Hill saturation, `revenue_anomaly_score`, Fourier terms, `is_sep_nov_trough`, `is_structural_peak`, and optional DDA-informed priors.
- Survival feature engineering: cohort matrices, trial exclusion, rebill-period mapping, fallback pooling, and Gamma-Gamma monetary layer.
- End this section with the first four output-contract tables only:
- `eval_dda`: `expected_conversions` vs `actual_conversions`; `expected_cac_usd` vs `actual_cac_usd`; `confidence_weight` marks fallback confidence.
- `dda_weights`: no `actual_*`/`expected_*` pair; this table stores fold-scoped learned attribution weights and should be described as a model-state contract, not an evaluation table.
- `eval_survival`: `expected_active_users` vs `actual_active_users`; `expected_ltv_usd` vs `actual_ltv_usd`; `confidence_weight` marks fallback/boundary cases.
- `eval_mmm`: `expected_net_revenue_usd` vs `actual_net_revenue_usd`; `base_sales_intercept`, `mean_saturation_point`, `confidence_weight`, `prior_source`.
- Add one anti-leakage paragraph: all models write only `expected_*`; `actual_*` is filled later by the consolidation layer.

### 3.6. DAG Architecture
- Give this section its own diagram/subsection around the Airflow orchestration.
- Phase 1: build `rocv_folds`, `users_attribution_imputed`, `touchpoints_log`, intermediate tables, and model-specific marts.
- Master ROCV DAG: launch four fold-specific runs in parallel.
- Phase 2 DDA DAG: initialize eval tables, run Last-Click/Markov/Shapley, then trigger survival and MMM.
- Phase 2 survival DAG: run baseline, `sBG`, `BdW`, and persist model/monetary parameters.
- Phase 2 MMM DAG: run OLS, Bayesian regression, BSTS, and persist posterior artifacts and channel contributions.
- Phase 3 evaluation DAG: consolidate `actual_*` via SQL, then compute metrics and render thesis figures.
- Make the architectural argument explicit: BigQuery owns heavy joins/aggregations; Python/Docker owns model fitting; fold integrity, anti-leakage, idempotency, and fallback traceability are built into the DAG design.

### 3.7. Formal Comparison Framework, Hypotheses, and Objective Proof
- Introduce concrete metrics here.
- DDA in-sample hypothesis: probabilistic attribution should redistribute value from lower-funnel closure channels toward upper-funnel initiators while remaining statistically stable; expected signal is small Shapley bootstrap SD (<0.02 for reported channels).
- DDA out-of-sample hypothesis: all three attribution models should converge to similarly poor forecasting quality because CAC is non-stationary; expected aggregate WAPE range is roughly 0.46–0.55, worsening in `fold_3–fold_4`. Objective proof: convergence itself proves DDA is retrospective, not forecast-capable.
- MMM hypothesis: deterministic OLS should show high WAPE and severe sign-switching bias, while probabilistic MMM should reduce WAPE into roughly 0.12–0.25 and keep WBIAS near zero; DDA-informed BSTS is expected to be best. Objective proof: lower error plus near-zero bias plus calibrated interval coverage.
- Calibration hypothesis: empirical coverage should follow the nominal diagonal closely; expected aggregate pattern is about 48% at nominal 50% and 94% at nominal 95%. Objective proof: the Bayesian model can be evaluated on calibration; OLS cannot.
- Survival hypothesis: deterministic curve fitting should increasingly overestimate retention and LTV in the long tail; expected baseline retention MPE is roughly +20% to +82%, while BdW stays near +1% to +7%; expected RMSE gap is about 4x–7x in BdW’s favor; expected D90/D180 LTV inflation is about +21% to +41% for baseline and about -1% to +6% for BdW.
- State the chapter-level proof rule explicitly: the experiment is considered successful only if each plane confirms its intended role: DDA as retrospective diagnostic memory, MMM as forward-looking calibrated revenue model, and survival as realistic long-tail retention/LTV engine.

### 3.8. Model Specifications
- DDA subsection: Last-Click as deterministic baseline; second-order Markov with removal effect and Laplacian smoothing; Shapley with coalition approximation; exclusion of `organic` and `legacy_untracked` from paid ROI outputs; EWMA-based holdout projection used only to prove non-forecastability.
- MMM subsection: OLS with STL residualization and normalized spend; Bayesian regression with AdStock + Hill + exogenous priors; BSTS with local linear trend for adaptive baseline tracking; heuristic vs Shapley-informed prior scenarios; region-to-global fallback.
- Survival subsection: deterministic curve-fit baseline; `sBG` as discrete contractual heterogeneity model; `BdW` as duration-dependent extension; Gamma-Gamma for monetary value; confidence-weight hierarchy and the `SUB_3_MONTH` caveat.
- End by stating which model won in each plane and why: Shapley/Markov reveal attribution structure better than Last-Click, `MMM_BSTS_DDA` is the strongest macro forecaster, and `BdW` is the strongest retention/LTV model.

### 3.9. Visualizations and Figure-by-Figure Results
- F1: attribution shift. Show Last-Click concentration in Search/Facebook (0.42/0.31) versus Shapley redistribution toward YouTube/Instagram/TikTok/Discover; interpret this as deterministic greediness and stable cooperative reallocation.
- F2: Markov transition heatmap and removal effects. Emphasize that Search closes most often, but TikTok and other upper-funnel channels initiate and feed the path network; removal effects prove ecosystem dependence.
- DDA out-of-sample bridge figure: report WAPE convergence around 0.55 / 0.53 / 0.54 and flat CAC lines against volatile actual CAC; use this as the formal proof that attribution is retrospective.
- F3: posterior density vs OLS. Show OLS over-concentration on Facebook (60.3%) and near-zero valuation of some video channels versus posterior means spread more evenly with credible intervals.
- F4: MMM fold metrics. Contrast OLS WAPE 0.52–0.72 and WBIAS from +0.38 to -0.55 with `MMM_BSTS_DDA` WAPE 0.12–0.18 and WBIAS about ±0.02.
- F5: calibration coverage. State the near-diagonal pattern from 50%→48% up to 95%→94% and make the categorical argument that OLS cannot participate in this evaluation axis.
- F6: posterior predictive time series in Q4. Show rigid regression decline versus adaptive Bayesian interval widening during the holiday shock.
- F7.1: retention-rate curves. Use the three highlighted LATAM cohorts and report baseline vs BdW MPE: `+19.5% vs +0.7%`, `+34.3% vs +7.0%`, `+81.7% vs +2.1%`.
- F7.2: survival-decay curves. Explain the same divergence in active-user space, emphasizing the shaded overestimation zone of the deterministic model.
- F8: RMSE by rebill horizon. Report that the baseline is already 3.7x worse at `t=1` and roughly 5x–7x worse through later rebills.
- F9: LTV extrapolation bias. Report D90/D180 inflation by region, with deterministic inflation around `+21% to +41%` and BdW around `-1% to +6%`; interpret this as direct protection from CAC/LTV mispricing.

### 3.10. Further Research
- Hierarchical or partial-pooling MMM to replace global fallback disaggregation for sparse regions.
- Fully Bayesian survival modeling with posterior intervals, not only point-estimated MLE curves.
- Explicit measurement-error modeling for ATT-era orphaned attribution and touchpoint uncertainty.
- Longer observation window to reduce fold overlap and to test post-peak generalization.
- More exogenous drivers in MMM: pricing actions, promotions, product changes, macroeconomic shocks, competitor pressure.
- Cross-plane optimization with full uncertainty propagation from DDA → MMM → survival rather than sequential point translation.
- Controlled geo-experiments or incrementality studies to validate attribution/MMM priors externally.

## Validation Checklist
- Every analytical plane has four elements in the chapter: business role, chosen models and exclusions, data inputs/features, and objective proof criteria.
- The data section explicitly cites the imperfections from [initial_data_discoveries.md](/C:/Users/Gigabyte/Desktop/diploma/docs/data/initial/initial_data_discoveries.md).
- The output-contract subsection covers exactly `eval_dda`, `dda_weights`, `eval_survival`, and `eval_mmm`, with `actual_*` vs `expected_*` semantics explained.
- The results section covers all figures F1–F9 plus the DDA out-of-sample bridge.
- The chapter states the main limitations: fold overlap, Q4 asymmetry, sparse regions/cohorts, and the fact that some inputs are imperfect by design.

## Assumptions and Defaults
- Section numbering above assumes a standard `3.1–3.10` chapter layout.
- The plan is written in English, but the structure is language-neutral and can be translated directly.
- In the narrative, `users_attribution_imputed` and `touchpoints_log` are treated as initial analytical inputs, as requested.
- Results emphasis is asymmetric by design: the chapter should fully document all tested models, but the discussion should center on the final winning contrasts that support the thesis most clearly.
