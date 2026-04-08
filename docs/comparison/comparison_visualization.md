Related docs:
- Comparison framework (defines input metrics): **./comparison_framework.md**
- Current challenges and limitations: **./visualization_challenges.md**

# F1 — Attribution Shift (dodged bar + error bars).
What to plot: X-axis = channels (paid + organic). Two bar groups per channel: Last Click credit share vs Shapley credit share, averaged across bootstrap iterations. Error bars on the Shapley bars only — ±1 SD from the 100–200 bootstrap resamples. Last Click bars have no error bars by construction.

The story the shape tells: Last Click bars will be tall on the last-touch channel and zero or near-zero on upper-funnel channels. Shapley bars redistribute credit up the funnel, and their error bars show that this redistribution is statistically stable — not noise.

Build notes: Use matplotlib with ax.bar at offset positions. Add ax.errorbar over the Shapley bars only. Annotate the zero-height Last Click bars for upper-funnel channels with a small text label ("0% credit assigned") to make the absence explicit. Color: one muted tone for Last Click, one saturated tone for Shapley.

Data requirement: `dda_weights` table + bootstrap per-iteration weights.
Implementation note: bootstrap resampling should be computed post-hoc from `attribution_paths` without rerunning the pipeline in the visualization script.


# F2 — Markov Transition Heatmap.
What to plot: Square matrix where rows = origin state (channel or "Start"), columns = destination state (channel, "Conversion" or "Null"). Cell values = transition probability. Annotate each cell with the probability. Add a title annotation: total journeys analyzed and base conversion rate.

The story: High-probability paths that never end in a last-touch conversion (e.g., TikTok → Meta → Conversion where TikTok gets zero Last Click credit) become visually obvious. The heatmap is the qualitative proof of Last-Click's blindness to path structure. It supports the framing that Markov discovers sequences while Shapley informs priors.

Build notes: seaborn.heatmap with annot=True, fmt=".2f", cmap="YlOrRd". Mask the diagonal if self-loops aren't meaningful in your Markov chain. Sort rows/columns by total outbound flow so the most active channels cluster top-left.

Data requirement: transition probability matrix from `attribution_paths`.

Implementation note: transition matrix should be recomputed from `attribution_paths` at visualization time.


# F3 — MCMC Posterior KDE vs OLS Point Estimates.
What to plot: One subplot per major model coefficient (e.g., one per marketing channel's spend coefficient). In each subplot: KDE of the MCMC posterior samples (a smooth bell curve). Overlaid as a vertical dashed line: the OLS point estimate for the same coefficient.
The story: The OLS estimate will often sit off-center relative to the posterior, and the width of the posterior KDE shows the uncertainty that OLS silently suppresses. For any coefficient where the posterior's 95% credible interval crosses zero, the OLS point estimate is making a confident claim about a parameter that is statistically ambiguous.

Build notes: `scipy.stats.gaussian_kde` on the MCMC traces, plotted with `ax.fill_between` for the density. Add `ax.axvline` for the OLS estimate with a label in the legend. Shade the 95% HDI (highest density interval) region. Arrange as a grid of small multiples - one per coefficient — using subplots with shared y-axis scaling.

Data requirement: Full MCMC traces per coefficient + OLS coefficients provided in corresponding `posterior_trace_{model_name}_fold_4` and `ols_channel_coeffs_fold_4` files present locally.

Implementation note: one fold (#4) is sufficient for this figure. Ensure the KDE is computed over the unthinned chain for the 6 channel beta coefficients.


# F4 — WAPE + WBIAS dual-panel (the MMM out-of-sample proof).
What to plot: 
- Panel A (top): X-axis = fold index. Y-axis = WAPE (0 to ~1). Three lines: OLS/STL, Bayesian MMM and BSTS. Add individual fold points as markers so the reader can see that the mean difference isn't driven by one outlier fold.
- Panel B (bottom): Same X-axis. Y-axis = WBIAS, centered at zero with a horizontal reference line. Three bar series (or lines), one per model. OLS bars will lean positive in growth folds and negative in contraction folds - the sign-switching pattern is the key visual. Bayesian models' bars should hover near zero.

Build notes: Use plt.subplots(2, 1, sharex=True) so the fold axis is literally shared. This is the most important implementation detail — a reader should be able to draw a vertical line at fold 3 and read both error and bias simultaneously. Use the same color scheme across both panels for each model. Add a grey shaded band at ±5% WBIAS to visually define "acceptable" bias.

Data requirement: `eval_mmm` with actuals populated, aggregated per fold.


# F5 — Calibration Coverage Chart.
What to plot: X-axis = nominal coverage level (50%, 60%, 70%, 80%, 90%, 95%). Y-axis = empirical coverage rate (percentage of held-out actuals that fell within the corresponding credible interval). One line for BSTS. One diagonal reference line (the perfect calibration ideal). OLS is not plotted — annotate its absence explicitly in the figure caption.
The story: This is your categorical proof. A well-calibrated BSTS traces near the diagonal. OLS cannot exist on these axes because it produces no intervals. The annotation "OLS/STL: no probabilistic output - cannot be evaluated on this criterion" placed inside the plot area is more persuasive than any bar comparison.
Build notes: Compute empirical coverage for each nominal level by checking, for each holdout observation, whether it falls within the alpha/2 and 1-alpha/2 quantiles of the posterior predictive distribution. Average across folds. Plot with ax.plot for BSTS, ax.plot([0,1],[0,1], '--', color='gray') for the diagonal. Keep the aspect ratio square so the diagonal is literally at 45 degrees.

Data requirement: Full posterior predictive distribution for holdout observations provided in `posterior_predictive_{model_name}_{fold_id}.npy` PPC files.
Implementation note: should be compared across all 4 folds.


# F6 — Cohort Survival Decay Curves.
What to plot: X-axis = rebill number (period since acquisition, 1 through 13). Y-axis = retention rate (log-scaled, range approximately 0.1 to 0.8 as per your legacy spec). Three series: empirical observed retention, exponential curve fit, BdW fit. Optionally show multiple cohorts as separate small multiples if they tell materially different stories.

The story: all three lines will be close at early periods (rebill 1–3). At rebill 5+, the exponential curve will diverge upward from empirical, overestimating retention. The BdW curve will track the empirical tail much more closely. The log y-axis is important - it makes the divergence at the long tail legible that would be visually compressed on a linear scale.

Build notes: ax.set_yscale('log'). Plot empirical as ax.step or ax.scatter with markers to make its discreteness visible. Plot both model fits as smooth lines. Shade the region between the exponential fit and empirical as a light fill — this shaded area is the LTV overestimation region, which connects visually to F9.

Data requirement: `cohorts_retention` (empirical) + model S(t) predictions (`eval_survival`) tables.


# F7 — RMSE Divergence (retention out-of-sample proof).
What to plot: X-axis = cohort age in months (1 through max horizon). Y-axis = RMSE. Three lines: exponential and BdW. The lines should start close and diverge - the crossing/divergence point is the visual thesis of this plot.

Data requirement: Per-cohort model predictions at multiple horizons.


# F8 — LTV Extrapolation Bias (small multiples per cohort).
What to plot: one panel per acquisition cohort (e.g., cohorts acquired in months 1–8 of your dataset). In each panel: observed cumulative revenue curve, exponential model extrapolation, BdW model extrapolation. Mark the train/holdout boundary with a vertical dashed line. The extrapolation zone is to the right of that line.

The story: in the extrapolation zone, the exponential line will sit consistently above the observed curve (overestimates LTV). The BdW model will track the observed curve. The shaded gap between exponential and observed is the business cost of using the wrong model — quantify it as a percentage in the subplot title (e.g., "+23% LTV overestimation at D180").

Build notes: `plt.subplots` grid, shared y-axis across cohorts. Use `ax.axvline` at the train/holdout boundary. `ax.fill_between` to shade the overestimation gap. Label each panel with the cohort acquisition month and the overestimation percentage at D180.

Data requirement: cumulative observed revenue per cohort + model-projected LTV curves (derived from `eval_survival.expected_ltv_usd`).


# F9 — Portfolio mROAS (Business "Climax").
What to plot: grouped bar chart. X-axis = two groups: "Deterministic strategy" (Last Click CAC + OLS MMM + Exponential LTV) and "Probabilistic strategy" (Shapley CAC + BSTS MMM + BdW LTV). Y-axis = simulated portfolio marginal ROAS under identical total budget. Within each group, optionally show sub-bars for each component's contribution.

The story: this is the synthesis figure and should be last in the results chapter. Under the same budget, the probabilistic strategy's allocation (informed by more accurate channel weights, better trend modeling, and corrected LTV) should yield a higher predicted ROAS. The magnitude of the difference is your thesis's business case quantified in a single number.

Build notes: the simulation logic matters more than the chart itself: run an optimization (even a simple one — allocate budget proportionally to channel mROAS as estimated by each strategy) and compute predicted revenue under each allocation using your MMM's response curves. Document the simulation assumptions transparently — a thesis committee will ask about them. The chart is just `ax.bar` with two groups and a clear legend.

Data requirement: channel-level response curves from MMM (`mmm_channel_contribs` table) + CAC from DDA (`dda_weights` table) + LTV from survival (`survival_monetary_params` and `survival_model_params` tables).

Implementation note: algorithm is explicitly specified in __./portfolio_optimization.md__. 