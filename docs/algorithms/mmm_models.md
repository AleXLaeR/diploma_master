# Context & Objective.
This module defines the contract for the implementation of **macro-analytical** plane of the experiment. The objective is to forecast different financial metrics (e.g., base sales, saturation point) based on aggregated marketing spend and exogenous factors, proving that probabilistic models with memory (AdStock), saturation (Hill) and integrated knowledge (Bayesian framework) out-predict simplistic linear assumptions.

Implementation constraints:
- Baseline model must be built using `statsmodels` module.
- Bayesian models must be built strictly using `PyMC`.


# 1. Input Data & Preparation.
Source Table: `mmm_timeseries`. Denormalized and non-stationary, integrating marketing spend, baseline revenue, and fixed exogenous factor values.
**ROCV Strategy**: Model fitting and prediction must happen inside an expanding-window loop spanning Folds 1 through 4. For each fold, only time-series rows matching the fold's designated training horizon are exposed to the algorithms.


# 2. Feature Engineering.

## 2.1. Exogenous Factors (control variables).

To isolate the true incremental effect of marketing, the algorithms must control for systemic variance. All models must include the exogenous variables ($X$), which are materialized as columns in the `mmm_timeseries` mart.

**Prior coefficient directions for Bayesian models:**

The 5 exogenous covariates carry domain-specific directional expectations that MUST be encoded as informative priors for Bayesian models (§3.2 and §3.3). OLS receives no such constraints.

| Variable                | Prior Distribution          | Rationale                                                                                                                                                                                                                                                                                            |
| ----------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `revenue_anomaly_score` | Normal(mu=0, sigma=0.5)     | The z-score is already centered at 0; positive scores indicate above-trend revenue (should correlate positively with Y), negative scores indicate below-trend. Keeping mu=0 lets the data decide direction while the tight sigma prevents the score from dominating the likelihood.                  |
| `fourier_cos_q1`        | Normal(mu=0, sigma=1.0)     | Fourier terms are symmetric by construction. The prior is deliberately uninformative — the harmonic's phase and amplitude should be learned from data without directional constraint.                                                                                                                |
| `fourier_sin_q1`        | Normal(mu=0, sigma=1.0)     | Same as `fourier_cos_q1`.                                                                                                                                                                                                                                                                            |
| `is_sep_nov_trough`     | Normal(mu=-0.5, sigma=0.75) | Domain expectation: the Sep-Nov period is a structural revenue trough for the business. The negative prior mean encodes downward revenue pressure during this regime. Sigma=0.75 allows the sampler to override if the data contradicts (e.g., if a specific region doesn't experience this trough). |
| `is_structural_peak`    | Normal(mu=0.5, sigma=0.75)  | Domain expectation: structural peaks (driven by Q4 holiday spike) have upward revenue pressure. Positive prior mean. Same sigma logic as above.                                                                                                                                                      |

**OLS treatment:** all exogenous variables are included as raw regressors in the OLS equation. OLS receives no prior information and must infer coefficient directions from data alone.

## 2.2. Non-Linear Marketing Transformations.

Before Bayesian estimation, the raw spend data ($S_{t}$) for each channel must pass through two transformation functions:

1. AdStock (Geometric Decay): models the memory effect of advertising. $$A_{t} = S_{t} + \alpha A_{t-1}$$(Where $\alpha \in [0,1)$ is the decay retention rate). To account the concrete carruover effects based on the holdout dataset observations (@see **docs/data/initial/initial_data_discoveries.md**), this effect can be skewed to boost the models` performance.
2. Hill/Reach (Diminishing Returns): models market saturation. $$H(A_{t}^{norm}) = \frac{1}{1 + (K / A_{t}^{norm})^S}$$(Where $K \in (0,1)$ is the half-saturation point on the normalised scale, and $S > 0$ is the shape parameter controlling the steepness of diminishing returns).

**NOTE: Before applying the Hill function, all AdStocked spend values ($A_t$) MUST be min-max normalized to the [0, 1] range per channel: $A_{t}^{norm} = \frac{A_t - \min(A)}{\max(A) - \min(A)}$. This ensures the Beta(2,2) prior for $K$ (the half-saturation point) operates on a consistent scale across all channels. The Hill function is then applied to $A_{t}^{norm}$.** Afternormalisation, $A_{t}^{norm}$ MUST be clipped to $[\varepsilon, 1.0]$ where $\varepsilon = 10^{-8}$ before applying the Hill function. Clipping prevents numerical overflow without materially affecting estimation (a near-zero spend still produces a near-zero Hill output).


# 3. Mathematical Cores.
**Important Note:** both Bayesian models (3.2 and 3.3, across scenarios A and B) must be evaluated against each other by accuracy via WAIC / LOO-CV metrics.

## 3.1. The Deterministic Baseline (OLS Regression).

This represents the baseline MMM strategy. It assumes zero memory, zero saturation, and linear scaling.
Equation: $Y_t = \text{Trend}_t + \text{Seasonality}_t + \beta_0 + \sum \beta_i S_{i,t} + \sum \gamma_j X_{j,t} + \epsilon_t$.
Implementation: Use `statsmodels.api.OLS`. Raw spend ($S$) is used directly without AdStock or Hill transformations.

**Stationary Pre-Processing (MANDATORY):** OLS cannot handle non-stationary data. The original process fits the OLS to the residual component after STL decomposition, and adds the trend + seasonality components back to predictions to get the final estimates.
**Net Revenue Calculation**: total net revenue is derived from `order_status IN ('approved', 'settled_ok', 'refunded')` to capture the actual structural macro-revenue footprint, including voided transactions.

**Input Spend Normalization (MANDATORY for fair comparison):** to prevent the scale asymmetry that would arise from comparing OLS (raw spend) against Bayesian models (min-max normalized spend after AdStock), the OLS pipeline MUST also min-max normalize each spend column to the [0, 1] range using the training window's min/max _before_ regressing. This affects only the scale of estimated $\beta_i$ coefficients (which are not directly compared across models), not the sign or significance. Out-of-sample spend values are transformed using the corresponding fold's **training window** min/max to avoid future data leakage.

## 3.2. Bayesian Regression (Multivariate Non-Linear).


A fully probabilistic regression that jointly learns the optimal intercept, $\alpha_i$, $K_i$, and $S_i$ parameters for **each** channel $i$ alongside the channel coefficients $\beta_i$. The AdStock and Hill transformations are expressed inside the PyMC model graph so all uncertainty propagates through NUTS sampling.
Equation: $Y_t = \text{Intercept} + \sum_i \beta_i H(A_{i,t}^{\text{norm}}) + \sum_j \gamma_j X_{j,t} + \epsilon_t$.
Implementation: Use `PyMC` and NUTS sampler.

Min-max normalisation per channel (using per-fold training window stats only) is applied after AdStock and before the Hill function.

**Important Note:** AdStock is **pre-computed** at the prior-mean $\alpha$ for each channel (Python-level recurrence before model construction, **not** a `pytensor.scan` inside the graph) to avoid the underfitting problem. With $T \le 35$ training observations and 6 channels, jointly sampling $\{\alpha_i, K_i, S_i, \beta_i\}$ per channel creates a near-underidentified system (34+ free parameters, 35 observations); fixing $\alpha$ at its prior mean is a recognised methodological constraint for short-panel MMM (Chan & Perry 2017; Berman & van den Bulte 2022). The residual uncertainty from imprecise $\alpha$ is absorbed into the $\beta$ posteriors - the Intercept baseline is estimated entirely independently of $\alpha$ and is not affected by this choice.

## 3.3. Bayesian Structural Time Series (BSTS).

A state-space model that decomposes observed revenue into explicit latent structural components (trend, seasonality) plus an MMM regression layer. Unlike OLS (which requires extracting non-stationarity) and simple Bayesian regression (which is enabled temporal structure explicitly through AdStock), this model implicitly tracks the underlying baseline trend through MCMC geometry.

**Observation (space) Equation:**
$$Y_t = \mu_t + \sum_i \beta_i H(\text{AdStock}(S_{i,t})) + \sum_j \gamma_j X_{j,t} + \varepsilon_t, \quad \varepsilon_t \sim \text{HalfCauchy}(\cdot)$$

**Latent state components:**

1. **Local Linear Trend, 2nd-order:**
   $$\mu_t = \mu_{t-1} + \nu_{t-1} + \eta_t, \quad \nu_t = \nu_{t-1} + \zeta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2_\eta), \quad \zeta_t \sim \mathcal{N}(0, \sigma^2_\zeta)$$
   where $\mu_t$ is the unobserved trend, $\nu_t$ is the unobserved slope.

2. **Seasonality:** already modeled via exogenous dummies that map into the state vector.

All state equations use the **non-centered parametrization** (Matt Trick) to prevent NUTS hierarchical funnels. The slope variance $\sigma_\zeta$ uses a tight `HalfNormal(sigma = 0.05 × σ_Y)` prior — approximately 5% of the level innovation's reference scale - so that a short-run negative slope at the end of the Sep-Nov trough does **not** extrapolate aggressively over the 13-week holdout. The structural direction of the Q4 holiday spike is instead carried by the exogenous dummies: the trend models only smooth baseline drift. The level variance $\sigma_\eta$ uses `HalfCauchy(beta = 0.1 × σ_Y)`.

**Viability Assessment — Decision: KEEP BSTS with explicit positioning.**

Empirical evidence from fold_3 execution shows BSTS performs **marginally better** than plain Bayesian regression (1-5% lower WAPE with noticeably lower systematic bias). This marginal improvement is consistent with the theoretical expectation:

**Why BSTS adds value beyond Bayesian regression (despite parameter count concerns):**

1. **Adaptive baseline tracking.** The LLT component learns a smooth, time-varying intercept ($\mu_t$) that absorbs non-stationary baseline drift. Bayesian regression uses a fixed intercept, which forces the marketing coefficients ($\beta_i$) to compensate for baseline shifts — conflating organic trend with marketing effect. Even with tight slope priors ($\sigma_\zeta$ = 0.05 × σ_Y), the LLT provides enough flexibility to track the Sep-Nov → Q4 regime transition without aggressive extrapolation.
2. **Structural decomposition for thesis narrative.** BSTS produces an explicit Level + Regression decomposition (analogous to OLS + STL but probabilistic). This makes the F4 posterior KDE comparison richer: a thesis reviewer can see that BSTS's intercept is time-varying while Bayesian regression's is constant, illustrating the structural flexibility difference.
3. **Lower bias, not just lower WAPE.** The bias reduction matters more than WAPE improvement. BSTS's near-zero WBIAS across the holdout means it doesn't systematically over/under-predict during regime changes — the LLT absorbs the shift. Bayesian regression's fixed intercept creates systematic bias during transition periods.

**Why the overfitting concern is manageable:**
Effective parameter count: BSTS adds 4 structural parameters ($\sigma_\eta, \sigma_\zeta, \mu_0, \nu_0$) + T latent states. But the T latent states are not free parameters in the classical sense — they are hierarchically constrained by $\sigma_\eta$ and $\sigma_\zeta$. The tight slope prior ($\sigma_\zeta$ = 0.05 × σ_Y) effectively regularizes the slope to near-zero, meaning the LLT operates as a local-mean smoother (≈ 2-3 effective additional degrees of freedom, not T). The 107s sampling time is acceptable for a Docker-based pipeline with 4 chains.

**Framing directive for the thesis:** Position BSTS as "the probabilistic equivalent of OLS + STL decomposition" — not as a fundamentally different model from Bayesian regression. If BSTS and Bayesian regression converge in WAPE, frame it as: _"The probabilistic framework itself is the driver of superiority over OLS, not the specific state-space architecture."_ If BSTS's lower bias is statistically distinguishable, credit the LLT for adaptive baseline tracking.


# 4. Dual Prior Specifications (The DDA Synergy).

**CRITICAL EXPERIMENT ARCHITECTURE**: Bayesian models (3.2 and 3.3) must be instantiated and evaluated twice to test the impact of data-driven prior knowledge.

## 4.1. Scenario A: Domain Heuristic Priors (Uninformed).
The models rely purely on standard industry assumptions:

- Channel Coefficients ($\beta_i$): HalfNormal(sigma=1). (Constrained to be $\ge 0$, assuming marketing never hurts sales).
- AdStock ($\alpha$): Beta(alpha=2, beta=2) (Centered around 0.5 for all channels).
- Hill (K, S): K=Beta(2, 2), S=Gamma(3, 1). (The primary purpose of `S` is to quantify how much the impact of additional media spending declines after reaching a threshold. A gamma+ distribution is often used as it can easily represent both moderate and more pronounced diminishing returns depending on the shape params).
- Exogenous Coefficients ($\gamma_j$): Normal(mu=0, sigma=1) (can be negative).
- Noise ($\epsilon$): HalfCauchy(beta=1).

## 4.2. Scenario B: DDA-Informed Priors (The "Micro-to-Macro" Link).
The models' initial beliefs are shaped by the Relative Weights ($W_x$) calculated by the **Shapley Value** attribution model from the micro-plane.

**Why Shapley, not Markov Chain?**
Markov Chain weights measure a channel's **removal effect** on macro conversion probability — they capture sequencing and path dependency. Shapley values, by contrast, measure each channel's **average marginal contribution** across all possible coalitions - a concept that is directly analogous to a regression coefficient (the marginal increment in outcome per unit of presence). Because the MMM's $\beta_x$ parameters are marginal revenue coefficients, Shapley values represent a more epistemically coherent prior mean than Markov removal effects, which embed sequencing assumptions that the regression model does not model.

- Channel Coefficients ($\beta_x$): use a TruncatedNormal(mu=W*x \* C, sigma=0.75, lower=0). (where $C$ is a per-country scaling constant: $C = \bar{Y}*{\text{train}} / n\_{\text{channels}}$, mapping attribution proportions to the unit scale of weekly revenue. $\sigma = 0.75$ is deliberately chosen to be **informative** - wide enough to permit sampler to override if the data contradicts the DDA prior, but narrow enough to make Scenario B statistically distinguishable from Scenario A).
- Shapley weights $W_x$ are read at runtime from `dda_weights` table where `model_name = 'Shapley_DDA'` **AND `fold_id` matches the current MMM training fold**. If no Shapley rows exist (e.g., due to DDA fallback triggering `confidence_weight = -1`), the model **must log a warning** and fall back to Scenario A heuristic priors. This fallback state is recorded in the `prior_source` field of `eval_mmm` (@see **docs/data/final/output_contract.md** `eval_mmm` schema).
- AdStock ($\alpha$): adjust the Beta priors based on funnel position:
  - **Top Funnel** (tiktok, gads:discover): Beta(alpha=3, beta=1) - slower decay, longer memory.
  - **Mid Funnel** (gads:youtube, metads:inst): Beta(alpha=2, beta=2) - balanced decay.
  - **Bottom Funnel** (gads:search, metads:fb): Beta(alpha=1, beta=3) - fast decay, immediate purchase intent.


# 5. Dimensionality Reduction (Fallback).
Bayesian sampling requires dense, continuous data. If a region has sparse marketing spend, the chains will fail to converge.

1. Target Granularity (`confidence_weight = 0`): `date_week` + `macro_region`. Trigger conditions: if any (channel, region) tuple has a non-zero spend density $< 30\%$ (proportional sparsity check to handle expanding fold durations), OR if the region's mean weekly revenue is < $150. Same sparsity / r_hat conditions. Action: drop the `macro_region` dimension, move to Fallback 1.

2. Fallback 1 (`confidence_weight = -1`): `date_week` only. Logically aggregate all regions on a weekly basis and fit a single global model. Physically disaggregate the expected revenue back to region-level using the following proportional allocation:

**Revenue-Share Disaggregation Algorithm:**

1. Calculate each region $r$'s historical revenue share within the training window of the current fold: $\omega_r = \frac{\sum_{t \in \text{train}} Y_{r,t}}{\sum_{t \in \text{train}} Y_{\text{global},t}}$.
2. For each holdout week $t$, distribute the global model's predicted revenue: $\hat{Y}_{r,t} = \hat{Y}_{\text{global},t} \times \omega_r$.
3. Tag all disaggregated rows with `confidence_weight = -1` to signal that regional predictions are derived from a global fit, not region-specific estimation.

**Acknowledged limitation:** this proportional disaggregation assumes stable regional revenue shares across the holdout period. If a region's share shifts materially (e.g., due to a region-specific campaign burst), the disaggregation will be biased. A hierarchical (partial-pooling) model would theoretically solve this by learning region-specific intercepts shrunk toward the global mean. However, hierarchical models require more data per group than is available for the sparse regions that triggered the fallback in the first place (this is a catch-22: the region failed the density check and was routed to fallback because it lacked sufficient observations for region-level estimation, so it would equally lack observations for hierarchical shrinkage estimation). The proportional approach is the pragmatic, honest choice for this data regime. **Important Note:** Report the fallback rate (% of regions routed to Fallback 1) per fold alongside the regional metrics to quantify the scope of this limitation within the local file logs.


# 6. Output Contract Translation.
- Bayesian models sample intercept, saturation parameter $S$, and channel-contribution coefficients as part of their runtime. These acquired values are expected to be persisted table as `base_sales_intercept` and `mean_saturation_point` (within the `eval_mmm` table), as well as `incr_{channel_name}` (within the separate `mmm_channel_contribs` table).
- The `expected_net_revenue_usd` may be calculated as the mean of the posterior predictive distribution for the current fold for Bayesian models. For OLS, it may be the reconstructed level forecast (Trend + Seasonality + OLS Residual).

**Important Notes regarding result persistence:**
- Inferred bayesian posterior values, as well as sampling traces for the coefficients, MUST be persisted locally in the .npy and .nc files correspondingly. The file names MUST be of the form `posterior_trace_{model_name}_{fold_id}.nc` and `posterior_predictive_{model_name}_{fold_id}.npy`. Traces may be saved in a "thinned" format (e.g., every 3rd or 5th sample) rather than full chains - 1000 thinned samples should be sufficient for a smooth KDE, but takes only a fraction of the storage.
- Acquired beta coefficients from the baseline OLS model MUST be persisted locally in the `ols_channel_coeffs_{fold_id}` NPY or CSV file.
