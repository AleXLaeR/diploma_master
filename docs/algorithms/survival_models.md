# Context & Objective

This module defines the contract for the implementation of the **customer-base** analytical plane. Since the business operates on a subscription model, continuous-time BTYD models (like Pareto/NBD) are invalid. We must model discrete renewal periods (billing cycles).
The objective is to predict the exact number of active subscriptions a cohort will retain during the holdout periods.
Constraint: the mathematical optimization cores for sBG, BdW must be built from scratch using `numpy` and `scipy.optimize`. The architecture must follow a strict Object-Oriented design, inheriting from a base `LTVModel` class.

# 1. Input Data & Preparation

Source Table: `cohorts_retention`.
The input is a flattened table containing the absolute counts of `active_users_at_t` and `churned_users_at_t` for each `cohort_id` at each observed `rebill_number` ($t$).
**ROCV Strategy**: Execution runs in an expanding window loop (folds 1..4). During each fold's fit phase, ONLY include cohorts whose `acquisition_week` falls strictly within the training boundary of the fold. Drop strictly out-of-time future cohorts from the estimator.
**Important Note**: for any fold, if a cohort's `acquisition_week` is within the training boundary, but its `max(rebill_number)` is within the holdout period, it should be included in the training data, but not used for final evaluation.

## 1.1. Cohort Matrix Assembly

Before passing data to the mathematical optimizers, it must be pivoted into a cohort matrix.
The `LTVModel.fit()` base method expects data to be a 2D array, where each inner list represents a contiguous cohort, and the values are the absolute number of retained users at period $t \in \{1, 2, ..., T_{obs}\}$. $n_{T_{obs}}$ is the number of users who successfully renewed at the final observed period $T_{obs}$ (right-censored users).

Concrete steps:

1. Group the flat dataframe by the current active granularity (as per **2.5. Dimensionality Reduction**).
2. Sort by `rebill_number` ascending.
3. Extract the `active_users_at_t` column as a list.
4. Assemble lists into a single `multi_cohort_data = [[...], [...], ...]` matrix.

# 2. Mathematical Cores & Dimensionality Reduction (Fallback)

## 2.1. Base Class: LTVModel

Abstract base class handling the optimization logic.

- `log_likelihood_multi_cohort(self, params, data)`: Calculates the negative LL. It evaluates the churn between periods and adds the right-censored survival probability for the final observed period.
- `optimize(self, data, is_multi_cohort)`: uses `scipy.optimize.minimize` passing `self.bounds`. MUST include a failure check: if np.isclose(res.x, self.bounds[:, 0]).any() to catch failed convergences. Fits MLE parameters via L-BFGS-B with multiple random restarts (required to rescue ~20-30% of convergence failures from poor starting points).

## 2.2. Survival Curve Baseline

A simple fits for quadratic (exact algebraic calculation), exponential (log-linearisation), and power-law fit (log-log linearisation) curves to the observed training periods and projected forward onto the holdout (Note: take either the best result or mean of the three). Are presented to show that deterministic "curve fitting" methods are not well suited for the retention projections in cases of business uncertainty.

## 2.3. Shifted-Beta-Geometric (sBG)

The sBG model assumes that a user's probability of churning ($p$) remains constant across renewal periods, but $p$ varies across the cohort according to a Beta distribution with parameters $\alpha$ and $\beta$, capturing cohort-level heterogenity.
Bounds: [[0.0001, 10000], [0.0001, 10000]] for $\alpha, \beta$.

Probabilities:

- probability of churning exactly at period $t$, $P(T=t)$: $P(T=1) = \frac{\alpha}{\alpha + \beta}$ $P(T=t) = \frac{\beta + t - 2}{\alpha + \beta + t - 1} P(T=t-1)$ for $t > 1$.
- Survival function (probability of surviving past period $t$), $S(t)$:$S(t) = 1 - \sum_{i=1}^{t} P(T=i)$. This is equivalent to the closed-form product: $$S(t) = \prod_{i=0}^{t-1} \frac{\beta + i}{\alpha + \beta + i}$$Recursive calculation: $S(0)=1$, $S(1)=1 - P(T=1)] = \frac{\beta}{\alpha+\beta}$, $S(2)=S(1) \times \frac{\beta+1}{\alpha+\beta+1} = \frac{\beta}{\alpha+\beta} \times \frac{\beta+1}{\alpha+\beta+1}$

NLL Function: minimization for $\alpha, \beta$:
$$LL(\alpha, \beta) = \sum_{t=1}^{T_{obs}} [ s_t \ln(P(T=t)) ] + n_{T_{obs}} \ln(S(T_{obs}))$$.

## 2.4. Beta-Discrete-Weibull (BdW)

The BdW model is an advanced extension of sBG. While sBG assumes a constant individual churn hazard, BdW introduces a shape parameter $c$ to model duration dependence (e.g. users become less likely to churn the longer they stay subscribed - introducing user-level heterogeneity). When the $c$ parameter is =1, BdW falls back to sBG.
Bounds: [[0.0001, 10000], [0.0001, 10000], [0.0001, 3]] for $\alpha, \beta, c$.

Probabilities: survival function $S(t)$ under BdW relies on the Beta function $B(x, y)$:
$$S(t|\alpha, \beta, c) = \frac{B(\alpha, \beta + t^c)}{B(\alpha, \beta)}$$.
**(Note: algorithm should use `scipy.special.betaln` for numerical stability when calculating the ratio). Formula: np.exp(betaln(alpha, beta + t\*\*c) - betaln(alpha, beta)), which guarantees $S(0) = 1$ since $t^c = 0^c = 0$.**

Probability of churning exactly at period $t$: $P(T=t) = S(t-1) - S(t)$ (where $S(0) = 1$).

NLL Function: minimization for $\alpha, \beta, c$:
$$LL(\alpha, \beta, c) = \sum_{t=1}^{T_{obs}} [ s_t \ln(S(t-1) - S(t)) ] + n_{T_{obs}} \ln(S(T_{obs}))$$.

## 2.5. Dimensionality Reduction (Fallback)

Because MLE optimization requires sufficient statistical mass to converge, algorithm MUST execute this hierarchical fallback per segment if the cohort size is too small or if `LTVModel.optimize()` raises a convergence exception.

1. Target Granularity (`confidence_weight = 0`): `acquisition_week` + `subscription_type` + `macro_region`. Trigger conditions: initial user count < 10 OR max(rebill_number) < 2 OR Optimization Fails. Action: drop `macro_region` dimension. Move to Fallback 1, state the dimension change in `segment` column: replace the region suffix with `_MACRO_GLOBAL`.
   - **Note on `max(rebill_number) < 2`**: At $T_{obs} = 1$, the sBG LL has exactly two degrees of freedom ($\alpha$, $\beta$) but only one observable transition (churn vs. survival at $t=1$). This is a severely underdetermined system: MLE will explore an unbounded ridge in the likelihood surface and produce extreme or near-boundary parameter estimates. At $T_{obs} = 2$, BdW adds a third parameter ($c$) to two observations, which is equally underdetermined. **Routing these cohorts directly to Fallback 1/2 is mandatory — do NOT attempt MLE on <=2 observed rebill periods.**

2. Fallback 1 (`confidence_weight = -1`): `acquisition_week` + `subscription_type`. Trigger conditions: same as above. Action: drop `subscription_type` dimension, group all users who ever bought this `subscription_type` into a per-week matrix. Move to Fallback 2, replace the sub id with `_ALL_SUB_`.

**Pooled-MLE + Weighted Disaggregation Algorithm:**
When a specific `(acquisition_week, subscription_type, macro_region)` segment fails at Target Granularity, pool all regions for that `(acquisition_week, subscription_type)` pair into a single cohort vector and fit sBG/BdW once on the pooled data. Then disaggregate the predicted $S(t)$ back to regional segments by applying the pooled survival curve to each region's starting cohort size $N_{r,0}$:
$$\hat{\text{Active}}_{r,t} = N_{r,0} \times S_{\text{pooled}}(t)$$

This assumes homogeneous churn dynamics across regions within the same subscription type and acquisition cohort — a strong assumption, but the same catch-22 applies as in MMM: the region failed the density check precisely because it lacks sufficient observations for region-specific estimation, so it equally lacks data for hierarchical shrinkage. The pooled approach is the pragmatic minimum.

**Acknowledged limitation:** Regional heterogeneity in churn behaviour (e.g., LATAM users may churn faster than EU_WEST users within the same subscription type) is lost in the pooled fit. If region-level evaluation reveals that fallback segments have systematically higher RMSE than full-confidence segments, this confirms the assumption's weakness and should be reported as a scope limitation.

3. Fallback 2 (`confidence_weight = -2`): only `acquisition_week`. Action: fit the model on per month basis, remove the week denotion from the `segment` name, and apply the resulting retention curve to the specific holdout week's starting user base.

4. Note: use `confidence_weight = -0.5` for boundary-hit (lower or upper) parameter values.

**3. The Translation Algorithm:**

**3.1. CRITICAL: Trial Period Exclusion.** Trials (`rebill_number = 0`) represent a discounted acquisition event, not recurring subscription revenue. The survival models MUST forecast **only** rebill periods starting from $t = 1$. The `cohorts_retention` mart provides $S(t)$ starting from $t = 0$, but the translation algorithm must use $N_{initial\_cohort} = \text{active\_users\_at\_t}(t=0)$ simply as the denominator for the survival curve - the first forecasted revenue event is at $t = 1$.

1. Determine the Horizon: for each `cohort_id`, identify the target forecast_period (iterating over the specific weeks in the fold's respective 3-month holdout).
2. Map Week to Rebill Period ($t$): calculate which discrete rebill period ($t_{target}, t \geq 1$) this forecast week represents for the cohort, based on their `acquisition_week` and the sub type's `rebill_duration`. Formula: $t_{target} = \lceil \frac{\text{holdout\_week\_start} - \text{acquisition\_week} - \text{trial\_duration}}{\text{rebill\_duration}} \rceil$
3. Predict Survival: extract the predicted survival rate $S(t_{target})$ from the trained model's `predicted_survival()` list.
4. Calculate Expected Retention (users): $\text{Expected Active Users} = N_{initial\_cohort} \times S(t_{target})$.
5. Calculate Expected Cohort LTV using the **Discrete Gamma-Gamma** monetary model:

**3.2. Discrete Gamma-Gamma (adapted for subscription context).**
The classical Gamma-Gamma model (Fader and Hardie, 2005) was originally designed for BTYD (continuous-time, non-contractual) contexts where purchase frequency and monetary value are modeled jointly. In this discrete, contractual subscription setting, the frequency dimension is already captured by the sBG/BdW survival models ($S(t)$ defines expected renewal count). The Gamma-Gamma model is adapted here to model **per-renewal monetary value only**, decoupled from frequency:

1. **Input data:** For each subscription type and region, extract all observed per-renewal monetary values $m_{i,t}$ from the training window (where $t \geq 1$, excluding trials). Each observation is one user's `order_amount_in_usd - COALESCE(refund_amount_in_usd, 0)` for a single rebill event.
2. **Model assumption:** Per-user average transaction value $\bar{M}_i$ follows a Gamma distribution $\text{Gamma}(p, \nu_i)$ across users, where $\nu_i$ itself follows $\text{Gamma}(q, \gamma)$ — creating the Gamma-Gamma mixture that captures cross-user monetary heterogeneity.
3. **MLE estimation:** Fit $(p, q, \gamma)$ per `subscription_type` using L-BFGS-B on the Gamma-Gamma log-likelihood. Fallback: if a segment has fewer than 30 observed monetary values, use the pooled (all-region) parameter estimates for that subscription type.
4. **Expected monetary value per renewal:** For a user with observed average transaction value $\bar{m}$ over $n$ renewals:
   $$E[M | \bar{m}, n, p, q, \gamma] = \frac{q \cdot \gamma + n \cdot \bar{m} \cdot p}{q + n \cdot p - 1}$$
   For cohort-level projection (where individual $\bar{m}$ is unavailable), use the unconditional expectation: $E[M] = \frac{q \cdot \gamma}{q - 1}$ (valid when $q > 1$).
5. **LTV calculation per cohort:**
   $$\text{Expected\_LTV}_{\text{cohort}} = N_{0} \times \sum_{t=1}^{T_{\text{horizon}}} S(t) \times E[M] \times (1 - \text{refund\_rate}_{\text{sub\_type}})$$
   where $S(t)$ comes from the survival model, $E[M]$ from Gamma-Gamma, and `refund_rate` from the fold-specific `refund_rates` lookup table.

**Why this works in discrete space:** key insight is that GG models the **monetary value distribution**, not the purchase timing. Timing in continuous BTYD is handled by Pareto/NBD; in our discrete context, timing is handled by sBG/BdW. The two models are naturally complementary: survival determines _when_ (and _if_) a renewal occurs, GG determines _how much_ that renewal is worth. The decoupling is standard in the BTYD literature and transfers directly to the discrete contractual setting.

**New table requirement:** persist fitted Gamma-Gamma parameters in `survival_monetary_params` table with schema: `fold_id`, `segment`, `p`, `q`, `gamma`, `expected_arpu`.

**Persistence semantics (critical for visualization joins):**
- `survival_monetary_params.segment` must reuse the same segment key space as `eval_survival.segment` (cohort ID or fallback descriptor), not plain `subscription_type`.
- Persistence is BdW-only.
- If a subscription type has no eligible monetary training rows in the fold (`rebill_number >= 1`, positive net amount, and `order_date < train_end`), its affected segments are still persisted with NULL `p/q/gamma` and `expected_arpu = 0.0`.
- Typical example: Fold 4 (`train_end = 2021-12-01`) can legitimately miss `SUB_3_MONTH` monetary fit, because first rebills may start after `2021-12-01`.

**3.2. Final Payload Generation:**
The output contract for this analytical plane is `eval_survival` table. The predictions must be strictly tagged with the active `fold_id`. If the Fallback was triggered, the `segment` name must reflect the rolled-up group (as per 2.5: Dimensionality Reduction).

**3.3. Evaluation Scope**
Due to the extreme sparsity of late-stage cohorts (e.g., `SUB_3_MONTH` having only 3 rebill observations in the training window), evaluation metrics (WAPE) are strictly filtered to `confidence_weight = 0`. Fallback segments (`-1`, `-2`) introduce disproportionate variance that masks the true predictive capability of the models on fully-observed baseline cohorts.

**SUB_3_MONTH Evaluation Caveat:**
For the `SUB_3_MONTH` product (90-day rebill cycle), the $t_{target}$ formula will produce $t_{target} = 1$ for nearly all cohorts within a 3-month holdout window - meaning the entire effectiveness of the model is evaluated on a single discrete renewal point. This has two consequences:

1. The sBG/BdW vs baseline **"crossing point"** proof cannot be visually demonstrated within the 3-month horizon for `SUB_3_MONTH`. This is an inherent constraint of the product's billing period, not a model deficiency.
2. **Mitigation:** "crossing point" thesis proof must be anchored primarily on `SUB_MONTHLY` cohorts (30-day cycle, yielding up to 3 measurable rebill periods within the 3-month holdout) and `SUB_WEEKLY` cohorts (7-day cycle, yielding up to 13 periods). The `SUB_3_MONTH` plane is evaluated purely on the accuracy of the $t=1$ survival prediction, which is still a valid but weaker claim. This scope limitation must be explicitly stated when presenting the survival analysis results.
