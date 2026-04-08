# F10: Portfolio Budget Optimization Algorithm (mROAS Simulation)

This document specifies the procedure for synthesizing the outputs of all three analytical planes — Data-Driven Attribution, Media-Mix Modeling, and Survival Analysis — into a single actionable business simulation. The objective is to calculate the optimum budget allocation across paid marketing channels and demonstrate that the Probabilistic Strategy generates significantly higher expected LTV return on investment than the currently-used Deterministic Strategy.

## 1. Prerequisites and Inputs

The simulation requires the following components to be fully populated for the target fold (usually the final fold, `fold_4`):

1. **MMM Response Curves:** `mmm_channel_contribs` providing incremental revenue ($Y_c$) per channel, and `eval_mmm` providing the global baseline (`base_sales_intercept`) and scale.
2. **DDA Acquisition Conversion Costs:** `dda_weights` yielding attribution coefficients ($W_c$), mapped against historical spend to compute channel-level $CAC_c$.
3. **LTV Coefficients:** `survival_model_params` (BdW) and `survival_monetary_params` (Gamma-Gamma) providing expected long-term cohort value $E[LTV]$ per acquired user.
4. **Historical Spend Bounds:** Minimum and maximum weekly ad spend $S_{c}^{\text{lower, upper}}$ per channel from `insights_channel_spend` appended over the training horizon.

## 2. Strategy Definition

The simulation compares two competing strategic frameworks:

### A. Deterministic Heuristic Strategy (Status Quo)
*   **Response Curves:** OLS linear regression coefficients. Assumes linear scaling with no diminishing returns.
*   **Acquisition Cost:** Last-Click attribution CAC.
*   **LTV Multiplier:** Average CLV based on the `Baseline_Survival` exponential decay estimates.

### B. Probabilistic Data-Driven Strategy (Proposed)
*   **Response Curves:** Bayesian Structural Time Series (BSTS) evaluating Hill saturation curves.
*   **Acquisition Cost:** Shapley Value DDA projected CAC.
*   **LTV Multiplier:** Expected CLV derived from BdW survival + Discrete Gamma-Gamma monetary model.

## 3. The Objective Function

Let $B$ be the arbitrary total constrained budget (e.g., $100,000, $250,000, $500,000). The optimizer defines the optimal vector of channel budget allocations $\vec{b} = [b_1, b_2, ..., b_n]$ such that $\sum_{c=1}^n b_c = B$, subject to domain constraints $S_{c}^{\text{lower}} \le b_c \le S_{c}^{\text{upper}}$.

The optimization target is to **maximize the resulting expected Portfolio Net Revenue (LTVized)** subject to the constraint $\sum b_c \le B$.

### Sub-component 1: Channel Acquisitions
Calculate expected new user acquisitions ($A_c$) for a given budget allocation $b_c$:
*   **Deterministic:** $A_c = \frac{b_c}{\text{CAC}_{c, \text{ Last-Click}}}$
*   **Probabilistic:** $A_c = \text{Hill}(b_c \mid K_c, S_c) \times \frac{1}{\text{CAC}_{c, \text{ Shapley}}}$ (leveraging marginal saturation curves directly if acquisition saturation is decoupled, otherwise purely dependent on MMM saturation logic below).

### Sub-component 2: Expected Contribution
Let $Y(b_c)$ be the revenue contribution for channel $c$.
*   **Deterministic:** $Y(b_c) = \beta_{c,\text{OLS}} \times b_c$
*   **Probabilistic:** $Y(b_c) = \beta_{c,\text{BSTS}} \times \text{Hill}(b_c \mid K_c, S_c)$

*Note: $\beta_c$ and $K_c$ must be de-normalized from the [0, 1] scaling applied during MMM training to operate on raw dollar budgets.*

### Sub-component 3: LTV Multiplier Alignment
The simulated revenue $Y(b_c)$ projects the immediate-term (usually short-horizon) modeled revenue. We scale this generated marginal contribution by the expected LTV to represent the true long-term value created by this budget:
$$\text{Expected LTV Revenue}_c = Y(b_c) \times \frac{E[\text{LTV}]_{c}}{\text{ARPU}_{\text{short-term}}}$$
*(Assuming channel cohorts are heterogeneous; if channel-agnostic, the universal expected LTV applies.)*

## 4. Execution Logic (`scipy.optimize.minimize`)

For each budget scenario (e.g. $B \in \{100k, 250k, 500k\}$):
1.  Initialize standard budget bounds derived from empirical spend (prevent the optimizer from allocating 100% of the budget to an untested channel, bounding to e.g., $3 \times$ the maximum historical weekly spend).
2.  **Define Objective:** Negative LTVized Portfolio Revenue.
    $$-\left( \text{Base Sales} + \sum_{c=1}^n \left( Y(b_c) \times \frac{\text{LTV}}{\text{ARPU}} \right) \right)$$
3.  Execute Sequential Least SQuares Programming (SLSQP).
4.  Calculate metrics for the optimal allocation vector $\vec{b}^*$:
    *   **mROAS (Marginal ROAS):** The derivative of the response curve at the allocated amount $b_c^*$. For probabilistic models, this demonstrates the principle of equalizing marginal returns across all channels (the true mathematical optimum).
5.  Persist the final vectors to the `portfolio_budget_simulation` table.

## 5. Visual Proof Generation
The resulting dataset powers the F10 visual:
*   A stacked bar chart comparing the total LTVized expected revenue for Strategy A vs. Strategy B under the same constrained budgets.
*   Annotations demonstrating that Strategy B shifted budget away from quickly-saturating channels toward channels identified by Shapley as highly efficient, scaled by the persistent BdW LTV multiplier.
