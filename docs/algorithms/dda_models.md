# Context & Objective

This module defines the contract for the implementation of **micro-analytical** plane of the experiment. The objective is to replace the deterministic Last-Click heuristic used by the business subject with probabilistic models that distribute acquisition value across all touchpoints in a customer's journey.
Constraint: the mathematical cores must be built from scratch using `numpy`, `scipy` and other baseline libraries. Direct attribution packages are not allowed.

# 1. Input Data & Preparation

Source Table: `attribution_paths`.
**ROCV Strategy**: All data preparation, transition matrices, and coalitions MUST be strictly re-evaluated per fold (1..4), isolated to the dynamically expanding training window of that specific fold. The resultant attribution weights will naturally drift between folds, proving the method's stability.

## 1.1. Parsing for 2nd-Order Markov Chains (State Space)

The script must parse the journey string into sequential transitions. A "state" is defined as a tuple of the last 2 channels.
Example journey: tiktok > metads:fb > gads:search (converted = 1)
2nd-Order States: (Start, tiktok), (tiktok, metads:fb), (metads:fb, gads:search).
Absorbing States: `Conversion` (if converted = 1) or `Null` (if converted = 0).

The algorithm must aggregate absolute frequencies of all observed transitions to build a 2D adjacency matrix, which is then row-normalized to form the empirical transition probability matrix $P$.

## 1.2. Parsing for Shapley Value (Coalitions)

Chronology is ignored. The script must convert the path string into a unique, unordered mathematical set (coalition) of distinct channels, removing duplicates within the same session.
Example journey: tiktok > metads:fb > tiktok $\rightarrow$ Coalition: {tiktok, metads:fb}.
The algorithm groups the data by these unique coalitions and calculates the characteristic function $v(S)$, which represents the total conversions or conversion rate achieved by that specific combination.

# 2. Mathematical Cores & Dimensionality Reduction (Fallback)

## 2.1. The Deterministic Baseline (Last-Click)

This represents the baseline attribution strategy. This should be a straightforward SQL query that calculates exact distribution of channel "weights" based on all records within `touchpoints_log` table where `is_conversion = True` (query all channels incl. `organic`, then normalize only PAID weights to 1.0).

## 2.2. Markov Chains (N-Order)

Algorithm simulates the removal of each channel to measure how critical it is to the overall probability of conversion.
Simulation: iteratively remove channel `x`. All transition probabilities pointing to `x` are redirected to the `Null` (churn) state, forming a modified matrix $P_{-x}$.

1. Parse journeys into state transitions (2nd-order → state = tuple of last N channels).
2. Build transition probability matrix $P$ (row-normalised adjacency). Compute base conversion rate ($CR_{total}$): the probability of reaching the absorbing state from the `Start` state using $P$.
3. Removal effect: for each channel $x$, redirect transitions to x into Null, re-compute CR, calculate $RE_{x}=1-\frac{CR_{-x}}{CR_{total}}$.
4. Normalise: $W_{x}=\frac{RE_{x}}{\sum{RE_{i}}}$.

Fallback Logic (Smoothing): sparse 2nd-order transitions that have fewer than K observations can cause mathematical instability when calculating probabilities. The algorithm must NOT drop the state to 1st-order (which breaks heterogeneous matrix summation rules and Markov chain property). Instead, implement **Add-1 (Laplacian) smoothing**: artificially add a frequency of 1 to all possible transitions in the $N$-order matrix. This strictly guarantees a dense 2D matrix where all rows sum to 1.0, preserving the deep channel connections. If smoothing is triggered, the final output is flagged with `confidence_weight = -1`.

## 2.3. Shapley Value

The algorithm calculates the fair share of each channel based on its marginal contribution to all possible coalitions.

1. Convert each journey into an unordered coalition of distinct channels $i \in N$.
2. Build the characteristic function $v(S)$ = total conversions for each observed coalition.
3. Approximate missing coalitions via weighted sub-coalition averages.
4. Compute Shapley values using the $v(S)$:
   $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} (v(S \cup \{i\}) - v(S))$$,
   where $n$ is the total number of available channels, and $|S|$ is the size of the coalition excluding $i$.
5. Normalise: $$W_i = \frac{\phi_i}{\sum \phi_k}$$.

Fallback Logic (Approximation): if a specific coalition $S$ of size $k$ never occurred in the historical data, $v(S)$ cannot be empirically measured. The algorithm must approximate $v(S)$ by taking the mean of its $k-1$ immediate sub-coalitions. If approximation is triggered, the final output is flagged with `confidence_weight = -1`.

**CRITICAL: Handling `legacy_untracked` and `organic` channels.**
The `legacy_untracked` channel (introduced by the iOS ATT imputation rule) has **zero advertising spend** and must not be persisted in `dda_weights` table, as well as `organic` to prevent division-by-zero and inflation of paid channel ROI. To allow that final weights should be re-normalized to the target sum of 1.0 after removing these channels from the result set.
In case of Shapley algorithm which can produce negative weights this way, affine shift should be used to maintain relative distances between channels but make all weights strictly positive, and add a 5% baseline range so the minimum channel isn't completely zeroed.

# 3. The Translation Algorithm.

**Important note: even though attribution isn't the right tool for forecasting, we still project the holdout CAC to show that ALL models (including baseline) results in relatively same metric scores, which solidifies the point that DDA is a RETROspective analysis instrument and not a predictive one.**

1. On the training dataset, calculate the total number of paid conversions and the exact **paid-only** net acquisition revenue (joining `purchases` and `users_attribution_imputed` tables, fold-scoped).
2. Distribute this conversion volume across **PAID** channels using calculated weights ($W_x$) for a given fold.
3. Query the `insights_channel_spend` table for the training period to get the actual historical spend per paid channel for a given fold.
4. Calculate Historical CAC and CR for a given fold, per channel.
5. Project these values on the holdout period using **EWMA-smoothed static carry-forward**:

**DDA Holdout Projection Algorithm (identical for all 3 models):**
The core assumption is that attribution-derived CAC and conversion rates are retrospective instruments and inherently non-stationary. To demonstrate this thesis claim, the projection method must be simple enough that all three models' out-of-sample WAPE converges (proving the error is structural, not method-dependent), but not naively rigid (which would exaggerate the error unfairly).

**Step 1: Train-Period Weekly CAC/CR Series.**
For each paid channel $c$ in the training window, compute weekly metrics:

- $\text{CAC}_{c,w} = \frac{\text{Total Spend}_{c,w}}{\text{Weighted Conversions}_{c,w}}$, where $\text{Weighted Conversions}_{c,w} = \text{Total Paid Conversions}_w \times W_c$ (using the model's attribution weights $W_c$).
- $\text{CR}_{c,w} = \text{Weighted Conversions}_{c,w}$ (absolute volume per week).

**Step 2: EWMA Smoothing.**
Apply exponentially-weighted moving average (smoothing factor $\alpha = 0.3$, emphasizing recent training weeks) to both series:
$$\hat{\text{CAC}}_c = \text{EWMA}(\text{CAC}_{c,1}, ..., \text{CAC}_{c,T_{\text{train}}}, \alpha=0.3)$$
$$\hat{\text{CR}}_c = \text{EWMA}(\text{CR}_{c,1}, ..., \text{CR}_{c,T_{\text{train}}}, \alpha=0.3)$$
The final EWMA value at $T_{\text{train}}$ becomes the static projection for all holdout weeks.

**Step 3: Holdout Projection.**
For each holdout week $w_h$:

- $\text{expected\_conversions}_{w_h} = \sum_c \hat{\text{CR}}_c$ (summed across all paid channels).
- $\text{expected\_cac\_usd}_{w_h} = \frac{\sum_c (\hat{\text{CR}}_c \times \hat{\text{CAC}}_c)}{\sum_c \hat{\text{CR}}_c}$ (volume-weighted average CAC).

**Rationale for EWMA α=0.3:** A low α gives more weight to recent periods, capturing the latest trend in CAC/CR while smoothing weekly noise. This makes the projection "generous" — if even an EWMA-smoothed static projection fails badly in the holdout, it conclusively demonstrates that DDA cannot forecast.
