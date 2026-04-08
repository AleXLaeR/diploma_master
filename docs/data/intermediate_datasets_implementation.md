Supplementary to: __./intermediate_datasets.md__

# 1. Generating `touchpoints_log` (Reverse-Engineering Journeys)
Because the raw `users_attribution` table only contains Last-Click data, we must synthetically reconstruct multi-touch journeys for DDA.

## 1.1. Relevant Data Anomaly: "orphaned" transactions (iOS 14.5 ATT Gap)
@see __./initial_data_discoveries.md__
**For DDA (micro-plane):** the `legacy_untracked` channel approach IS adequate. These ~17k orphaned iOS ATT users are assigned as a 1-touchpoint Bottom-Funnel path with `media_source = 'legacy_untracked'`. This channel is excluded from `dda_weights` (alongside `organic`), so it has zero impact on paid channel credit shares ($W_x$). It absorbs its own conversion value silently, preventing CAC inflation of tracked channels.

**For MMM (macro-plane):** orphaned user revenue MUST be **excluded** from the `mmm_timeseries` mart. Rationale: since these users have zero attributed ad spend, their revenue inflates the Bayesian `Intercept` (base sales / organic floor) and the OLS constant term, creating a systematically higher organic baseline that masks the true incremental effect of marketing. Since orphaned users are defined as those completely missing from the `users_attribution` table, the SQL simply needs to enforce an `INNER JOIN` on `users_attribution` during the revenue aggregation stage.

**For Survival (customer-base plane):** orphaned users ARE included in `cohorts_retention` because retention modeling is media-agnostic — the subscription renewal behaviour of an iOS opt-out user is no different from a tracked user. Excluding them would artificially reduce cohort sizes and bias survival estimates upward.

---
Funnel Stages:
- Top Funnel (Awareness): tiktok, gads:discover;
- Mid Funnel (Consideration): metads:inst, gads:youtube;
- Bottom Funnel (Intent): organic, gads:search, metads:fb.

## 1.2. Rule for Successful paths (conversions)
- Base: iterate over unique users present in the `purchases` table (left-joined with the `users_attribution`).
- Path Length Distribution (for known sources):
    - 35%: 1 touchpoint (Final Channel only).
    - 30%: 2 touchpoints (70% probability of [Mid $\rightarrow$ Final]; 30% probability of [Top $\rightarrow$ Final]).
    - 25%: 3 touchpoints (60% probability of [Top $\rightarrow$ Mid $\rightarrow$ Final]; 25% probability of [Mid $\rightarrow$ Mid $\rightarrow$ Final], 10% probability of [Top $\rightarrow$ Top $\rightarrow$ Final], 5% probability of [Mid $\rightarrow$ Top $\rightarrow$ Final]).
    - 10%: (4,5,6)+ touchpoints (Random mix of Top/Mid Funnel channels, ending with Final).
- Synthetic Path Generation:
    - Given that `users_attribution` only has `facebook` / `organic`, we generate the final touchpoint deterministically based on domain insights:
        - If `legacy_source` == `organic`: Final stage is **Bottom Funnel** (80% probability), **Mid Funnel** (20% probability).
        - If `legacy_source` == `facebook`: Final stage is **Mid Funnel** (50% probability), **Top Funnel** (40% probability), **Bottom Funnel** (10% probability).
    - The concrete channel selected within the dictated funnel stage shouldn't be purely random (e.g., "gads:youtube": 0.275488 and "metads:fb": 0.276335). Instead, it should be selected based on the actual distribution of funnel's channels in the ecommerce domain.
- Time Decay Generation (must be within (training_start_date, `subscription_date`] boundaries):
    - $T_0$ (Final Touch) = `purchases.subscription_date` (has `is_conversion = True`).
    - $T_{-1}$ (Previous Touch) = (1 week, minutes) $T_0 - \text{random}(60 \text{ to } 10800 \text{ minutes })$.
    - $T_{-2..-5}$ = (3 weeks, hours) $T_{-N} -  \text{random}(1 \text{ to } 504 \text{ hours })$.
- Important Note #0: provide deterministic jitter (e.g., +/-3,5%, different for every length to avoid 5/10 divisors) for path length distribution for realism.
- Important Note #1: across 4 ROCV folds, the path length distribution should drift deterministically.
- Important Note #2: across 4 ROCV folds, the convertion rate MUST be derived from the augmented `purchases` table.
- Orphan Handling (if approach above is chosen): strictly apply the `COALESCE(a.media_source, 'legacy_untracked')` rule.

## 1.3. Rule 2 for Unsuccessful paths (churn)
- Base: generate synthetic users (`user_id`::UUID, `is_conversion = False`) matching certain fold's churn rate (e.g., 100% - convertion rate).
- Path Logic: churns usually break at the Top or Mid funnel. But to maintain realistic conversion ratios, a portion of such paths (~12%) should reach the Bottom Funnel.

# 2. Generating `insights_channel_spend` (Dynamic Budget Allocation)
The raw `insights` table provides ad spendings only at the `(day, country)` level. To expand the modeling capabilities (to the `(day, country, media_source)` level), we must allocate this macro-budget to specific PAID channels using the traffic volume from `touchpoints_log`, weighted by **region-specific** CPC costs empirically derived **from actual data**: $$CLAMP(SUM(insights.spend) / COUNT(*), 0.1, 5.0)$$, normalized by that region's mean.

Important Note: CPC weights should be first persisted within an intermediate `channel_cpc_weights` view, and than queried to construct the `insights_channel_spend`.

Calculation Steps (per Country, per Day):
1. Count the absolute number of clicks ($N$) for each channel in that country on that day using `touchpoints_log` and augmented `users_attribution`.
2. Calculate the Weighted Clicks per channel: $WC_{channel} = N_{channel} \times \text{CPC\_Weight}_{channel}$.
3. Calculate the Total Weighted Clicks for the (day, country) pair: $\text{Total\_WC} = \sum WC_{i}$.
4. Allocate the spend:
$$Spend_{channel} = \text{Country\_Spend} \times \left( \frac{WC_{channel}}{\text{Total\_WC}} \right)$$
    - **Decision:** Use `Country_Spend` as-is (per-country, per-day). The `insights` table has 81% density, which is sufficient for per-country allocation without pooling. Regional pooling would mask country-level spend heterogeneity that the downstream MMM models need to capture (e.g., US receives disproportionately more spend than other countries in the same NA_US region).