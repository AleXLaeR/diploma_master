-- Allocates the daily country-level macro-budget from `insights` into per-channel spend buckets using:
--   1. Traffic volume (click counts) from `touchpoints_log`
--   2. Empirically-calibrated per-(region, channel) CPC weights from `channel_cpc_weights`.
--
-- Spec reference: docs/data/intermediate_datasets_implementation.md §2

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.insights_channel_spend`
PARTITION BY DATE_TRUNC(date, MONTH)
AS
WITH
-- Step 1: Map each touchpoint to (date, country_code) via the user's
-- attribution record.
touchpoint_with_country AS (
    SELECT
        DATE(tl.created_at)  AS touch_date,
        ua.country_code,
        tl.media_source
    FROM
        `{{ project }}.{{ dataset }}.touchpoints_log` AS tl
    INNER JOIN
        `{{ project }}.{{ dataset }}.users_attribution` AS ua
        ON tl.user_id = ua.user_id
    WHERE
        -- Exclude non-paid channels (CPC = 0)
        tl.media_source NOT IN ('organic', 'legacy_untracked')
),

-- Step 2: Count absolute clicks per (date, country, channel)
click_counts AS (
    SELECT
        touch_date,
        country_code,
        media_source,
        COUNT(*) AS click_count
    FROM touchpoint_with_country
    GROUP BY touch_date, country_code, media_source
),

-- Step 3: Apply per-(fold, region, channel) empirical CPC weight from channel_cpc_weights.
-- COALESCE to 1.0 ensures channels missing from the weight table (too sparse to calibrate)
-- still contribute at neutral weight rather than being zeroed out.
weighted_clicks_base AS (
    SELECT
        f.fold_id,
        cc.touch_date,
        cc.country_code,
        cc.media_source,
        cc.click_count,
        cc.click_count * COALESCE(w.cpc_weight, 1.0) AS weighted_clicks
    FROM click_counts AS cc
    CROSS JOIN (SELECT fold_id FROM `{{ project }}.{{ dataset }}.rocv_folds`) f
    LEFT JOIN `{{ project }}.{{ dataset }}.countries` AS c
        ON cc.country_code = c.country_code
    LEFT JOIN `{{ project }}.{{ dataset }}.channel_cpc_weights` AS w
        ON  COALESCE(c.region, 'ROW') = w.region
        AND cc.media_source           = w.media_source
        AND f.fold_id                 = w.fold_id
),

-- Step 4: Compute total weighted clicks per (fold, date, country)
total_wc AS (
    SELECT
        fold_id,
        touch_date,
        country_code,
        SUM(weighted_clicks) AS total_weighted_clicks
    FROM weighted_clicks_base
    GROUP BY fold_id, touch_date, country_code
),

-- Step 5: Join with insights spend and allocate proportionally.
allocated AS (
    SELECT
        wc.fold_id,
        wc.touch_date           AS date,
        wc.country_code,
        wc.media_source,
        imp.spend * (wc.weighted_clicks / twc.total_weighted_clicks)
            AS alloc_spend_in_usd
    FROM
        weighted_clicks_base AS wc
    INNER JOIN
        total_wc AS twc
        ON  wc.touch_date    = twc.touch_date
        AND wc.country_code  = twc.country_code
        AND wc.fold_id       = twc.fold_id
    INNER JOIN
        `{{ project }}.{{ dataset }}.insights` AS imp
        ON  wc.touch_date    = imp.date
        AND wc.country_code  = imp.country_code
    WHERE
        twc.total_weighted_clicks > 0       -- guard against division by zero
        AND imp.spend > 0                   -- skip zero-guard / ineligible countries
)

-- Final: exclude zero-spend rows (per spec)
SELECT
    fold_id,
    date,
    country_code,
    media_source,
    alloc_spend_in_usd
FROM allocated
WHERE alloc_spend_in_usd > 0;