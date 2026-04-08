-- Input for: MMM models.

-- Grain: date_week × country_group (16 macro-regions).
-- All revenue is net (order_amount - refund_amount): no further substractions are required at model level.
--
-- Exogenous variables (5 columns):
--   1, 2: fourier_sin_q1, fourier_cos_q1    — 13-week quarterly harmonic (identifiable
--                                           in as little as 20 training weeks for Fold #1)
--   3: revenue_anomaly_score             — Per-region 8-week rolling z-score of
--                                           net revenue. Data-driven and region-scoped.
--   4: is_structural_peak                — 1 when z-score > +1.5. Replaces previous collinear
--                                           `is_q4_holiday` and `is_omicron_wave` dummies.
--   5: is_sep_nov_trough                 — 1 when z-score < -0.75 for >=2 consecutive
--                                           weeks. Region-scoped (replaces global dummy).
--
-- Spec reference: docs/data/model_specific_marts.md §3
--                 docs/algorithms/mmm_models.md §2.1

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.mmm_timeseries`
PARTITION BY DATE_TRUNC(date_week, MONTH)
AS

WITH
-- -------------------------------------------------------------------------
-- Step 1: Continuous weekly spine × all observed macro-regions
-- -------------------------------------------------------------------------
week_spine AS (
    SELECT date_week
    FROM UNNEST(
        GENERATE_DATE_ARRAY(
            DATE '2021-04-05',    -- First Monday of the observation window
            DATE '2022-02-28',    -- Last Day of the observation window
            INTERVAL 7 DAY
        )
    ) AS date_week
),

all_regions AS (
    SELECT DISTINCT region AS country_group
    FROM `{{ project }}.{{ dataset }}.countries`
    WHERE region IS NOT NULL
),

base_grid AS (
    SELECT
        f.fold_id,
        ws.date_week,
        ar.country_group
    FROM week_spine AS ws
    CROSS JOIN all_regions AS ar
    CROSS JOIN (SELECT fold_id FROM `{{ project }}.{{ dataset }}.rocv_folds`) f
),

-- -------------------------------------------------------------------------
-- Step 2: Aggregate net revenue per (week, country_group)
-- Uses countries.region directly — no inline country_map CTE needed.
-- -------------------------------------------------------------------------
weekly_revenue AS (
    SELECT
        DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY)) AS date_week,
        COALESCE(c.region, 'ROW') AS country_group,
        SUM(
            CASE 
                WHEN p.order_status IN ('approved', 'settled_ok') 
                THEN p.order_amount_in_usd
                ELSE -COALESCE(p.refund_amount_in_usd, 0)
            END
        ) AS total_net_revenue_usd
    FROM
        `{{ project }}.{{ dataset }}.purchases` AS p
    INNER JOIN
        `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua
        ON p.user_id = ua.user_id
        AND ua.fold_id = 'fold_4'
        AND ua.is_synthetic = FALSE
    LEFT JOIN
        `{{ project }}.{{ dataset }}.countries` AS c
        ON ua.country_code = c.country_code
    WHERE
        p.order_status IN ('approved', 'settled_ok', 'refunded')
    GROUP BY
        date_week, country_group
),

-- -------------------------------------------------------------------------
-- Step 3: Pivot per-channel spend from insights_channel_spend
-- -------------------------------------------------------------------------
weekly_spend AS (
    SELECT
        ics.fold_id,
        DATE_TRUNC(ics.date, WEEK(MONDAY)) AS date_week,
        COALESCE(c.region, 'ROW') AS country_group,
        SUM(CASE WHEN ics.media_source = 'gads:search'   THEN ics.alloc_spend_in_usd ELSE 0 END) AS spend_gads_search,
        SUM(CASE WHEN ics.media_source = 'gads:youtube'   THEN ics.alloc_spend_in_usd ELSE 0 END) AS spend_gads_youtube,
        SUM(CASE WHEN ics.media_source = 'gads:discover'  THEN ics.alloc_spend_in_usd ELSE 0 END) AS spend_gads_discover,
        SUM(CASE WHEN ics.media_source = 'metads:inst'    THEN ics.alloc_spend_in_usd ELSE 0 END) AS spend_metads_inst,
        SUM(CASE WHEN ics.media_source = 'metads:fb'      THEN ics.alloc_spend_in_usd ELSE 0 END) AS spend_metads_fb,
        SUM(CASE WHEN ics.media_source = 'tiktok'         THEN ics.alloc_spend_in_usd ELSE 0 END) AS spend_tiktok
    FROM
        `{{ project }}.{{ dataset }}.insights_channel_spend` AS ics
    LEFT JOIN
        `{{ project }}.{{ dataset }}.countries` AS c
        ON ics.country_code = c.country_code
    GROUP BY
        fold_id, date_week, country_group
),

-- -------------------------------------------------------------------------
-- Step 4: Assemble raw grid
-- Zero-fill spend (zero spend on a week is a valid observation).
-- Zero-fill revenue only (NULL means no transaction in that week, which is
-- correctly represented as 0 revenue for MMM purposes).
-- -------------------------------------------------------------------------
raw_assembly AS (
    SELECT
        bg.fold_id,
        bg.date_week,
        bg.country_group,
        COALESCE(wr.total_net_revenue_usd, 0.0) AS total_net_revenue_usd,
        COALESCE(ws.spend_gads_search,   0.0)   AS spend_gads_search,
        COALESCE(ws.spend_gads_youtube,  0.0)   AS spend_gads_youtube,
        COALESCE(ws.spend_gads_discover, 0.0)   AS spend_gads_discover,
        COALESCE(ws.spend_metads_inst,   0.0)   AS spend_metads_inst,
        COALESCE(ws.spend_metads_fb,     0.0)   AS spend_metads_fb,
        COALESCE(ws.spend_tiktok,        0.0)   AS spend_tiktok
    FROM
        base_grid AS bg
    LEFT JOIN weekly_revenue AS wr
        ON bg.date_week = wr.date_week AND bg.country_group = wr.country_group
    LEFT JOIN weekly_spend AS ws
        ON bg.date_week = ws.date_week AND bg.country_group = ws.country_group AND bg.fold_id = ws.fold_id
),

-- -------------------------------------------------------------------------
-- Step 5: Compute 8-week rolling revenue statistics per region.
-- These drive the data-driven exogenous variables.
--
-- rolling_mean_8w / rolling_std_8w use ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
-- (not including the current week) to avoid look-ahead.
-- First 4 weeks of each region's series have insufficient history; their
-- anomaly score is set to 0 (neutral) by COALESCE.
-- -------------------------------------------------------------------------
rolling_stats AS (
    SELECT
        fold_id,
        date_week,
        country_group,
        total_net_revenue_usd,
        spend_gads_search, spend_gads_youtube, spend_gads_discover,
        spend_metads_inst, spend_metads_fb, spend_tiktok,
        AVG(total_net_revenue_usd) OVER (
            PARTITION BY fold_id, country_group
            ORDER BY date_week
            ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
        ) AS rolling_mean_8w,
        STDDEV_SAMP(total_net_revenue_usd) OVER (
            PARTITION BY fold_id, country_group
            ORDER BY date_week
            ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_8w,
        -- Consecutive-week trough flag helper: 1 if this week's revenue is
        -- more than 0.75 std below the rolling mean.
        CASE
            WHEN STDDEV_SAMP(total_net_revenue_usd) OVER (
                     PARTITION BY fold_id, country_group ORDER BY date_week
                     ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) > 0
            AND  total_net_revenue_usd
                 < AVG(total_net_revenue_usd) OVER (
                       PARTITION BY fold_id, country_group ORDER BY date_week
                       ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING)
                   - 0.75 * STDDEV_SAMP(total_net_revenue_usd) OVER (
                                 PARTITION BY fold_id, country_group ORDER BY date_week
                                 ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING)
            THEN 1 ELSE 0
        END AS is_below_trough_threshold
    FROM raw_assembly
)

-- -------------------------------------------------------------------------
-- Final: add exogenous control variables.
--
-- Fourier terms use a 13-WEEK (quarterly) period instead of 52-week annual.
-- Rationale: with 20–35 observable training weeks (ROCV Folds 1–4), a 52-week
-- harmonic cannot be statistically identified - it covers only 38–67% of one
-- full cycle and is near-collinear with a linear time trend. A 13-week period
-- requires only ~15 weeks of data to observe 1+ full cycles, making it
-- identifiable in all four folds.
--   fourier_sin_q1 = sin(2π × (ISOWEEK mod 13) / 13)
--   fourier_cos_q1 = cos(2π × (ISOWEEK mod 13) / 13)
-- -------------------------------------------------------------------------
SELECT
    fold_id,
    date_week,
    country_group AS macro_region,
    total_net_revenue_usd,
    spend_gads_search,
    spend_gads_youtube,
    spend_gads_discover,
    spend_metads_inst,
    spend_metads_fb,
    spend_tiktok,

    SIN(2 * ACOS(-1) * MOD(CAST(EXTRACT(ISOWEEK FROM date_week) AS INT64), 13) / 13.0)
        AS fourier_sin_q1,
    COS(2 * ACOS(-1) * MOD(CAST(EXTRACT(ISOWEEK FROM date_week) AS INT64), 13) / 13.0)
        AS fourier_cos_q1,

    COALESCE(
        SAFE_DIVIDE(
            total_net_revenue_usd - rolling_mean_8w,
            NULLIF(rolling_std_8w, 0)
        ),
        0.0
    ) AS revenue_anomaly_score,

    CASE
        WHEN rolling_std_8w > 0
         AND total_net_revenue_usd > rolling_mean_8w + 1.5 * rolling_std_8w
        THEN 1 ELSE 0
    END AS is_structural_peak,

    CASE
        WHEN is_below_trough_threshold = 1
         AND LAG(is_below_trough_threshold, 1, 0) OVER (
                 PARTITION BY fold_id, country_group ORDER BY date_week) = 1
        THEN 1 ELSE 0
    END AS is_sep_nov_trough

FROM
    rolling_stats;
