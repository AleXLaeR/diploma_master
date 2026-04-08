-- Builds the flattened survival-analysis retention matrix.
-- Input for: survival analysis models.
--
-- Grain: cohort_id (acquisition_week + product_id + country_group) x rebill_number.
-- Tracks the exact decay of a specific cohort over discrete billing cycles (t).
--
-- Spec reference: docs/data/model_specific_marts.md §1
--                 docs/data/model_specific_marts_implementation.md §1
-- ============================================================================

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.cohorts_retention`
PARTITION BY DATE_TRUNC(acquisition_week, MONTH)
AS

WITH
-- -------------------------------------------------------------------------
-- Step 1: Identify the base cohort (rebill_number = 0 = acquisition event)
-- LEFT JOIN to users_attribution for country_code, then use countries.region.
-- -------------------------------------------------------------------------
base_cohort AS (
    SELECT
        p.user_id,
        DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY)) AS acquisition_week,
        p.product_id,
        COALESCE(c.region, 'ROW') AS country_group
    FROM
        `{{ project }}.{{ dataset }}.purchases` AS p
    LEFT JOIN
        `{{ project }}.{{ dataset }}.users_attribution` AS ua
        ON p.user_id = ua.user_id
    LEFT JOIN
        `{{ project }}.{{ dataset }}.countries` AS c
        ON ua.country_code = c.country_code
    WHERE
        p.rebill_number = 0
        AND p.order_status IN ('approved', 'settled_ok', 'refunded')
    GROUP BY
        p.user_id, acquisition_week, p.product_id, country_group
),

-- -------------------------------------------------------------------------
-- Step 2: Track rebills — join base users back to purchases (rebill >= 0).
-- -------------------------------------------------------------------------
user_rebills AS (
    SELECT
        bc.acquisition_week,
        bc.product_id,
        bc.country_group,
        p.user_id,
        p.rebill_number
    FROM
        base_cohort AS bc
    INNER JOIN
        `{{ project }}.{{ dataset }}.purchases` AS p
        ON bc.user_id = p.user_id
        AND bc.product_id = p.product_id
    WHERE 
        p.order_status IN ('approved', 'settled_ok', 'refunded')
        AND p.rebill_number > 0
        -- Filter out any rebills that technically occurred *after* the train limit
        -- (Even though the base purchase was before it).
        AND CAST(p.order_date AS DATE) <= '2021-11-30'
),

-- Also find each user's maximum rebill_number to derive churn point.
user_max_rebill AS (
    SELECT
        acquisition_week,
        product_id,
        country_group,
        user_id,
        MAX(rebill_number) AS max_rebill
    FROM user_rebills
    GROUP BY acquisition_week, product_id, country_group, user_id
),

-- -------------------------------------------------------------------------
-- Step 3: Aggregate per (cohort, rebill_number t)
-- active_users_at_t:  distinct users who paid at period t
-- churned_users_at_t: distinct users whose max_rebill = t (implicitly churned)
-- -------------------------------------------------------------------------
aggregated AS (
    SELECT
        ur.acquisition_week,
        ur.product_id,
        ur.country_group,
        ur.rebill_number,
        COUNT(DISTINCT ur.user_id)                                            AS active_users_at_t,
        COUNT(DISTINCT CASE
            WHEN umr.max_rebill = ur.rebill_number
            THEN ur.user_id
        END) AS churned_users_at_t
    FROM
        user_rebills AS ur
    INNER JOIN
        user_max_rebill AS umr
        ON  ur.user_id          = umr.user_id
        AND ur.acquisition_week = umr.acquisition_week
        AND ur.product_id       = umr.product_id
        AND ur.country_group    = umr.country_group
    GROUP BY
        ur.acquisition_week,
        ur.product_id,
        ur.country_group,
        ur.rebill_number
)

-- -------------------------------------------------------------------------
-- Final: assemble cohort_id and output
-- -------------------------------------------------------------------------
SELECT
    FORMAT_DATE('%G_Week-%V', a.acquisition_week)
        || '_' || a.product_id
        || '_' || a.country_group               AS cohort_id,
    a.acquisition_week,
    a.product_id AS subscription_type,
    a.country_group AS macro_region,
    a.rebill_number,
    a.active_users_at_t,
    a.churned_users_at_t
FROM aggregated AS a;
