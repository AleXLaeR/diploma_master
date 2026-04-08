-- Computes per-(region, media_source) empirical CPC weight from observed data.
-- Replaces otherwise hardcoded scalar weights in create_insights_channel_spend.sql.
--
-- Method (as specified in spec):
--   Empirical CPC = SUM(insights.spend) / COUNT(*)
--   for paid channels grouped by region.
--
-- Spec reference: docs/data/intermediate_datasets_implementation.md §2

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.channel_cpc_weights`
AS

WITH
rocv_folds AS (
    SELECT fold_id, train_end
    FROM `{{ project }}.{{ dataset }}.rocv_folds`
),

-- -------------------------------------------------------------------------
-- Compute Empirical CPC per region, channel, and fold
-- -------------------------------------------------------------------------
empirical_cpc_base AS (
    SELECT
        f.fold_id,
        cm.region,
        tl.media_source,
        SUM(i.spend) / COUNT(*) AS empirical_cpc
    FROM rocv_folds AS f
    CROSS JOIN `{{ project }}.{{ dataset }}.touchpoints_log` AS tl
    JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua 
        ON tl.user_id = ua.user_id
    JOIN `{{ project }}.{{ dataset }}.countries` AS cm 
        ON ua.country_code = cm.country_code
    JOIN `{{ project }}.{{ dataset }}.insights` AS i 
        ON DATE(tl.created_at) = i.date AND ua.country_code = i.country_code
    WHERE 
        tl.media_source NOT IN ('organic', 'legacy_untracked')
        AND DATE(tl.created_at) < f.train_end
    GROUP BY f.fold_id, cm.region, tl.media_source
),

-- -------------------------------------------------------------------------
-- Compute Regional Mean Empirical CPC as normalisation baseline
-- -------------------------------------------------------------------------
region_mean AS (
    SELECT
        fold_id,
        region,
        AVG(empirical_cpc) AS mean_cpc
    FROM empirical_cpc_base
    GROUP BY fold_id, region
)

-- -------------------------------------------------------------------------
-- Final: per-(fold, region, channel) CPC weight, normalised and clamped
-- -------------------------------------------------------------------------
SELECT
    ec.fold_id,
    ec.region,
    ec.media_source,
    GREATEST(
        LEAST(
            SAFE_DIVIDE(ec.empirical_cpc, NULLIF(rm.mean_cpc, 0)),
            5.0
        ),
        0.1
    ) AS cpc_weight
FROM empirical_cpc_base AS ec
JOIN region_mean AS rm
    ON ec.fold_id = rm.fold_id AND ec.region = rm.region
ORDER BY ec.fold_id, ec.region, ec.media_source;

