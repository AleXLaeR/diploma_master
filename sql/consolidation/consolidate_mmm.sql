-- ============================================================================
-- consolidate_mmm.sql
-- ============================================================================
-- Phase 3.1c: populate factual columns in eval_mmm.
--
-- Output-contract columns updated:
--   - actual_net_revenue_usd
--
-- Ground-truth rules (docs/data/final/consolidation_phase.md §2.3):
--   - all purchases (no rebill filter) within fold holdout window
--   - statuses: approved / settled_ok / refunded
--   - net revenue = SUM(order_amount_in_usd - COALESCE(refund_amount_in_usd, 0))
--
-- Segment support:
--   - Total_Macro_{country_code}
--   - Total_Macro_{region}
--   - Total_Macro_Global
--
-- Important: source rows are driven by eval_mmm keys so missing factuals are
-- written as 0.0 instead of leaving stale/null values.
-- ============================================================================

MERGE `{{ project }}.{{ dataset }}.eval_mmm` AS T
USING (
    WITH target_rows AS (
        SELECT DISTINCT
            e.fold_id,
            e.forecast_period,
            e.segment
        FROM `{{ project }}.{{ dataset }}.eval_mmm` AS e
        WHERE e.model_name IN (
            'Baseline_MMM_Reg',
            'MMM_Bayesian_Heuristic',
            'MMM_Bayesian_DDA',
            'MMM_BSTS_Heuristic',
            'MMM_BSTS_DDA'
        )
    ),

    fold_windows AS (
        SELECT
            t.fold_id,
            t.forecast_period,
            t.segment,
            f.holdout_start,
            f.holdout_end
        FROM target_rows AS t
        INNER JOIN `{{ project }}.{{ dataset }}.rocv_folds` AS f
            ON t.fold_id = f.fold_id
    ),

    unique_folds AS (
        SELECT DISTINCT fold_id, holdout_start, holdout_end
        FROM fold_windows
    ),

    factual_country AS (
        SELECT
            uf.fold_id,
            DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY)) AS forecast_period,
            COALESCE(ua_imputed.country_code, ua_raw.country_code, 'ROW') AS country_code,
            SUM(
                p.order_amount_in_usd
                - COALESCE(p.refund_amount_in_usd, 0)
            ) AS actual_net_revenue_usd
        FROM unique_folds AS uf
        INNER JOIN `{{ project }}.{{ dataset }}.purchases` AS p
            ON CAST(p.order_date AS DATE) >= uf.holdout_start
            AND CAST(p.order_date AS DATE) < uf.holdout_end
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
            AND ua_imputed.fold_id = uf.fold_id
            AND ua_imputed.is_synthetic = FALSE
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        GROUP BY uf.fold_id, forecast_period, country_code
    ),

    factual_region AS (
        SELECT
            fc.fold_id,
            fc.forecast_period,
            COALESCE(c.region, 'ROW') AS region,
            SUM(fc.actual_net_revenue_usd) AS actual_net_revenue_usd
        FROM factual_country AS fc
        LEFT JOIN `{{ project }}.{{ dataset }}.countries` AS c
            ON fc.country_code = c.country_code
        GROUP BY fc.fold_id, fc.forecast_period, region
    ),

    factual_global AS (
        SELECT
            fold_id,
            forecast_period,
            SUM(actual_net_revenue_usd) AS actual_net_revenue_usd
        FROM factual_country
        GROUP BY fold_id, forecast_period
    ),

    factual_unified AS (
        SELECT fold_id, forecast_period, segment, actual_net_revenue_usd
        FROM (
            SELECT
                fold_id,
                forecast_period,
                CONCAT('Total_Macro_', country_code) AS segment,
                actual_net_revenue_usd,
                1 AS priority
            FROM factual_country

            UNION ALL

            SELECT
                fold_id,
                forecast_period,
                CONCAT('Total_Macro_', region) AS segment,
                actual_net_revenue_usd,
                2 AS priority
            FROM factual_region

            UNION ALL

            SELECT
                fold_id,
                forecast_period,
                'Total_Macro_Global' AS segment,
                actual_net_revenue_usd,
                3 AS priority
            FROM factual_global
        )
        QUALIFY ROW_NUMBER() OVER(PARTITION BY fold_id, forecast_period, segment ORDER BY priority DESC) = 1
    )

    SELECT
        fw.fold_id,
        fw.forecast_period,
        fw.segment,
        COALESCE(fu.actual_net_revenue_usd, 0.0) AS actual_net_revenue_usd
    FROM fold_windows AS fw
    LEFT JOIN factual_unified AS fu
        ON fw.fold_id = fu.fold_id
        AND fw.forecast_period = fu.forecast_period
        AND fw.segment = fu.segment
) AS S
ON  T.model_name IN (
        'Baseline_MMM_Reg',
        'MMM_Bayesian_Heuristic',
        'MMM_Bayesian_DDA',
        'MMM_BSTS_Heuristic',
        'MMM_BSTS_DDA'
    )
AND T.fold_id = S.fold_id
AND T.forecast_period = S.forecast_period
AND T.segment = S.segment
WHEN MATCHED THEN
    UPDATE SET
        T.actual_net_revenue_usd = S.actual_net_revenue_usd;
