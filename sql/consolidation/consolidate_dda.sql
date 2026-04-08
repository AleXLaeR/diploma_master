-- ============================================================================
-- consolidate_dda.sql
-- ============================================================================
-- Phase 3.1b: populate factual columns in eval_dda.
--
-- Output-contract columns updated:
--   - actual_conversions
--   - actual_cac_usd
--
-- Ground-truth rules (docs/data/final/consolidation_phase.md §2.2):
--   - purchases filter: rebill_number = 0
--   - statuses: approved / settled_ok / refunded
--   - paid-only attribution (exclude legacy_untracked, organic)
--   - spend source: insights_channel_spend (fold-scoped)
--
-- Important: source rows are driven by eval_dda keys so every predicted row
-- receives an update. Missing factuals are COALESCE'd to zero.
-- ============================================================================

MERGE `{{ project }}.{{ dataset }}.eval_dda` AS T
USING (
    WITH target_rows AS (
        SELECT DISTINCT
            e.fold_id,
            e.forecast_period
        FROM `{{ project }}.{{ dataset }}.eval_dda` AS e
        WHERE e.model_name IN ('Markov_DDA', 'Shapley_DDA', 'Baseline_LastClick')
    ),

    fold_windows AS (
        SELECT
            t.fold_id,
            t.forecast_period,
            f.holdout_start,
            f.holdout_end
        FROM target_rows AS t
        INNER JOIN `{{ project }}.{{ dataset }}.rocv_folds` AS f
            ON t.fold_id = f.fold_id
    ),

    factual_conversions AS (
        SELECT
            fw.fold_id,
            DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY)) AS forecast_period,
            COUNT(DISTINCT p.user_id) AS actual_conversions
        FROM fold_windows AS fw
        INNER JOIN `{{ project }}.{{ dataset }}.purchases` AS p
            ON CAST(p.order_date AS DATE) >= fw.holdout_start
            AND CAST(p.order_date AS DATE) < fw.holdout_end
            AND p.rebill_number = 0
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
            AND ua_imputed.fold_id = fw.fold_id
            AND ua_imputed.is_synthetic = FALSE
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        WHERE COALESCE(ua_imputed.media_source, ua_raw.media_source, 'legacy_untracked')
                NOT IN ('legacy_untracked', 'organic')
        GROUP BY fw.fold_id, forecast_period
    ),

    weekly_spend AS (
        SELECT
            fw.fold_id,
            DATE_TRUNC(ics.date, WEEK(MONDAY)) AS forecast_period,
            SUM(ics.alloc_spend_in_usd) AS total_spend
        FROM fold_windows AS fw
        INNER JOIN `{{ project }}.{{ dataset }}.insights_channel_spend` AS ics
            ON ics.fold_id = fw.fold_id
            AND ics.date >= fw.holdout_start
            AND ics.date < fw.holdout_end
            AND ics.media_source NOT IN ('legacy_untracked', 'organic')
        GROUP BY fw.fold_id, forecast_period
    )

    SELECT
        fw.fold_id,
        fw.forecast_period,
        CAST(COALESCE(fc.actual_conversions, 0) AS INT64) AS actual_conversions,
        COALESCE(
            SAFE_DIVIDE(
                COALESCE(ws.total_spend, 0.0),
                NULLIF(COALESCE(fc.actual_conversions, 0), 0)
            ),
            0.0
        ) AS actual_cac_usd
    FROM fold_windows AS fw
    LEFT JOIN factual_conversions AS fc
        ON fw.fold_id = fc.fold_id
        AND fw.forecast_period = fc.forecast_period
    LEFT JOIN weekly_spend AS ws
        ON fw.fold_id = ws.fold_id
        AND fw.forecast_period = ws.forecast_period
) AS S
ON  T.model_name IN ('Markov_DDA', 'Shapley_DDA', 'Baseline_LastClick')
AND T.fold_id = S.fold_id
AND T.forecast_period = S.forecast_period
WHEN MATCHED THEN
    UPDATE SET
        T.actual_conversions = S.actual_conversions,
        T.actual_cac_usd = S.actual_cac_usd;
