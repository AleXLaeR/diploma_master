-- ============================================================================
-- consolidate_survival.sql
-- ============================================================================
-- Phase 3.1a: populate factual columns in eval_survival.
--
-- Output-contract columns updated:
--   - actual_active_users
--   - actual_ltv_usd
--
-- Ground-truth rules (docs/data/final/consolidation_phase.md §2.1):
--   - purchases filter: rebill_number > 0
--   - statuses: approved / settled_ok / refunded
--   - net revenue = SUM(order_amount_in_usd - COALESCE(refund_amount_in_usd, 0))
--   - fallback-aware matching by confidence_weight
--
-- actual_ltv_usd strategy:
--   per matched segment grain, compute cumulative holdout net revenue over
--   rebill periods up to current t (windowed SUM ordered by rebill_period_t).
--
-- Important: source rows are driven by eval_survival keys so every predicted row
-- receives an update; missing factuals are COALESCE'd to 0.
-- ============================================================================

MERGE `{{ project }}.{{ dataset }}.eval_survival` AS T
USING (
    WITH target_rows AS (
        SELECT DISTINCT
            e.fold_id,
            e.model_name,
            e.forecast_period,
            e.segment,
            e.rebill_period_t,
            e.confidence_weight
        FROM `{{ project }}.{{ dataset }}.eval_survival` AS e
        WHERE e.model_name IN ('sBG', 'BdW', 'Baseline_Survival')
    ),

    fold_windows AS (
        SELECT
            t.fold_id,
            t.model_name,
            t.forecast_period,
            t.segment,
            t.rebill_period_t,
            t.confidence_weight,
            f.holdout_start,
            f.holdout_end
        FROM target_rows AS t
        INNER JOIN `{{ project }}.{{ dataset }}.rocv_folds` AS f
            ON t.fold_id = f.fold_id
    ),

    parsed_targets AS (
        SELECT
            fw.fold_id,
            fw.model_name,
            fw.forecast_period,
            fw.segment,
            fw.rebill_period_t,
            fw.confidence_weight,
            REGEXP_EXTRACT(
                fw.segment,
                r'^(.+?)_(SUB_WEEKLY|SUB_MONTHLY|SUB_3_MONTH)_.+$'
            ) AS parsed_acq_key,
            REGEXP_EXTRACT(
                fw.segment,
                r'(SUB_WEEKLY|SUB_MONTHLY|SUB_3_MONTH)'
            ) AS parsed_product_id,
            REGEXP_EXTRACT(
                fw.segment,
                r'(?:SUB_WEEKLY|SUB_MONTHLY|SUB_3_MONTH)_(.+)$'
            ) AS parsed_country_group
        FROM fold_windows AS fw
    ),

    cohort_origins AS (
        SELECT
            f.fold_id,
            p.user_id,
            p.product_id,
            FORMAT_DATE('%G-W%V', DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY))) AS acq_week_iso,
            FORMAT_DATE('%Y-%m-%d', DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY))) AS acq_week_date,
            FORMAT_DATE('%G_Week-%V', DATE_TRUNC(CAST(p.order_date AS DATE), WEEK(MONDAY))) AS acq_week_legacy,
            COALESCE(c.region, 'ROW') AS country_group
        FROM `{{ project }}.{{ dataset }}.rocv_folds` AS f
        INNER JOIN `{{ project }}.{{ dataset }}.purchases` AS p
            ON p.rebill_number = 0
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
            AND ua_imputed.fold_id = f.fold_id
            AND ua_imputed.is_synthetic = FALSE
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        LEFT JOIN `{{ project }}.{{ dataset }}.countries` AS c
            ON COALESCE(ua_imputed.country_code, ua_raw.country_code) = c.country_code
    ),

    factual_fine_raw AS (
        SELECT
            f.fold_id,
            o.acq_week_iso,
            o.acq_week_date,
            o.acq_week_legacy,
            o.product_id,
            o.country_group,
            p.rebill_number AS rebill_period_t,
            COUNT(DISTINCT p.user_id) AS actual_active_users,
            SUM(
                p.order_amount_in_usd
                - COALESCE(p.refund_amount_in_usd, 0)
            ) AS actual_period_net_revenue
        FROM `{{ project }}.{{ dataset }}.rocv_folds` AS f
        INNER JOIN cohort_origins AS o
            ON f.fold_id = o.fold_id
        INNER JOIN `{{ project }}.{{ dataset }}.purchases` AS p
            ON p.user_id = o.user_id
            AND p.product_id = o.product_id
            AND p.rebill_number > 0
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
            AND CAST(p.order_date AS DATE) >= f.holdout_start
            AND CAST(p.order_date AS DATE) < f.holdout_end
        GROUP BY
            f.fold_id,
            o.acq_week_iso,
            o.acq_week_date,
            o.acq_week_legacy,
            o.product_id,
            o.country_group,
            rebill_period_t
    ),

    factual_fine AS (
        SELECT
            ff.*,
            SUM(ff.actual_period_net_revenue) OVER (
                PARTITION BY ff.fold_id, ff.acq_week_iso, ff.product_id, ff.country_group
                ORDER BY ff.rebill_period_t
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS actual_ltv_usd
        FROM factual_fine_raw AS ff
    ),

    factual_fb1_raw AS (
        SELECT
            fold_id,
            acq_week_iso,
            acq_week_date,
            acq_week_legacy,
            product_id,
            rebill_period_t,
            SUM(actual_active_users) AS actual_active_users,
            SUM(actual_period_net_revenue) AS actual_period_net_revenue
        FROM factual_fine_raw
        GROUP BY
            fold_id,
            acq_week_iso,
            acq_week_date,
            acq_week_legacy,
            product_id,
            rebill_period_t
    ),

    factual_fb1 AS (
        SELECT
            fb1.*,
            SUM(fb1.actual_period_net_revenue) OVER (
                PARTITION BY fb1.fold_id, fb1.acq_week_iso, fb1.product_id
                ORDER BY fb1.rebill_period_t
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS actual_ltv_usd
        FROM factual_fb1_raw AS fb1
    ),

    factual_fb2_raw AS (
        SELECT
            fold_id,
            product_id,
            rebill_period_t,
            SUM(actual_active_users) AS actual_active_users,
            SUM(actual_period_net_revenue) AS actual_period_net_revenue
        FROM factual_fine_raw
        GROUP BY fold_id, product_id, rebill_period_t
    ),

    factual_fb2 AS (
        SELECT
            fb2.*,
            SUM(fb2.actual_period_net_revenue) OVER (
                PARTITION BY fb2.fold_id, fb2.product_id
                ORDER BY fb2.rebill_period_t
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS actual_ltv_usd
        FROM factual_fb2_raw AS fb2
    ),

    matched AS (
        SELECT
            p.fold_id,
            p.model_name,
            p.forecast_period,
            p.segment,
            p.rebill_period_t,
            CAST(
                COALESCE(
                    CASE
                        WHEN p.confidence_weight >= -0.5 THEN ff.actual_active_users
                        WHEN p.confidence_weight > -1.5 THEN fb1.actual_active_users
                        ELSE fb2.actual_active_users
                    END,
                    0
                ) AS INT64
            ) AS actual_active_users,
            COALESCE(
                CASE
                    WHEN p.confidence_weight >= -0.5 THEN ff.actual_ltv_usd
                    WHEN p.confidence_weight > -1.5 THEN fb1.actual_ltv_usd
                    ELSE fb2.actual_ltv_usd
                END,
                0.0
            ) AS actual_ltv_usd
        FROM parsed_targets AS p
        LEFT JOIN factual_fine AS ff
            ON p.fold_id = ff.fold_id
            AND p.parsed_product_id = ff.product_id
            AND p.parsed_country_group = ff.country_group
            AND p.rebill_period_t = ff.rebill_period_t
            AND p.parsed_acq_key IN (ff.acq_week_iso, ff.acq_week_date, ff.acq_week_legacy)
            AND p.confidence_weight >= -0.5
        LEFT JOIN factual_fb1 AS fb1
            ON p.fold_id = fb1.fold_id
            AND p.parsed_product_id = fb1.product_id
            AND p.rebill_period_t = fb1.rebill_period_t
            AND p.parsed_acq_key IN (fb1.acq_week_iso, fb1.acq_week_date, fb1.acq_week_legacy)
            AND p.confidence_weight > -1.5
            AND p.confidence_weight < -0.5
        LEFT JOIN factual_fb2 AS fb2
            ON p.fold_id = fb2.fold_id
            AND p.parsed_product_id = fb2.product_id
            AND p.rebill_period_t = fb2.rebill_period_t
            AND p.confidence_weight <= -1.5
    )

    SELECT DISTINCT
        fold_id,
        model_name,
        forecast_period,
        segment,
        rebill_period_t,
        actual_active_users,
        actual_ltv_usd
    FROM matched
) AS S
ON  T.model_name = S.model_name
AND T.fold_id = S.fold_id
AND T.forecast_period = S.forecast_period
AND T.segment = S.segment
AND T.rebill_period_t = S.rebill_period_t
WHEN MATCHED THEN
    UPDATE SET
        T.actual_active_users = S.actual_active_users,
        T.actual_ltv_usd = S.actual_ltv_usd;
