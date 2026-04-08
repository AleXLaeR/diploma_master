-- ============================================================================
-- consolidate_channel_contribs.sql
-- ============================================================================
-- Phase 3.1d: populate factual channel-contribution columns in
-- `mmm_channel_contribs`.
--
-- Output-contract columns updated:
--   - actual_contrib_gads_search
--   - actual_contrib_gads_youtube
--   - actual_contrib_gads_discover
--   - actual_contrib_metads_inst
--   - actual_contrib_metads_fb
--   - actual_contrib_tiktok
--
-- Ground-truth strategy (task #5 + consolidation spec §2.3):
--   1) Take holdout net revenue from `purchases` (status in approved/settled/refunded).
--   2) Build per-user paid-channel allocation weights from `touchpoints_log`.
--   3) Weight touchpoints by daily channel spend-share from `insights_channel_spend`
--      (same fold/date/country/media_source) with neutral fallback (=1.0).
--   4) Allocate each holdout user's revenue across paid channels via normalized
--      user weights.
--   5) Aggregate by fold + segment and normalize paid-channel contributions.
--
-- Segment support:
--   - Total_Macro_{country_code}
--   - Total_Macro_{region}
--   - Total_Macro_Global
--
-- Important: updates are target-row driven from `mmm_channel_contribs` so
-- missing factuals are written as 0.0.
-- ============================================================================

MERGE `{{ project }}.{{ dataset }}.mmm_channel_contribs` AS T
USING (
    WITH target_rows AS (
        SELECT DISTINCT
            m.fold_id,
            m.model_name,
            m.segment
        FROM `{{ project }}.{{ dataset }}.mmm_channel_contribs` AS m
    ),

    target_folds AS (
        SELECT DISTINCT fold_id
        FROM target_rows
    ),

    fold_windows AS (
        SELECT
            tf.fold_id,
            f.holdout_start,
            f.holdout_end
        FROM target_folds AS tf
        INNER JOIN `{{ project }}.{{ dataset }}.rocv_folds` AS f
            ON tf.fold_id = f.fold_id
    ),

    holdout_user_revenue AS (
        SELECT
            fw.fold_id,
            p.user_id,
            COALESCE(ua_imputed.country_code, ua_raw.country_code, 'ROW') AS country_code,
            SUM(
                p.order_amount_in_usd
                - COALESCE(p.refund_amount_in_usd, 0)
            ) AS holdout_net_revenue_usd
        FROM fold_windows AS fw
        INNER JOIN `{{ project }}.{{ dataset }}.purchases` AS p
            ON CAST(p.order_date AS DATE) >= fw.holdout_start
            AND CAST(p.order_date AS DATE) < fw.holdout_end
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
            AND ua_imputed.fold_id = fw.fold_id
            AND ua_imputed.is_synthetic = FALSE
        LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        GROUP BY fw.fold_id, p.user_id, country_code
    ),

    daily_channel_spend AS (
        SELECT
            ics.fold_id,
            ics.date,
            ics.country_code,
            ics.media_source,
            SUM(ics.alloc_spend_in_usd) AS alloc_spend_in_usd
        FROM `{{ project }}.{{ dataset }}.insights_channel_spend` AS ics
        WHERE ics.media_source IN (
            'gads:search', 'gads:youtube', 'gads:discover',
            'metads:inst', 'metads:fb', 'tiktok'
        )
        GROUP BY ics.fold_id, ics.date, ics.country_code, ics.media_source
    ),

    daily_country_spend AS (
        SELECT
            dcs.fold_id,
            dcs.date,
            dcs.country_code,
            SUM(dcs.alloc_spend_in_usd) AS total_alloc_spend_in_usd
        FROM daily_channel_spend AS dcs
        GROUP BY dcs.fold_id, dcs.date, dcs.country_code
    ),

    user_channel_scores AS (
        SELECT
            hur.fold_id,
            hur.user_id,
            hur.country_code,
            tl.media_source,
            SUM(
                COALESCE(
                    SAFE_DIVIDE(dcs.alloc_spend_in_usd, dct.total_alloc_spend_in_usd),
                    1.0
                )
            ) AS channel_score
        FROM holdout_user_revenue AS hur
        INNER JOIN `{{ project }}.{{ dataset }}.touchpoints_log` AS tl
            ON hur.user_id = tl.user_id
        INNER JOIN `{{ project }}.{{ dataset }}.rocv_folds` AS f
            ON hur.fold_id = f.fold_id
            AND DATE(tl.created_at) < f.holdout_end
        LEFT JOIN daily_channel_spend AS dcs
            ON dcs.fold_id = hur.fold_id
            AND dcs.date = DATE(tl.created_at)
            AND dcs.country_code = hur.country_code
            AND dcs.media_source = tl.media_source
        LEFT JOIN daily_country_spend AS dct
            ON dct.fold_id = hur.fold_id
            AND dct.date = DATE(tl.created_at)
            AND dct.country_code = hur.country_code
        WHERE tl.media_source IN (
            'gads:search', 'gads:youtube', 'gads:discover',
            'metads:inst', 'metads:fb', 'tiktok'
        )
        GROUP BY hur.fold_id, hur.user_id, hur.country_code, tl.media_source
    ),

    user_channel_weights AS (
        SELECT
            ucs.fold_id,
            ucs.user_id,
            ucs.country_code,
            ucs.media_source,
            SAFE_DIVIDE(
                ucs.channel_score,
                NULLIF(
                    SUM(ucs.channel_score) OVER (
                        PARTITION BY ucs.fold_id, ucs.user_id, ucs.country_code
                    ),
                    0
                )
            ) AS channel_weight
        FROM user_channel_scores AS ucs
    ),

    channel_revenue_country AS (
        SELECT
            hur.fold_id,
            CONCAT('Total_Macro_', hur.country_code) AS segment,
            ucw.media_source,
            SUM(hur.holdout_net_revenue_usd * ucw.channel_weight) AS channel_revenue_usd
        FROM holdout_user_revenue AS hur
        INNER JOIN user_channel_weights AS ucw
            ON hur.fold_id = ucw.fold_id
            AND hur.user_id = ucw.user_id
            AND hur.country_code = ucw.country_code
        GROUP BY hur.fold_id, segment, ucw.media_source
    ),

    channel_revenue_region AS (
        SELECT
            crc.fold_id,
            CONCAT('Total_Macro_', COALESCE(c.region, 'ROW')) AS segment,
            crc.media_source,
            SUM(crc.channel_revenue_usd) AS channel_revenue_usd
        FROM channel_revenue_country AS crc
        LEFT JOIN `{{ project }}.{{ dataset }}.countries` AS c
            ON REPLACE(crc.segment, 'Total_Macro_', '') = c.country_code
        GROUP BY crc.fold_id, segment, crc.media_source
    ),

    channel_revenue_global AS (
        SELECT
            crc.fold_id,
            'Total_Macro_Global' AS segment,
            crc.media_source,
            SUM(crc.channel_revenue_usd) AS channel_revenue_usd
        FROM channel_revenue_country AS crc
        GROUP BY crc.fold_id, crc.media_source
    ),

    channel_revenue_unified AS (
        SELECT * FROM channel_revenue_country
        UNION ALL
        SELECT * FROM channel_revenue_region
        UNION ALL
        SELECT * FROM channel_revenue_global
    ),

    factual_contribs AS (
        SELECT
            cru.fold_id,
            cru.segment,
            COALESCE(
                SAFE_DIVIDE(
                    SUM(IF(cru.media_source = 'gads:search', cru.channel_revenue_usd, 0.0)),
                    NULLIF(SUM(cru.channel_revenue_usd), 0.0)
                ),
                0.0
            ) AS actual_contrib_gads_search,
            COALESCE(
                SAFE_DIVIDE(
                    SUM(IF(cru.media_source = 'gads:youtube', cru.channel_revenue_usd, 0.0)),
                    NULLIF(SUM(cru.channel_revenue_usd), 0.0)
                ),
                0.0
            ) AS actual_contrib_gads_youtube,
            COALESCE(
                SAFE_DIVIDE(
                    SUM(IF(cru.media_source = 'gads:discover', cru.channel_revenue_usd, 0.0)),
                    NULLIF(SUM(cru.channel_revenue_usd), 0.0)
                ),
                0.0
            ) AS actual_contrib_gads_discover,
            COALESCE(
                SAFE_DIVIDE(
                    SUM(IF(cru.media_source = 'metads:inst', cru.channel_revenue_usd, 0.0)),
                    NULLIF(SUM(cru.channel_revenue_usd), 0.0)
                ),
                0.0
            ) AS actual_contrib_metads_inst,
            COALESCE(
                SAFE_DIVIDE(
                    SUM(IF(cru.media_source = 'metads:fb', cru.channel_revenue_usd, 0.0)),
                    NULLIF(SUM(cru.channel_revenue_usd), 0.0)
                ),
                0.0
            ) AS actual_contrib_metads_fb,
            COALESCE(
                SAFE_DIVIDE(
                    SUM(IF(cru.media_source = 'tiktok', cru.channel_revenue_usd, 0.0)),
                    NULLIF(SUM(cru.channel_revenue_usd), 0.0)
                ),
                0.0
            ) AS actual_contrib_tiktok
        FROM channel_revenue_unified AS cru
        GROUP BY cru.fold_id, cru.segment
    )

    SELECT
        tr.fold_id,
        tr.model_name,
        tr.segment,
        COALESCE(fc.actual_contrib_gads_search, 0.0) AS actual_contrib_gads_search,
        COALESCE(fc.actual_contrib_gads_youtube, 0.0) AS actual_contrib_gads_youtube,
        COALESCE(fc.actual_contrib_gads_discover, 0.0) AS actual_contrib_gads_discover,
        COALESCE(fc.actual_contrib_metads_inst, 0.0) AS actual_contrib_metads_inst,
        COALESCE(fc.actual_contrib_metads_fb, 0.0) AS actual_contrib_metads_fb,
        COALESCE(fc.actual_contrib_tiktok, 0.0) AS actual_contrib_tiktok
    FROM target_rows AS tr
    LEFT JOIN factual_contribs AS fc
        ON tr.fold_id = fc.fold_id
        AND tr.segment = fc.segment
) AS S
ON  T.fold_id = S.fold_id
AND T.model_name = S.model_name
AND T.segment = S.segment
WHEN MATCHED THEN
    UPDATE SET
        T.actual_contrib_gads_search = S.actual_contrib_gads_search,
        T.actual_contrib_gads_youtube = S.actual_contrib_gads_youtube,
        T.actual_contrib_gads_discover = S.actual_contrib_gads_discover,
        T.actual_contrib_metads_inst = S.actual_contrib_metads_inst,
        T.actual_contrib_metads_fb = S.actual_contrib_metads_fb,
        T.actual_contrib_tiktok = S.actual_contrib_tiktok;
