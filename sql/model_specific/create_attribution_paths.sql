-- Flattens chronological touchpoint events into a single path string per user.
-- Input for: Shapley Value & 2nd-Order Markov Chain DDA models.
--
-- Joins touchpoints_log journeys with purchases (rebill_number = 0) to
-- determine conversion status and monetary value.
--
-- Spec reference: docs/data/model_specific_marts.md §2
--                 docs/data/model_specific_marts_implementation.md §2

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.attribution_paths` AS

WITH
-- -------------------------------------------------------------------------
-- Step 1: Build chronological journey string per user from touchpoints_log
-- -------------------------------------------------------------------------
journeys AS (
    SELECT
        user_id,
        STRING_AGG(media_source, ' > ' ORDER BY created_at ASC) AS journey
    FROM
        `{{ project }}.{{ dataset }}.touchpoints_log`
    GROUP BY
        user_id
),

-- -------------------------------------------------------------------------
-- Step 2: Calculate net acquisition value per converted user
-- Only rebill_number = 0 purchases qualify (initial acquisition event).
-- Refunds are deducted here: do not propagate this to the model level.
-- -------------------------------------------------------------------------
acquisition_value AS (
    SELECT
        user_id,
        SUM(order_amount_in_usd - COALESCE(refund_amount_in_usd, 0))
            AS conversion_value_usd
    FROM
        `{{ project }}.{{ dataset }}.purchases`
    WHERE
        rebill_number = 0
        AND order_status IN ('approved', 'settled_ok', 'refunded')
    GROUP BY
        user_id
)

-- -------------------------------------------------------------------------
-- Final: LEFT JOIN to mark conversion status and attach monetary value.
-- Users with no matching purchase are non-converters (organic churn).
-- -------------------------------------------------------------------------
SELECT
    j.user_id,
    j.journey,
    CASE WHEN av.user_id IS NOT NULL THEN TRUE ELSE FALSE END AS is_converted,
    COALESCE(av.conversion_value_usd, 0.0)                    AS conversion_value_usd
FROM
    journeys AS j
LEFT JOIN
    acquisition_value AS av
    ON j.user_id = av.user_id;
