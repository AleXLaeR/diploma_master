-- Deterministic Last-Click attribution baseline.
-- Counts conversions attributed to the FINAL touchpoint per converted user
-- and normalises to relative channel weights.

-- This query is materialized - it is consumed in-flight by the
-- Airflow task which feeds the result into the shared DDA translation layer.
--
-- Time window: strictly the fold's training period (date < '{{ train_end }}').
--   '{{ train_end }}' is templated per fold by the DAG callable.
--   Prevents data leakage — each fold only sees its own training conversions.
--
-- Spec reference: docs/algorithms/dda_models.md §2.1

WITH last_clicks AS (
    SELECT
        media_source,
        COUNT(*) AS conversions
    FROM
        `{{ project }}.{{ dataset }}.touchpoints_log`
    WHERE
        is_conversion = TRUE
        AND created_at < '{{ train_end }}'
        -- Exclude non-paid channels (CPC = 0)
        AND media_source NOT IN ('legacy_untracked', 'organic')
    GROUP BY
        media_source
),
total AS (
    SELECT SUM(conversions) AS total_conversions
    FROM last_clicks
)

SELECT
    lc.media_source,
    lc.conversions,
    ROUND(SAFE_DIVIDE(lc.conversions, t.total_conversions), 6) AS weight
FROM
    last_clicks AS lc
CROSS JOIN
    total AS t;
