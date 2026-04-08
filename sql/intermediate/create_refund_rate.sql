-- Cross-cutting prerequisite: per-product refund rate lookup table.
-- Used by ALL statistical models in their calculations: net_revenue = gross_revenue * (1 - refund_rate)
-- Granularity: one row per product_id.
-- Time window: fold-specific training period only (date < fold train_end)
--              to prevent data leakage from the holdout.
--
-- Spec reference: docs/data/model_specific_marts_implementation.md §4

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.refund_rate` AS

WITH
rocv_folds AS (
    SELECT fold_id, train_end
    FROM `{{ project }}.{{ dataset }}.rocv_folds`
)

SELECT
    f.fold_id,
    p.product_id,
    COALESCE(
        SAFE_DIVIDE(
            SUM(COALESCE(p.refund_amount_in_usd, 0)),
            SUM(p.order_amount_in_usd)
        ),
        0.0
    ) AS refund_rate
FROM
    rocv_folds AS f
CROSS JOIN
    `{{ project }}.{{ dataset }}.purchases` AS p
WHERE
    p.order_status IN ('approved', 'settled_ok', 'refunded')
    AND CAST(p.order_date AS DATE) < f.train_end   -- Strict train-only (exclusive upper bound)
GROUP BY
    f.fold_id,
    p.product_id
ORDER BY
    f.fold_id,
    p.product_id;
