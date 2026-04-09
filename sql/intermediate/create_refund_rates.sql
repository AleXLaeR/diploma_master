-- Cross-cutting prerequisite: per-product refund rate lookup table.
-- Used by ALL statistical models in their calculations: net_revenue = gross_revenue * (1 - refund_rate)
-- Granularity: one row per product_id.
-- Time window: fold-specific training period only (date < fold train_end)
--              to prevent data leakage from the holdout.
--
-- Spec reference: docs/data/model_specific_marts_implementation.md §4

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.refund_rates` AS

WITH
rocv_folds AS (
    SELECT fold_id, train_end
    FROM `{{ project }}.{{ dataset }}.rocv_folds`
),

-- Explicitly define all possible products to ensure consistent grain across all folds,
-- even when certain products are missing from the raw 'purchases' table 
-- during specific periods (e.g. Apr 2021 - Sept 2021).
-- Reference: docs/data/initial/initial_schematics.md
all_products AS (
    SELECT 'SUB_WEEKLY' AS product_id UNION ALL
    SELECT 'SUB_MONTHLY' AS product_id UNION ALL
    SELECT 'SUB_3_MONTH' AS product_id
),

-- Construct the base grid (4 folds x 3 products = 12 rows)
grid AS (
    SELECT
        f.fold_id,
        f.train_end,
        p.product_id
    FROM
        rocv_folds AS f
    CROSS JOIN
        all_products AS p
),

-- Aggregate factual renewals and refunds per (fold, product)
aggregated AS (
    SELECT
        f.fold_id,
        p.product_id,
        SUM(COALESCE(p.refund_amount_in_usd, 0)) AS total_refunds,
        SUM(p.order_amount_in_usd) AS total_orders
    FROM
        rocv_folds AS f
    CROSS JOIN
        `{{ project }}.{{ dataset }}.purchases` AS p
    WHERE
        -- Only consider valid financial transactions
        p.order_status IN ('approved', 'settled_ok', 'refunded')
        -- Strict train-only window (prevent holdout leakage)
        AND CAST(p.order_date AS DATE) < f.train_end
    GROUP BY
        f.fold_id,
        p.product_id
)

SELECT
    g.fold_id,
    g.product_id AS sub_type,   -- Aliased to match docs/data/intermediate_datasets.md §5
    COALESCE(
        SAFE_DIVIDE(a.total_refunds, a.total_orders),
        0.0
    ) AS refund_rate
FROM
    grid AS g
LEFT JOIN
    aggregated AS a
    ON g.fold_id = a.fold_id AND g.product_id = a.product_id
ORDER BY
    fold_id,
    sub_type;
