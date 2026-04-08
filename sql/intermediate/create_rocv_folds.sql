-- Centralized definition of the 4-Fold Monthly Expanding Window (ROCV).
-- All models, metric aggregations, and evaluation scripts must join against
-- this table rather than hardcoding date boundaries in CTEs.
--
-- Fold 1: Train (Apr – Aug, 5 months) | Holdout (Sep – Nov, 3 months)
-- Fold 2: Train (Apr – Sep, 6 months) | Holdout (Oct – Dec, 3 months)
-- Fold 3: Train (Apr – Oct, 7 months) | Holdout (Nov – Jan, 3 months)
-- Fold 4: Train (Apr – Nov, 8 months) | Holdout (Dec – Feb, 3 months)

CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.rocv_folds` AS (
    SELECT 'fold_1' AS fold_id, DATE '2021-04-01' AS train_start, DATE '2021-09-01' AS train_end, DATE '2021-09-01' AS holdout_start, DATE '2021-12-01' AS holdout_end UNION ALL
    SELECT 'fold_2',            DATE '2021-04-01',              DATE '2021-10-01',              DATE '2021-10-01',                DATE '2022-01-01'               UNION ALL
    SELECT 'fold_3',            DATE '2021-04-01',              DATE '2021-11-01',              DATE '2021-11-01',                DATE '2022-02-01'               UNION ALL
    SELECT 'fold_4',            DATE '2021-04-01',              DATE '2021-12-01',              DATE '2021-12-01',                DATE '2022-03-01'
);