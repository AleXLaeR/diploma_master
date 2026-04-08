-- Fold-aware base input for the density-floor imputation algorithm.
-- Mirrors raw records into each fold's training window with no synthetic rows.
CREATE OR REPLACE TABLE `{{ project }}.{{ dataset }}.users_attribution_imputed`
PARTITION BY DATE(created_at)
CLUSTER BY fold_id, country_code, media_source
AS
SELECT
    f.fold_id,
    user_id,
    created_at,
    country_code,
    media_source,
    FALSE AS is_synthetic
FROM `{{ project }}.{{ dataset }}.users_attribution` AS ua
INNER JOIN `{{ project }}.{{ dataset }}.rocv_folds` AS f
    ON DATE(ua.created_at) < f.train_end;
