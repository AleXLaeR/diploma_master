-- Per-fold idempotency is NOT handled here — each model's Python layer
-- should issue a fold-scoped DELETE (WHERE fold_id = '...') before writing,
-- so re-running a single fold is always safe.
--
-- Spec reference: docs/data/final/output_contract.md

CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.eval_dda`
(
    fold_id                 STRING   NOT NULL,
    model_name              STRING,
    forecast_period         DATE,
    expected_conversions    FLOAT64,
    actual_conversions      INT64,
    expected_cac_usd        FLOAT64,
    actual_cac_usd          FLOAT64,
    confidence_weight       INT64
)
PARTITION BY DATE_TRUNC(forecast_period, MONTH);


CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.eval_mmm`
(
    fold_id                  STRING   NOT NULL,
    model_name               STRING,
    forecast_period          DATE,
    segment                  STRING,
    expected_net_revenue_usd FLOAT64,
    actual_net_revenue_usd   FLOAT64,
    base_sales_intercept     FLOAT64,
    mean_saturation_point    FLOAT64,
    prior_source             STRING,
    confidence_weight        INT64
)
PARTITION BY DATE_TRUNC(forecast_period, MONTH);


CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.eval_survival`
(
    fold_id                 STRING   NOT NULL,
    model_name              STRING,
    forecast_period         DATE,
    segment                 STRING,
    rebill_period_t         INT64,
    expected_active_users   FLOAT64,
    actual_active_users     INT64,
    expected_ltv_usd        FLOAT64,
    actual_ltv_usd          FLOAT64,
    confidence_weight       FLOAT64
)
PARTITION BY DATE_TRUNC(forecast_period, MONTH);
