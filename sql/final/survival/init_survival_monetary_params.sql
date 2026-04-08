CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.survival_monetary_params`
(
    fold_id       STRING NOT NULL,
    segment       STRING NOT NULL,
    p             FLOAT64,
    q             FLOAT64,
    gamma         FLOAT64,
    expected_arpu FLOAT64
);
