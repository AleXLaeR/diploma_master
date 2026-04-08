CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.survival_model_params`
(
    fold_id      STRING NOT NULL,
    segment      STRING NOT NULL,
    alpha        FLOAT64,
    beta         FLOAT64,
    c        FLOAT64
);

