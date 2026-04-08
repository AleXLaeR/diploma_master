CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.dda_weights`
(
    fold_id      STRING NOT NULL,
    model_name   STRING,
    media_source STRING,
    weight       FLOAT64
);

