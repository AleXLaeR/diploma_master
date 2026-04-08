CREATE TABLE IF NOT EXISTS `{{ project }}.{{ dataset }}.mmm_channel_contribs`
(
    fold_id      STRING NOT NULL,
    model_name   STRING NOT NULL,
    segment      STRING NOT NULL,
    incr_gads_search               FLOAT64,
    incr_gads_youtube              FLOAT64,
    incr_gads_discover             FLOAT64,
    incr_metads_inst               FLOAT64,
    incr_metads_fb                 FLOAT64,
    incr_tiktok                    FLOAT64,
    actual_contrib_gads_search     FLOAT64,
    actual_contrib_gads_youtube    FLOAT64,
    actual_contrib_gads_discover   FLOAT64,
    actual_contrib_metads_inst     FLOAT64,
    actual_contrib_metads_fb       FLOAT64,
    actual_contrib_tiktok          FLOAT64
);

