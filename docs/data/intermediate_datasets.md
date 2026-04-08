# Context & Objective
This document defines the reverse-engineered tables constructed from the initial data aimed to resolve the data sparsity issue.
See also: __./intermediate_datasets_implementation.md__ for the implementation details.

# 1. `touchpoints_log` view
Synthesized multi-touch customer journeys. Extends the initial single-touch attribution by adhering to a predefined funnel-stage heuristic and time-decay logic. Partitioned in BigQuery DAILY by `created_at` column.

Schema:
- user_id: unique customer / synthesized churn user.
- created_at (Timestamp): time of this exact interaction.
- media_source: expanded list of channels (tiktok, metads:inst, gads:discover, metads:fb, gads:youtube, organic, gads:search, and **'legacy_untracked'** for orphaned transactions).
- contact_ordinal (Int): position of this exact interaction within customer journey.
- is_conversion (Boolean): **True** ONLY on the final touchpoint of a successful acquisition.

Data Preview:
user_id | created_at | media_source | contact_ordinal | is_conversion
user_000001 | 2021-08-06 13:01:15 UTC | organic | 1 | true  |
churn_000001 | 2022-02-07 16:57:37 UTC | metads:fb | 1 | false |
churn_000001 | 2022-03-09 22:50:53 UTC | gads:search | 2 | false |

# 2. `insights_channel_spend` view
Disaggregated ad spendings by country, media source from the `insights` table. Spend is allocated into specific `media_source` buckets using the traffic volume from `touchpoints_log`. Partitioned in BigQuery MONTHLY by `date` column.

Schema:
- date (Date): the date of spend.
- country_code: ISO-2 format.
- media_source: paid channel name (excludes `organic`).
- alloc_spend_in_usd (Float): proportionally allocated monetary spend.

Data Preview:
date | country_code | media_source | alloc_spend_in_usd
2021-04-01 | ES | tiktok | 11.035 |
2021-06-03 | GR | metads:fb | 45.20 |
2021-06-03 | AU | gads:search | 456.1 |


# 3. `channel_cpc_weights` view
Provides regional-specific paid channel weights necessary to construct the `insights_channel_spend` view.

Schema: 
- fold_id: ROCV fold name
- macro_region: one of 16 macro-group labels.
- media_source: paid channel name.
- cpc_weight (Float): normalized CPC of that channel in that region in that fold.

Data Preview:
fold_id | macro_region | media_source | cpc_weight
1 | EU_SOUTH | tiktok | 1.234 |
1 | EU_SOUTH | metads:fb | 0.876 |
1 | EU_NORTH | gads:search | 1.543 |

# 4. `rocv_folds` view
Centralized definition of the 4-Fold Monthly Expanding Window (ROCV). All models, metric aggregations, and evaluation scripts must join against this table rather than hardcoding date boundaries in CTEs.

Schema:
- fold_id.
- train_start (Date).
- train_end (Date).

Data Preview:
fold_id | train_start | train_end
fold_1 | 2021-04-01 | 2021-04-07 |
fold_2 | 2021-04-01 | 2021-04-14 |
fold_3 | 2021-04-01 | 2021-04-21 |
fold_4 | 2021-04-01 | 2021-04-28 |

## 5. `refund_rates` table
Static lookup table for per subscription type, per fold refund rates calculated from the training period.
Data Volume: 12 rows (3 subscription types x 4 folds).

Schema:
- fold_id: fold-specific calibration.
- sub_type: subscription type.
- refund_rate (Float): monetary ratio (sum(refunds) / sum(orders)).

Data Preview:
fold_id | sub_type | refund_rate
fold_1 | SUB_WEEKLY | 0.23 |
fold_1 | SUB_MONTHLY | 0.09 |
fold_1 | SUB_3_MONTH | 0.11 |
