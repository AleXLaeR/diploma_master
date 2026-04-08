# Context & Objective
This document defines key observations about the initial data acquired without prior transformations.
These findings should be the basis used to design a relevant data augmentation pipeline before modeling begins.

TL;DR: Given the data sparsity, optimistic backtesting with a simple 8/3 training/holdout split is not enough. This leaves rolling-origin backtesting (e.g., forecasting period is repeatedly moved forward by a fixed number of observations, and forecasts are produced from each window) as the only viable option for evaluating the models' performance. 


# 1. `purchases` table
1. Revenue data is sporadic - if left as is, MMM models would NOT be able to capture meaningful results. They wouldn't produce an observed spike in the holdout dataset, because no analogous pattern exists in the training window (e.g., it is out-of-distribution). This requires to embed prior knowledge about the seasonality into all models (as exog. vars). Bayesian models would possibly require spike-and-slab priors or LLT to handle this. The aggregated monthly revenue is as follows:
y	m	sum_revenue_usd
2021	4	228515.689 |
2021	5	381577.672 |
2021	6	369654.644 |
2021	7	328918.778 |
2021	8	207961.142 |
2021	9	161929.130 |
2021	10	138759.966 |
2021	11	138911.486 |
2021	12	201826.774 |
2022	1	365211.632 |
2022	2	79115.189 |

2. Regarding rebills - it makes sense to track rebills only up to a year (11-th `rebill_number`) to capture meaningful retention volumes:
rebill_number	rebilled_tnxs
0	113829 |
1	57190  |
2	29510  |
3	17254  |
4	10814  |
5	6815   |
6	4188   |
7	2384 |
8	1259 |
9	608 |
10	246 |
11	93 |
...exclude >11
3. There are 114k distinct `subscription_id`s with total volume of purchases being 244k, giving an unrealistically high ~53% rebill rate.
4. There are only 324 users that have multiple distinct `subscription_id`s in their purchase history, and only 2 of them have 3.


# 2. `users_attribution` table
1. In initial repr, ~85% of the users are attributed to 'facebook' `media_source`, ~15% to 'organic'.
2. There are 225 countries in the dataset, 51 of which have <=15 active attributions, 27 have <=5 attributions. This data sparsity implies that per-country granularity is impossible without preliminary backfill / macro-aggregation to region level.
Per region, the distribution is as follows:
country_group	attributions
LATAM	21825 |
CIS	19181 |
MENA	10201 |
SSA	9790 |
SOUTH_ASIA	6722 |
ROW	6378 |
APAC_SEA	5814 |
EU_EAST	4968 |
NA_US	2985 |
EU_SOUTH	1900 |
APAC_EA	1672 |
APAC_ANZ	1259 |
EU_WEST	1123 |
UK	945 |
NA_CA	848 |
EU_DACH	809 |
EU_NORTH	384 |
3. Data Anomaly: "orphaned" transactions (iOS 14.5 ATT Gap)
The Issue: when `purchases` table is left-joined with this one, ~36k legitimate, recurring transactions between mid-2021 and early-2022 are missing their attribution record - highly likely caused by the Apple ATT rollout.
missing_attr_subs product_id
26720	SUB_MONTHLY |
9204	SUB_WEEKLY  |
390	    SUB_3_MONTH |


# 3. `insights` table
1. There are spendings recorded for 242 countries, 20 of which have <= 100$ in total, 69 <= 1000$.
2. There are 334 days, spending instances recorded from 1 (TF) to 300 (LR) observed spends per country.
3. There are 82 countries with avg spend <= 5$, 29 with <= 1.5$, 13 with <= 0.5$. This is expected as input for this model is used as a PPC output - so low spends are expected.

# 4. `countries` table: 27 countries from here have 0 records within `users_attribution` table.