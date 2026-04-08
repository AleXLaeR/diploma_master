# Context & Objective
This document defines the exact data structures acquired from the business subject. These are the foundational tables already residing in Google BigQuery.

The initial data is rather sparse, anomalous to some extent and cannot be used as is (see __./initial_data_discoveries.md__).

## 1. `purchases` table
Contains denormalized transaction data, including product renewals / refunds.
Data Volume: ~240k rows, ~114k unique subscriptions.
Partitioned in BigQuery DAILY by `order_date` column.
Schema:
- user_id
- product_price (Float)
- product_id: denotes length of purchased subscription. Can be SUB_WEEKLY, SUB_MONTHLY or SUB_3_MONTH
- order_id: the purchase order / refund request created by the customer. __Note: may contain duplicate (product_id, order_status) within the same `rebill_duration` window - only the first row should be considered as a valid purchase__
- order_date (Timestamp).
- order_status: state of an order (approved, refunded, settled_ok - same as approved).
- order_amount_in_usd (Float): monetary value of a tnx.
- subscription_id: a session ID bound to the customer, persistent across same product rebills/refunds.
- subscription_date (Timestamp): time of the first user-product transaction.
- refund_amount_in_usd (Float): monetary volume of requested refund. NULL if `order_status != refunded`.
- rebill_duration (Int): number of days before rebilling is required.
- product_trial_price (Float): monetary income of trialed product version. When __product_id='SUB_WEEKLY'__ always =1, when __product_id='SUB_3_MONTH'__ =39.99, 1 or =__product_price__ (some regions don't have trial) otherwise.
- trial_duration (Int): number of days before trial expiration. When __product_id='SUB_MONTHLY'__ always =7, equal to __rebill_duration__ otherwise. __NOTE: the first row in a subscription_id group is always a trial__.
- purch_is_trial (Boolean) - is the subject of tnx a trialed version.
- rebill_number (Int): number of periods the product subscription was prolonged (rebilled) for, starting from 0 for trials.


Data Preview:

user_id | order_id | order_date | order_status |	order_amount_in_usd | subscription_id | subscription_date | product_id | product_price | trial_duration | rebill_duration | product_trial_price | refund_date | refund_amount_in_usd | purch_is_trial

<uuid> | <uuid> | 2021-08-06 13:01:14 UTC | approved | 1.0 | <uuid> | 2021-08-06 13:01:15 UTC | SUB_MONTHLY | 19.99 | 7	| 30 | 1.0 | null | null | true


## 2. `users_attribution` table
Contains the **Last-Click** customer acquisition data.
Data Volume: ~96k rows, all users are unique.
Partitioned in BigQuery DAILY by `created_at` column.
Schema:
- user_id.
- created_at (Timestamp): time of first conversion.
- country_code: in ISO-2 format.
- media_source: either 'organic' or 'facebook'.

Data Preview:
user_id |created_at | country_code | media_source
<uuid> | 2021-04-01 12:02:07 UTC | ES | organic |
<uuid> | 2021-06-03 17:34:13 UTC | GR | facebook |
<uuid> | 2021-04-15 00:47:17 UTC | AU | facebook |


## 3. `insights` table
Contains macro-level marketing spend (PPC inputs), aggregated by country and day.
Data Volume: ~68k rows, 334 days, from 14 (GS) to 300 (LR) observed spends per country.
Partitioned in BigQuery DAILY by `date` column.
Schema: 
- date.
- country_code: ISO-2 format.
- spend (Float): overall ad spendings in specified country at specified day.

Data Preview:
date | country_code | spend
2021-09-18 | HT | 19.285 |
2021-10-19 | HR | 26.535 |
2021-11-24 | MH | 0.624 |

## 4. `countries` table
A simple reference/dimension table mapping ISO-2 country codes to their names and regions.
Data Volume: 252 rows.
Schema:
- country_code: ISO-2 format.
- country_name.
- region: set of 16 macro-regions + fallback (ex. NA_US, EU_EAST, CIS, MENA... or R.OW for unclassified).

Data Preview:
country_code | country_name | region
SZ | Swaziland | SSA |
JM | Jamaica | LATAM |
US | United States | NA_US |