"""
generate_touchpoints.py
=======================
Synthesizes multi-touch customer journeys from the `purchases`
and `users_attribution` BigQuery tables to produce the `touchpoints_log` table.

Spec reference: docs/data/intermediate_datasets_implementation.md §1

Algorithm:
  § Conversion paths (§1.2):
      - LEFT JOIN purchases → users_attribution (raw, NOT fold-scoped imputed).
        COALESCE(media_source, 'legacy_untracked')
      - legacy_untracked → 1-touch Bottom-Funnel path (is_conversion=True)
      - Known-source → path length from jittered + fold-drifted distribution:
            35% 1-touch | 30% 2-touch | 25% 3-touch | 10% 4-6 touch
      - Final-touch channel by legacy_source probabilities (with jitter)
      - Preceding touches by spec §1.2 sub-probabilities per path length
      - Time decay: T₀ = subscription_date;
                    T₋₁ = T₀ − rand(60..10800 min);
                    T₋₂+ = Tₙ − rand(1..504 h);
                    clamped to (training_start, subscription_date]

  § Churn paths (§1.3):
      - Synthetic churn users with UUID v4 identifiers.
      - Count derived from per-fold first-rebill retention rate as conv rate proxy.
      - Paths mostly Top/Mid funnel; 12% reach Bottom.

  Why raw `users_attribution` for the base join (not `users_attribution_imputed`):
      touchpoints_log is fold-independent, covering the full observation window.
      `users_attribution_imputed` is fold-scoped (created_at < fold's train_end),
      which would exclude ~30% of users attributed in the holdout period,
      inflating legacy_untracked from ~17k to ~47k.  The imputed table's
      synthetic records (designed for DDA model stability) should not influence
      the base user extraction.

Idempotency: WRITE_TRUNCATE to touchpoints_log.
"""

from __future__ import annotations

import logging
import math
import os
import uuid as _uuid

import numpy as np
import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
BQ_PROJECT = os.environ.get("BQ_PROJECT", "studious-spirit-440213-t6")
BQ_DATASET = os.environ.get("BQ_DATASET", "diploma_mg")

TRAINING_START = pd.Timestamp("2021-04-01", tz="UTC")
OBSERVATION_END = pd.Timestamp("2022-02-28", tz="UTC")

# ---------------------------------------------------------------------------
# Funnel stage definitions (spec §1.1)
#   Top    (Awareness):     tiktok, gads:discover
#   Mid    (Consideration): metads:inst, gads:youtube
#   Bottom (Intent):        organic, gads:search, metads:fb
# ---------------------------------------------------------------------------
TOP_FUNNEL = ["tiktok", "gads:discover"]
MID_FUNNEL = ["metads:inst", "gads:youtube"]
BOTTOM_FUNNEL = ["organic", "gads:search", "metads:fb"]

ALL_CHANNELS = TOP_FUNNEL + MID_FUNNEL + BOTTOM_FUNNEL

# ---------------------------------------------------------------------------
# In-funnel channel selection weights (spec §1.2:
#   "should be selected based on the actual distribution of funnel's channels
#    in the ecommerce domain" — non-equal, realistic CPC-based splits)
#
# Top:    TikTok dominates awareness vs Discovery for edu-apps
# Mid:    Instagram outpaces YouTube in mid-funnel for subscription apps
# Bottom: Search converts best; FB retargeting 2nd; Organic 3rd
# ---------------------------------------------------------------------------
TOP_FUNNEL_PROBS = [0.683, 0.317]             # tiktok, gads:discover
MID_FUNNEL_PROBS = [0.612, 0.388]             # metads:inst, gads:youtube
BOTTOM_FUNNEL_PROBS = [0.413, 0.351, 0.236]   # organic, gads:search, metads:fb

# Combined Top+Mid weights for ≥4-touch and churn paths
_TOP_W, _MID_W = 0.40, 0.60
TOP_MID_CHANNELS = TOP_FUNNEL + MID_FUNNEL
TOP_MID_PROBS = (
    [p * _TOP_W for p in TOP_FUNNEL_PROBS]
    + [p * _MID_W for p in MID_FUNNEL_PROBS]
)

# ---------------------------------------------------------------------------
# Base path-length distribution (spec §1.2)
# ---------------------------------------------------------------------------
PATH_LEN_WEIGHTS_BASE: dict[int, float] = {1: 0.35, 2: 0.30, 3: 0.25, 4: 0.10}

# Per-length deterministic jitter (spec §1.2 Note #0):
# "different for every length to avoid 5/10 divisors"
# Applied as MULTIPLICATIVE fraction of each length's base weight:
#   jittered_w = base_w × (1 + jitter_pct)
PATH_LEN_JITTER_PCT: dict[int, float] = {
    1: -0.0321,   # −3.21% of 0.35 → ~0.3388
    2: +0.0173,   # +1.73% of 0.30 → ~0.3052
    3: +0.0463,   # +4.63% of 0.25 → ~0.2616
    4: -0.0289,   # −2.89% of 0.10 → ~0.0971
}

# ---------------------------------------------------------------------------
# Final-touch probabilities per legacy_source (spec §1.2, jittered)
# ---------------------------------------------------------------------------
ORGANIC_FINAL_BOTTOM_PROB = 0.817    # ~80%, jittered
ORGANIC_FINAL_MID_PROB = 0.183       # complement

FACEBOOK_FINAL_MID_PROB = 0.513      # ~50%, jittered
FACEBOOK_FINAL_TOP_PROB = 0.387      # ~40%, jittered
FACEBOOK_FINAL_BOTTOM_PROB = 0.100   # ~10%

# 2-touch sub-probabilities (spec §1.2, jittered)
TWO_TOUCH_MID_FINAL = 0.697   # ~70%
TWO_TOUCH_TOP_FINAL = 0.303   # complement

# 3-touch sub-probabilities (spec §1.2 exact)
THREE_TOUCH_TOP_MID_FINAL = 0.60
THREE_TOUCH_MID_MID_FINAL = 0.25
THREE_TOUCH_TOP_TOP_FINAL = 0.10
THREE_TOUCH_MID_TOP_FINAL = 0.05

# ---------------------------------------------------------------------------
# Fold-aware drift targets (spec §1.2 Note #1)
# ---------------------------------------------------------------------------
_FOLD_DRIFT: dict[str, dict[int, float]] = {
    "fold_1": {1: 0.40, 2: 0.30, 3: 0.22, 4: 0.08},
    "fold_2": {1: 0.35, 2: 0.30, 3: 0.24, 4: 0.11},
    "fold_3": {1: 0.32, 2: 0.30, 3: 0.26, 4: 0.12},
    "fold_4": {1: 0.29, 2: 0.28, 3: 0.29, 4: 0.14},
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _normalise(w: dict[int, float]) -> dict[int, float]:
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def _apply_jitter(base: dict[int, float]) -> dict[int, float]:
    return _normalise({
        k: max(0.001, base[k] * (1.0 + PATH_LEN_JITTER_PCT.get(k, 0.0)))
        for k in base
    })


def _fold_path_weights(fold_id: str | None, rng: np.random.Generator) -> dict[int, float]:
    """Jittered + fold-drifted path-length weights (spec §1.2 Notes #0, #1)."""
    raw = dict(_FOLD_DRIFT.get(fold_id, PATH_LEN_WEIGHTS_BASE)) if fold_id else dict(PATH_LEN_WEIGHTS_BASE)
    alpha = np.array([raw[k] * 200 for k in sorted(raw)])
    noisy = rng.dirichlet(alpha)
    noisy_dict = {k: float(noisy[i]) for i, k in enumerate(sorted(raw))}
    return _apply_jitter(noisy_dict)


def _pick(rng: np.random.Generator, choices: list[str], probs: list[float]) -> str:
    cum = np.cumsum(probs)
    r = rng.random()
    for ch, c in zip(choices, cum):
        if r < c:
            return ch
    return choices[-1]


def _make_uuid(rng: np.random.Generator) -> str:
    """Deterministic UUID v4 from seeded RNG (spec §1.3: user_id::UUID)."""
    b = bytearray(rng.bytes(16))
    b[6] = (b[6] & 0x0F) | 0x40   # version 4
    b[8] = (b[8] & 0x3F) | 0x80   # variant 1
    return str(_uuid.UUID(bytes=bytes(b)))


def _select_final_channel(rng: np.random.Generator, legacy_source: str) -> str:
    """Final (converting) touchpoint channel per spec §1.2 probabilities."""
    if legacy_source == "organic":
        if rng.random() < ORGANIC_FINAL_BOTTOM_PROB:
            return _pick(rng, BOTTOM_FUNNEL, BOTTOM_FUNNEL_PROBS)
        return _pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS)

    if legacy_source == "facebook":
        roll = rng.random()
        if roll < FACEBOOK_FINAL_MID_PROB:
            return _pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS)
        if roll < FACEBOOK_FINAL_MID_PROB + FACEBOOK_FINAL_TOP_PROB:
            return _pick(rng, TOP_FUNNEL, TOP_FUNNEL_PROBS)
        return _pick(rng, BOTTOM_FUNNEL, BOTTOM_FUNNEL_PROBS)

    # Unknown source → conservative Bottom fallback
    return _pick(rng, BOTTOM_FUNNEL, BOTTOM_FUNNEL_PROBS)


def _generate_funnel_sequence(
    rng: np.random.Generator,
    path_len: int,
    legacy_source: str,
) -> list[str]:
    """Channel sequence [earliest … final] for a converting user (spec §1.2)."""
    final = _select_final_channel(rng, legacy_source)

    if path_len == 1:
        return [final]

    if path_len == 2:
        if rng.random() < TWO_TOUCH_MID_FINAL:
            return [_pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS), final]
        return [_pick(rng, TOP_FUNNEL, TOP_FUNNEL_PROBS), final]

    if path_len == 3:
        roll = rng.random()
        c1 = THREE_TOUCH_TOP_MID_FINAL
        c2 = c1 + THREE_TOUCH_MID_MID_FINAL
        c3 = c2 + THREE_TOUCH_TOP_TOP_FINAL
        if roll < c1:                     # Top → Mid → Final (60%)
            return [_pick(rng, TOP_FUNNEL, TOP_FUNNEL_PROBS),
                    _pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS), final]
        if roll < c2:                     # Mid → Mid → Final (25%)
            return [_pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS),
                    _pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS), final]
        if roll < c3:                     # Top → Top → Final (10%)
            return [_pick(rng, TOP_FUNNEL, TOP_FUNNEL_PROBS),
                    _pick(rng, TOP_FUNNEL, TOP_FUNNEL_PROBS), final]
        return [_pick(rng, MID_FUNNEL, MID_FUNNEL_PROBS),   # Mid → Top → Final (5%)
                _pick(rng, TOP_FUNNEL, TOP_FUNNEL_PROBS), final]

    # ≥4 touch: random Top/Mid mix, ending with final
    seq = [_pick(rng, TOP_MID_CHANNELS, TOP_MID_PROBS) for _ in range(path_len - 1)]
    seq.append(final)
    return seq


def _generate_timestamps(
    rng: np.random.Generator,
    subscription_date: pd.Timestamp,
    path_len: int,
    training_start: pd.Timestamp = TRAINING_START,
) -> list[pd.Timestamp]:
    """Backwards time-decay ending at subscription_date (spec §1.2).

    T₀   = subscription_date
    T₋₁  = T₀ − rand(60 … 10 800 min)
    T₋₂+ = Tₙ − rand(1 … 504 h)
    Clamped to (training_start, subscription_date].
    """
    ts: list[pd.Timestamp] = [subscription_date]
    current = subscription_date
    floor = training_start + pd.Timedelta(minutes=1)

    for step in range(1, path_len):
        if step == 1:
            delta = pd.Timedelta(minutes=int(rng.integers(60, 10801)))
        else:
            delta = pd.Timedelta(hours=int(rng.integers(1, 505)))

        candidate = current - delta
        if candidate < floor:
            candidate = floor
        current = candidate
        ts.append(current)

    ts.reverse()  # earliest first
    return ts


def _generate_churn_sequence(rng: np.random.Generator, path_len: int) -> list[str]:
    """Churn paths: mostly Top/Mid; 12% reach Bottom Funnel (spec §1.3)."""
    seq: list[str] = []
    for _ in range(path_len):
        if rng.random() < 0.12:
            seq.append(_pick(rng, BOTTOM_FUNNEL, BOTTOM_FUNNEL_PROBS))
        else:
            seq.append(_pick(rng, TOP_MID_CHANNELS, TOP_MID_PROBS))
    return seq


# ---------------------------------------------------------------------------
# Core pure-logic function (testable without BigQuery)
# ---------------------------------------------------------------------------

def build_touchpoints_log(
    users_df: pd.DataFrame,
    fold_id: str | None,
    n_churn: int,
    training_start: pd.Timestamp = TRAINING_START,
    observation_end: pd.Timestamp = OBSERVATION_END,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build the full touchpoints_log DataFrame.

    Parameters
    ----------
    users_df : must have columns: user_id, subscription_date, media_source.
               Contains ALL users with rebill_number=0 from purchases LEFT JOIN
               users_attribution.  No order_status filtering.
    fold_id  : ROCV fold for path distribution drift (spec §1.2 Note #1).
    n_churn  : number of synthetic churn users to generate.
    training_start  : lower timestamp bound for clamping.
    observation_end : upper bound for churn acquisition dates.
    seed            : deterministic RNG seed.

    Returns
    -------
    DataFrame: user_id, created_at, media_source, contact_ordinal, is_conversion.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    users_df = users_df.copy()
    users_df["subscription_date"] = pd.to_datetime(users_df["subscription_date"], utc=True)

    legacy_mask = users_df["media_source"] == "legacy_untracked"
    legacy_idx = users_df.index[legacy_mask]
    known_idx = users_df.index[~legacy_mask]

    # ── 1a. Legacy-untracked → single Bottom-Funnel touchpoint ────────────
    for idx in legacy_idx:
        row = users_df.loc[idx]
        rows.append({
            "user_id": row["user_id"],
            "created_at": row["subscription_date"],
            "media_source": "legacy_untracked",
            "contact_ordinal": 1,
            "is_conversion": True,
        })

    # ── 1b. Known-source → probabilistic multi-touch ─────────────────────
    if len(known_idx) > 0:
        path_weights = _fold_path_weights(fold_id, rng)
        keys = sorted(path_weights.keys())
        probs = [path_weights[k] for k in keys]
        lengths = rng.choice(keys, size=len(known_idx), p=probs)
        # bucket 4 → expand to random 4–6
        four_plus = lengths == 4
        if four_plus.any():
            lengths[four_plus] = rng.integers(4, 7, size=int(four_plus.sum()))

        for i, idx in enumerate(known_idx):
            row = users_df.loc[idx]
            path_len = int(lengths[i])
            sub_date: pd.Timestamp = row["subscription_date"]

            channel_seq = _generate_funnel_sequence(rng, path_len, row["media_source"])
            timestamps = _generate_timestamps(rng, sub_date, path_len, training_start)

            for ordinal, (ts, ch) in enumerate(zip(timestamps, channel_seq)):
                rows.append({
                    "user_id": row["user_id"],
                    "created_at": ts,
                    "media_source": ch,
                    "contact_ordinal": ordinal + 1,
                    "is_conversion": ordinal == path_len - 1,
                })

    logger.info(
        "Built %d touchpoint rows for %d converting users "
        "(%d legacy-untracked, %d known-source). fold_id=%s",
        len(rows), len(users_df), int(legacy_mask.sum()), len(known_idx), fold_id,
    )

    # ── 2. Churn journeys (spec §1.3) ────────────────────────────────────
    churn_rows: list[dict] = []
    obs_seconds = int((observation_end - training_start).total_seconds())
    churn_lengths = rng.choice([1, 2, 3], size=n_churn, p=[0.40, 0.40, 0.20])

    for i in range(n_churn):
        user_id = _make_uuid(rng)
        path_len = int(churn_lengths[i])
        acq_date = training_start + pd.Timedelta(seconds=int(rng.integers(0, obs_seconds)))

        seq = _generate_churn_sequence(rng, path_len)
        timestamps = _generate_timestamps(rng, acq_date, path_len, training_start)

        for ordinal, (ts, ch) in enumerate(zip(timestamps, seq)):
            churn_rows.append({
                "user_id": user_id,
                "created_at": ts,
                "media_source": ch,
                "contact_ordinal": ordinal + 1,
                "is_conversion": False,
            })

    logger.info("Built %d touchpoint rows for %d synthetic churn users.", len(churn_rows), n_churn)

    # ── 3. Combine & validate ────────────────────────────────────────────
    full_df = pd.DataFrame(rows + churn_rows)
    full_df["created_at"] = pd.to_datetime(full_df["created_at"], utc=True)
    full_df["contact_ordinal"] = full_df["contact_ordinal"].astype(int)
    full_df["is_conversion"] = full_df["is_conversion"].astype(bool)

    # Integrity: each converting user has exactly 1 is_conversion=True
    conv_counts = full_df.loc[full_df["is_conversion"]].groupby("user_id").size()
    multi = conv_counts[conv_counts > 1]
    if not multi.empty:
        raise AssertionError(
            f"Found {len(multi)} users with >1 is_conversion=True touchpoints!"
        )

    logger.info(
        "Total touchpoints_log: %d rows, %d distinct users "
        "(%d converters, %d churners). fold_id=%s",
        len(full_df), full_df["user_id"].nunique(),
        int(full_df["is_conversion"].any() and len(conv_counts)),
        n_churn, fold_id,
    )
    return full_df


# ---------------------------------------------------------------------------
# BigQuery I/O layer
# ---------------------------------------------------------------------------

def _extract_base_users(
    client: bigquery.Client,
    project: str,
    dataset: str,
) -> pd.DataFrame:
    """purchases LEFT JOIN hybrid attribution → one row per acquiring user.

    Attribution source priority (hybrid extraction):

      1. PRIMARY: users_attribution_imputed (fold_4, is_synthetic=FALSE).
         These are training-window users whose media_source is already aligned
         with every downstream fold-scoped join (channel_cpc_weights,
         insights_channel_spend, consolidate_dda). Using the imputed table
         as the primary source ensures downstream INNER JOINs against it
         will always find a matching record.

      2. FALLBACK: raw users_attribution, for users whose subscription_date
         falls in the holdout window (after fold_4 train_end = 2021-12-01)
         and therefore have NO record in users_attribution_imputed for any
         fold. Without this fallback they would be incorrectly labelled
         'legacy_untracked', silently deflating holdout paid-channel metrics.

    Includes ALL rebill_number=0 users regardless of order_status (refunded
    trials represent completed marketing journeys and must be counted).
    """
    sql = f"""
        WITH
        -- Primary: imputed attribution (fold_4 training window, non-synthetic).
        -- Guaranteed to match downstream fold-scoped INNER JOINs.
        imputed_source AS (
            SELECT DISTINCT user_id, media_source
            FROM `{project}.{dataset}.users_attribution_imputed`
            WHERE fold_id = 'fold_4'
              AND is_synthetic = FALSE
        ),
        -- Fallback: raw attribution only for users NOT covered by imputed above.
        -- Captures holdout-period attributions outside fold_4's training window.
        raw_fallback AS (
            SELECT DISTINCT user_id, media_source
            FROM `{project}.{dataset}.users_attribution`
            WHERE user_id NOT IN (SELECT user_id FROM imputed_source)
        ),
        -- Union: imputed first (no duplicate user_ids), raw fallback second.
        merged_attribution AS (
            SELECT user_id, media_source FROM imputed_source
            UNION ALL
            SELECT user_id, media_source FROM raw_fallback
        )
        SELECT
            p.user_id,
            MIN(p.subscription_date)                       AS subscription_date,
            COALESCE(ma.media_source, 'legacy_untracked')  AS media_source
        FROM `{project}.{dataset}.purchases` AS p
        LEFT JOIN merged_attribution AS ma ON p.user_id = ma.user_id
        WHERE p.rebill_number = 0
        GROUP BY p.user_id, ma.media_source
    """
    logger.info(
        "Extracting base users: hybrid attribution "
        "(imputed fold_4 primary + raw fallback for holdout-period users) …"
    )
    df = client.query(sql).to_dataframe()
    logger.info(
        "Extracted %d users (%d legacy_untracked).",
        len(df), int((df["media_source"] == "legacy_untracked").sum()),
    )
    return df


def _compute_churn_count(
    client: bigquery.Client,
    project: str,
    dataset: str,
    fold_train_end: str | None = None,
) -> int:
    """Derive churn user count from 1st-rebill retention rate (spec §1.3, Note #2).

    conv_rate = n_users_with_rebill_1 / n_users_with_rebill_0
    n_churn   = n_converters × (1 − conv_rate) / conv_rate

    Per-fold: only purchases within the training window (order_date < fold_train_end).
    Global fallback: all purchases if fold_train_end is None.
    """
    date_clause = (
        f"AND CAST(p.order_date AS DATE) < '{fold_train_end}'"
        if fold_train_end else ""
    )

    sql = f"""
        WITH trials AS (
            SELECT DISTINCT p.user_id
            FROM `{project}.{dataset}.purchases` p
            WHERE p.rebill_number = 0  {date_clause}
        ),
        rebillers AS (
            SELECT DISTINCT p.user_id
            FROM `{project}.{dataset}.purchases` p
            WHERE p.rebill_number = 1
              AND p.order_status IN ('approved', 'settled_ok')
              {date_clause}
        )
        SELECT
            (SELECT COUNT(*) FROM trials)    AS n_trials,
            (SELECT COUNT(*) FROM rebillers) AS n_rebills
    """
    row = client.query(sql).to_dataframe().iloc[0]
    n_trials = int(row["n_trials"])
    n_rebills = int(row["n_rebills"])

    if n_trials == 0:
        logger.warning("Zero trial users found; defaulting conv_rate = 0.40")
        conv_rate = 0.40
    else:
        conv_rate = n_rebills / n_trials

    # Safety clamp
    conv_rate = max(0.15, min(0.85, conv_rate))
    churn_count = math.ceil(n_trials * (1.0 - conv_rate) / conv_rate)

    logger.info(
        "Conversion rate = %.4f (n_trials=%d, n_rebills=%d) → %d churn users",
        conv_rate, n_trials, n_rebills, churn_count,
    )
    return churn_count


def _write_to_bq(
    client: bigquery.Client,
    df: pd.DataFrame,
    project: str,
    dataset: str,
) -> None:
    dest = f"{project}.{dataset}.touchpoints_log"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
            bigquery.SchemaField("media_source", "STRING"),
            bigquery.SchemaField("contact_ordinal", "INTEGER"),
            bigquery.SchemaField("is_conversion", "BOOLEAN"),
        ],
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="created_at",
        ),
    )
    client.load_table_from_dataframe(df, dest, job_config=job_config).result()
    logger.info("Uploaded touchpoints_log → %s (%d rows, WRITE_TRUNCATE).", dest, len(df))


def run(
    bq_project: str | None = None,
    bq_dataset: str | None = None,
    fold_id: str | None = None,
    **_kwargs,
) -> None:
    """Entry-point for Airflow PythonOperator or standalone execution."""
    project = bq_project or BQ_PROJECT
    dataset = bq_dataset or BQ_DATASET

    client = bigquery.Client(project=project)

    users_df = _extract_base_users(client, project, dataset)

    # Derive fold_train_end for per-fold conversion rate
    fold_train_end: str | None = None
    if fold_id:
        folds_sql = f"""
            SELECT CAST(train_end AS STRING) AS train_end
            FROM `{project}.{dataset}.rocv_folds`
            WHERE fold_id = '{fold_id}'
        """
        folds_row = client.query(folds_sql).to_dataframe()
        if not folds_row.empty:
            fold_train_end = str(folds_row.iloc[0]["train_end"])

    n_churn = _compute_churn_count(client, project, dataset, fold_train_end)

    full_df = build_touchpoints_log(
        users_df=users_df,
        fold_id=fold_id,
        n_churn=n_churn,
        training_start=TRAINING_START,
        observation_end=OBSERVATION_END,
        seed=SEED,
    )
    _write_to_bq(client, full_df, project, dataset)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run(
        bq_project=os.environ.get("BQ_PROJECT"),
        bq_dataset=os.environ.get("BQ_DATASET"),
        fold_id=os.environ.get("FOLD_ID"),
    )
