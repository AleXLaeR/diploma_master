"""
tests/test_generate_touchpoints.py
===================================
Automated tests for `scripts.generate_touchpoints`.

All tests use the pure-logic `build_touchpoints_log()` function
and run entirely in-memory — no BigQuery dependency.

Churn users are identified by the *absence* of any is_conversion=True
row (NOT by ID prefix — churn IDs are proper UUIDs per spec §1.3).
"""

from __future__ import annotations

import uuid as _uuid_module

import numpy as np
import pandas as pd
import pytest

from scripts.generate_touchpoints import (
    BOTTOM_FUNNEL,
    MID_FUNNEL,
    TOP_FUNNEL,
    TRAINING_START,
    OBSERVATION_END,
    build_touchpoints_log,
    _fold_path_weights,
    _generate_timestamps,
    _make_uuid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_users(
    n_organic: int = 50,
    n_facebook: int = 50,
    n_legacy: int = 10,
    sub_date: str = "2021-08-15T12:00:00Z",
) -> pd.DataFrame:
    """Build a minimal users DataFrame for testing."""
    rows = []
    ts = pd.Timestamp(sub_date, tz="UTC")
    for i in range(n_organic):
        rows.append({"user_id": f"org_{i}", "subscription_date": ts, "media_source": "organic"})
    for i in range(n_facebook):
        rows.append({"user_id": f"fb_{i}", "subscription_date": ts, "media_source": "facebook"})
    for i in range(n_legacy):
        rows.append({"user_id": f"lu_{i}", "subscription_date": ts, "media_source": "legacy_untracked"})
    return pd.DataFrame(rows)


EXPECTED_COLS = {"user_id", "created_at", "media_source", "contact_ordinal", "is_conversion"}


def _converter_ids(df: pd.DataFrame) -> set[str]:
    """Users who have at least one is_conversion=True row."""
    return set(df.loc[df["is_conversion"], "user_id"].unique())


def _churn_ids(df: pd.DataFrame) -> set[str]:
    """Users who have NO is_conversion=True row."""
    all_users = set(df["user_id"].unique())
    return all_users - _converter_ids(df)


# ---------------------------------------------------------------------------
# Test 1: Schema
# ---------------------------------------------------------------------------

def test_schema_columns() -> None:
    users = _make_users(n_organic=5, n_facebook=5, n_legacy=2)
    df = build_touchpoints_log(users, fold_id="fold_1", n_churn=10, seed=0)
    assert set(df.columns) == EXPECTED_COLS


# ---------------------------------------------------------------------------
# Test 2: Legacy-untracked → exactly 1 touchpoint, is_conversion=True
# ---------------------------------------------------------------------------

def test_legacy_untracked_single_touch() -> None:
    users = _make_users(n_organic=0, n_facebook=0, n_legacy=20)
    df = build_touchpoints_log(users, fold_id="fold_1", n_churn=0, seed=1)

    assert len(df) == 20
    assert (df["media_source"] == "legacy_untracked").all()
    assert (df["is_conversion"]).all()
    assert (df["contact_ordinal"] == 1).all()


# ---------------------------------------------------------------------------
# Test 3: Every converter has exactly 1 is_conversion=True
# ---------------------------------------------------------------------------

def test_conversion_flag_uniqueness() -> None:
    users = _make_users(n_organic=100, n_facebook=100, n_legacy=20)
    df = build_touchpoints_log(users, fold_id="fold_2", n_churn=50, seed=2)

    conv_counts = df.loc[df["is_conversion"]].groupby("user_id").size()
    multi = conv_counts[conv_counts > 1]
    assert multi.empty, f"Users with >1 conversion: {multi.to_dict()}"


# ---------------------------------------------------------------------------
# Test 4: All churn users have is_conversion=False on every row
# ---------------------------------------------------------------------------

def test_churn_users_no_conversion() -> None:
    users = _make_users(n_organic=20, n_facebook=20, n_legacy=0)
    df = build_touchpoints_log(users, fold_id="fold_1", n_churn=100, seed=3)

    churn = _churn_ids(df)
    churn_df = df[df["user_id"].isin(churn)]
    assert (churn_df["is_conversion"] == False).all()


# ---------------------------------------------------------------------------
# Test 5: Churn user count matches n_churn
# ---------------------------------------------------------------------------

def test_churn_user_count() -> None:
    users = _make_users(n_organic=5, n_facebook=5, n_legacy=0)
    n_churn = 200
    df = build_touchpoints_log(users, fold_id="fold_1", n_churn=n_churn, seed=4)

    assert len(_churn_ids(df)) == n_churn


# ---------------------------------------------------------------------------
# Test 6: Churn user IDs are valid UUIDs (spec §1.3: user_id::UUID)
# ---------------------------------------------------------------------------

def test_churn_ids_are_uuids() -> None:
    users = _make_users(n_organic=5, n_facebook=5, n_legacy=0)
    df = build_touchpoints_log(users, fold_id="fold_1", n_churn=50, seed=5)

    churn = _churn_ids(df)
    for uid in churn:
        try:
            _uuid_module.UUID(uid)
        except ValueError:
            pytest.fail(f"Churn user_id is not a valid UUID: {uid!r}")


# ---------------------------------------------------------------------------
# Test 7: All timestamps within valid bounds
# ---------------------------------------------------------------------------

def test_timestamp_bounds() -> None:
    users = _make_users(n_organic=100, n_facebook=100, n_legacy=20)
    df = build_touchpoints_log(users, fold_id="fold_4", n_churn=200, seed=6)

    assert df["created_at"].dt.tz is not None
    floor = TRAINING_START
    ceiling = OBSERVATION_END + pd.Timedelta(days=1)

    too_early = df[df["created_at"] < floor]
    too_late = df[df["created_at"] > ceiling]
    assert too_early.empty, f"{len(too_early)} rows before {floor}"
    assert too_late.empty, f"{len(too_late)} rows after {ceiling}"


# ---------------------------------------------------------------------------
# Test 8: T₋₁ gap is 60–10800 minutes (spec §1.2)
# ---------------------------------------------------------------------------

def test_time_decay_t_minus_1_range() -> None:
    users = _make_users(n_organic=0, n_facebook=500, n_legacy=0)
    df = build_touchpoints_log(users, fold_id="fold_2", n_churn=0, seed=7)

    converters = _converter_ids(df)
    conv_df = df[df["user_id"].isin(converters)]
    user_max_ord = conv_df.groupby("user_id")["contact_ordinal"].max()
    multi_uids = user_max_ord[user_max_ord >= 2].index

    gaps_min: list[float] = []
    for uid in multi_uids:
        g = conv_df[conv_df["user_id"] == uid].sort_values("contact_ordinal")
        max_ord = g["contact_ordinal"].max()
        t_final = g[g["contact_ordinal"] == max_ord]["created_at"].iloc[0]
        t_prev = g[g["contact_ordinal"] == max_ord - 1]["created_at"].iloc[0]
        gaps_min.append((t_final - t_prev).total_seconds() / 60.0)

    s = pd.Series(gaps_min)
    violations = s[(s < 60) | (s > 10800)]
    assert violations.empty, (
        f"{len(violations)} gaps outside 60–10800 min. "
        f"min={s.min():.1f}, max={s.max():.1f}"
    )


# ---------------------------------------------------------------------------
# Test 9: Path length distribution for known-source users (spec §1.2)
# ---------------------------------------------------------------------------

def test_path_length_distribution() -> None:
    n = 5_000
    rows = [
        {"user_id": f"fb_{i}", "subscription_date": pd.Timestamp("2021-08-15", tz="UTC"),
         "media_source": "facebook"}
        for i in range(n)
    ]
    users = pd.DataFrame(rows)
    df = build_touchpoints_log(users, fold_id=None, n_churn=0, seed=8)

    path_lengths = df.groupby("user_id")["contact_ordinal"].max().clip(upper=4)
    proportions = path_lengths.value_counts(normalize=True)

    targets = {1: 0.35, 2: 0.30, 3: 0.25, 4: 0.10}
    tol = 0.10
    for length, target in targets.items():
        obs = float(proportions.get(length, 0.0))
        assert abs(obs - target) <= tol, (
            f"Path len {length}: expected ~{target:.0%}, got {obs:.0%}"
        )


# ---------------------------------------------------------------------------
# Test 10: organic final-touch → Bottom ~82% / Mid ~18%
# ---------------------------------------------------------------------------

def test_organic_final_touch_distribution() -> None:
    n = 3_000
    rows = [
        {"user_id": f"org_{i}", "subscription_date": pd.Timestamp("2021-08-15", tz="UTC"),
         "media_source": "organic"}
        for i in range(n)
    ]
    users = pd.DataFrame(rows)
    df = build_touchpoints_log(users, fold_id=None, n_churn=0, seed=9)

    final = df[df["is_conversion"]]
    bottom_frac = final["media_source"].isin(BOTTOM_FUNNEL).mean()
    mid_frac = final["media_source"].isin(MID_FUNNEL).mean()

    assert 0.74 <= bottom_frac <= 0.89, f"organic Bottom: {bottom_frac:.2%}"
    assert 0.11 <= mid_frac <= 0.26, f"organic Mid: {mid_frac:.2%}"


# ---------------------------------------------------------------------------
# Test 11: facebook final-touch → Mid ~51% / Top ~39% / Bottom ~10%
# ---------------------------------------------------------------------------

def test_facebook_final_touch_distribution() -> None:
    n = 3_000
    rows = [
        {"user_id": f"fb_{i}", "subscription_date": pd.Timestamp("2021-08-15", tz="UTC"),
         "media_source": "facebook"}
        for i in range(n)
    ]
    users = pd.DataFrame(rows)
    df = build_touchpoints_log(users, fold_id=None, n_churn=0, seed=10)

    final = df[df["is_conversion"]]
    mid_frac = final["media_source"].isin(MID_FUNNEL).mean()
    top_frac = final["media_source"].isin(TOP_FUNNEL).mean()
    bot_frac = final["media_source"].isin(BOTTOM_FUNNEL).mean()

    assert 0.41 <= mid_frac <= 0.61, f"facebook Mid: {mid_frac:.2%}"
    assert 0.29 <= top_frac <= 0.49, f"facebook Top: {top_frac:.2%}"
    assert 0.02 <= bot_frac <= 0.20, f"facebook Bottom: {bot_frac:.2%}"


# ---------------------------------------------------------------------------
# Test 12: Churn paths ≥82% Top+Mid (spec §1.3: 12% reach Bottom)
# ---------------------------------------------------------------------------

def test_churn_mostly_top_mid_funnel() -> None:
    users = pd.DataFrame([
        {"user_id": "u1", "subscription_date": pd.Timestamp("2021-08-15", tz="UTC"),
         "media_source": "organic"}
    ])
    df = build_touchpoints_log(users, fold_id=None, n_churn=1000, seed=11)

    churn = _churn_ids(df)
    churn_df = df[df["user_id"].isin(churn)]
    non_bottom = set(TOP_FUNNEL + MID_FUNNEL)
    frac = churn_df["media_source"].isin(non_bottom).mean()
    assert frac >= 0.82, f"Churn Top+Mid fraction too low: {frac:.2%}"


# ---------------------------------------------------------------------------
# Test 13: Fold path weights differ across folds
# ---------------------------------------------------------------------------

def test_fold_path_weight_drift() -> None:
    weights = {}
    for fid in ["fold_1", "fold_2", "fold_3", "fold_4"]:
        weights[fid] = _fold_path_weights(fid, np.random.default_rng(42))

    assert weights["fold_1"][1] > weights["fold_4"][1], \
        "fold_1 should have more single-touch than fold_4"
    assert weights["fold_4"][4] > weights["fold_1"][4], \
        "fold_4 should have more 4+ touch than fold_1"


# ---------------------------------------------------------------------------
# Test 14: contact_ordinal is [1, 2, …, N] per user
# ---------------------------------------------------------------------------

def test_contact_ordinal_sequential() -> None:
    users = _make_users(n_organic=50, n_facebook=50, n_legacy=10)
    df = build_touchpoints_log(users, fold_id="fold_3", n_churn=30, seed=12)

    for uid, group in df.groupby("user_id"):
        ordinals = sorted(group["contact_ordinal"].tolist())
        assert ordinals == list(range(1, len(ordinals) + 1)), \
            f"User {uid}: non-sequential ordinals {ordinals}"


# ---------------------------------------------------------------------------
# Test 15: Timestamp clamping for early subscription dates
# ---------------------------------------------------------------------------

def test_timestamp_clamping() -> None:
    sub_date = TRAINING_START + pd.Timedelta(hours=2)
    rng = np.random.default_rng(0)
    timestamps = _generate_timestamps(rng, sub_date, path_len=6, training_start=TRAINING_START)
    floor = TRAINING_START + pd.Timedelta(minutes=1)
    for ts in timestamps:
        assert ts >= floor, f"Timestamp {ts} breaches floor {floor}"


# ---------------------------------------------------------------------------
# Test 16: _make_uuid produces valid UUID v4
# ---------------------------------------------------------------------------

def test_make_uuid_is_valid() -> None:
    rng = np.random.default_rng(99)
    for _ in range(100):
        uid = _make_uuid(rng)
        parsed = _uuid_module.UUID(uid)
        assert parsed.version == 4


# ---------------------------------------------------------------------------
# Test 17: Total converter count equals input user count
# ---------------------------------------------------------------------------

def test_converter_count_matches_input() -> None:
    n_org, n_fb, n_lu = 80, 120, 15
    users = _make_users(n_organic=n_org, n_facebook=n_fb, n_legacy=n_lu)
    df = build_touchpoints_log(users, fold_id=None, n_churn=50, seed=13)

    n_converters = len(_converter_ids(df))
    assert n_converters == n_org + n_fb + n_lu, \
        f"Expected {n_org + n_fb + n_lu} converters, got {n_converters}"
