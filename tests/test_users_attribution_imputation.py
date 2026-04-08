from __future__ import annotations

import pandas as pd

from scripts.users_attribution_imputation import build_imputed_users_attribution


def _rows(country: str, media: str, n: int, start: str) -> list[dict[str, object]]:
    base = pd.Timestamp(start, tz="UTC")
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "user_id": f"{country}_{media}_{i}",
                "created_at": base + pd.Timedelta(hours=i),
                "country_code": country,
                "media_source": media,
            }
        )
    return rows


def test_density_floor_and_volume_cap() -> None:
    users = pd.DataFrame(
        _rows("AA", "organic", 6, "2021-01-01T00:00:00Z")
        + _rows("AA", "facebook", 2, "2021-01-02T00:00:00Z")
        + _rows("CC", "organic", 2, "2021-01-03T00:00:00Z")
        + _rows("CC", "facebook", 2, "2021-01-04T00:00:00Z")
        + _rows("BB", "organic", 1, "2021-01-10T12:00:00Z")
    )
    countries = pd.DataFrame(
        [
            {"country_code": "AA", "region": "R1"},
            {"country_code": "BB", "region": "R1"},
            {"country_code": "CC", "region": "R1"},
        ]
    )
    folds = pd.DataFrame([{"fold_id": "fold_1", "train_end": "2021-02-01"}])

    out = build_imputed_users_attribution(users, countries, folds, seed=7)
    fold = out[out["fold_id"] == "fold_1"]

    bb_rows = fold[fold["country_code"] == "BB"]
    # R1 median country volume = median([8,4,1]) = 4; BB should be lifted to 4.
    assert len(bb_rows) == 4
    assert int(bb_rows["is_synthetic"].sum()) == 3
    assert set(bb_rows["media_source"]) == {"organic"}

    bb_observed_ts = set(users[users["country_code"] == "BB"]["created_at"])
    bb_synth_ts = set(bb_rows[bb_rows["is_synthetic"]]["created_at"])
    assert bb_synth_ts.issubset(bb_observed_ts)


def test_fold_training_window_anti_leakage() -> None:
    users = pd.DataFrame(
        _rows("AA", "organic", 6, "2021-01-01T00:00:00Z")
        + _rows("CC", "organic", 4, "2021-01-03T00:00:00Z")
        + _rows("BB", "organic", 1, "2021-01-10T12:00:00Z")
        + _rows("BB", "organic", 5, "2021-03-10T12:00:00Z")
    )
    countries = pd.DataFrame(
        [
            {"country_code": "AA", "region": "R1"},
            {"country_code": "BB", "region": "R1"},
            {"country_code": "CC", "region": "R1"},
        ]
    )
    folds = pd.DataFrame(
        [
            {"fold_id": "fold_1", "train_end": "2021-02-01"},
            {"fold_id": "fold_2", "train_end": "2021-04-01"},
        ]
    )

    out = build_imputed_users_attribution(users, countries, folds, seed=9)
    fold_1 = out[out["fold_id"] == "fold_1"]
    fold_2 = out[out["fold_id"] == "fold_2"]

    assert (fold_1["created_at"].dt.date < pd.Timestamp("2021-02-01").date()).all()
    assert (fold_2["created_at"].dt.date < pd.Timestamp("2021-04-01").date()).all()

    # Late-March rows should only exist in fold_2.
    assert not (fold_1["created_at"] >= pd.Timestamp("2021-03-01", tz="UTC")).any()
    assert (fold_2["created_at"] >= pd.Timestamp("2021-03-01", tz="UTC")).any()


def test_eligibility_and_media_distribution_preserved() -> None:
    users = pd.DataFrame(
        _rows("AA", "organic", 5, "2021-01-01T00:00:00Z")
        + _rows("AA", "facebook", 5, "2021-01-02T00:00:00Z")
        + _rows("CC", "organic", 3, "2021-01-03T00:00:00Z")
        + _rows("CC", "facebook", 3, "2021-01-04T00:00:00Z")
        + _rows("BB", "organic", 1, "2021-01-10T12:00:00Z")
        + _rows("BB", "facebook", 1, "2021-01-10T15:00:00Z")
    )
    countries = pd.DataFrame(
        [
            {"country_code": "AA", "region": "R1"},
            {"country_code": "BB", "region": "R1"},
            {"country_code": "CC", "region": "R1"},
        ]
    )
    folds = pd.DataFrame([{"fold_id": "fold_1", "train_end": "2021-02-01"}])

    out = build_imputed_users_attribution(users, countries, folds, seed=11)
    bb_rows = out[out["country_code"] == "BB"]
    bb_synth = bb_rows[bb_rows["is_synthetic"]]

    # Median([10,6,2]) = 6 => BB receives +4 synthetic rows.
    assert len(bb_rows) == 6
    assert len(bb_synth) == 4
    assert set(bb_synth["media_source"]) == {"organic", "facebook"}
    media_counts = bb_synth.groupby("media_source").size().to_dict()
    assert abs(media_counts["organic"] - media_counts["facebook"]) <= 1

