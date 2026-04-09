from __future__ import annotations

import pandas as pd
import pytest

from models.attribution import dda_common_new


def test_normalise_paid_weights_drops_zero_spend_channels() -> None:
    weights = {
        "gads:search": 2.0,
        "tiktok": 1.0,
        "organic": 9.0,
        "legacy_untracked": 11.0,
    }

    normalised = dda_common_new.normalise_paid_weights(weights)

    assert set(normalised) == {"gads:search", "tiktok"}
    assert normalised["gads:search"] == pytest.approx(2.0 / 3.0)
    assert normalised["tiktok"] == pytest.approx(1.0 / 3.0)
    assert sum(normalised.values()) == pytest.approx(1.0)


def test_compute_channel_ewma_projection_matches_spec_formula() -> None:
    weekly_paid_conversions = pd.DataFrame(
        {
            "week_start": [
                pd.Timestamp("2021-10-04").date(),
                pd.Timestamp("2021-10-11").date(),
                pd.Timestamp("2021-10-18").date(),
            ],
            "paid_conversions": [100.0, 80.0, 120.0],
        }
    )
    weekly_channel_spend = pd.DataFrame(
        {
            "week_start": [
                pd.Timestamp("2021-10-04").date(),
                pd.Timestamp("2021-10-11").date(),
                pd.Timestamp("2021-10-18").date(),
                pd.Timestamp("2021-10-04").date(),
                pd.Timestamp("2021-10-11").date(),
                pd.Timestamp("2021-10-18").date(),
            ],
            "media_source": [
                "gads:search",
                "gads:search",
                "gads:search",
                "tiktok",
                "tiktok",
                "tiktok",
            ],
            "weekly_spend": [250.0, 240.0, 360.0, 750.0, 720.0, 1080.0],
        }
    )
    weights = {"gads:search": 0.25, "tiktok": 0.75}

    projections = dda_common_new.compute_channel_ewma_projection(
        weekly_paid_conversions=weekly_paid_conversions,
        weekly_channel_spend=weekly_channel_spend,
        normalised_weights=weights,
        alpha=0.3,
    )

    # neutral conversions per channel:
    # week1: 100/2=50, week2: 80/2=40, week3: 120/2=60
    # gads CAC series = [250/50, 240/40, 360/60] = [5, 6, 6]  -> EWMA=5.51
    # tiktok CAC series = [750/50, 720/40, 1080/60] = [15, 18, 18] -> EWMA=16.53
    assert projections["gads:search"] == pytest.approx(5.51, abs=1e-6)
    assert projections["tiktok"] == pytest.approx(16.53, abs=1e-6)


def test_translate_weights_to_forecasts_ewma_is_spend_aware_per_holdout_week(monkeypatch: pytest.MonkeyPatch) -> None:
    weekly_paid_conversions = pd.DataFrame(
        {
            "week_start": [pd.Timestamp("2021-10-04").date(), pd.Timestamp("2021-10-11").date()],
            "paid_conversions": [100.0, 80.0],
        }
    )
    weekly_channel_spend = pd.DataFrame(
        {
            "week_start": [
                pd.Timestamp("2021-10-04").date(),
                pd.Timestamp("2021-10-11").date(),
                pd.Timestamp("2021-10-04").date(),
                pd.Timestamp("2021-10-11").date(),
            ],
            "media_source": ["gads:search", "gads:search", "tiktok", "tiktok"],
            "weekly_spend": [200.0, 160.0, 800.0, 640.0],
        }
    )

    monkeypatch.setattr(
        dda_common_new,
        "fetch_training_weekly_paid_conversions",
        lambda *args, **kwargs: weekly_paid_conversions,
    )
    monkeypatch.setattr(
        dda_common_new,
        "fetch_training_weekly_spend",
        lambda *args, **kwargs: weekly_channel_spend,
    )
    monkeypatch.setattr(
        dda_common_new,
        "fetch_holdout_weekly_spend",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "week_start": [
                    pd.Timestamp("2021-11-29").date(),
                    pd.Timestamp("2021-12-06").date(),
                    pd.Timestamp("2021-11-29").date(),
                    pd.Timestamp("2021-12-06").date(),
                ],
                "media_source": ["gads:search", "gads:search", "tiktok", "tiktok"],
                "weekly_spend": [100.0, 200.0, 100.0, 200.0],
            }
        ),
    )

    df = dda_common_new.translate_weights_to_forecasts_ewma(
        weights={"gads:search": 0.2, "tiktok": 0.8, "organic": 0.1},
        model_name="Markov_DDA",
        client=object(),
        project="p",
        dataset="d",
        fold_id="fold_4",
        train_end="2021-12-01",
        holdout_start="2021-12-01",
        holdout_end="2021-12-10",
        confidence_weight=-1,
    )

    assert len(df) == 2
    assert list(df["forecast_period"]) == [
        pd.Timestamp("2021-11-29").date(),
        pd.Timestamp("2021-12-06").date(),
    ]
    # model_aggregate_cac = 0.2*4 + 0.8*16 = 13.6
    # holdout total spend per week = [200, 400]
    # expected_conversions = spend / 13.6
    assert df["expected_conversions"].tolist() == [14.71, 29.41]
    assert set(df["expected_cac_usd"].tolist()) == {13.6}
    assert set(df["confidence_weight"].tolist()) == {-1}


def test_translate_weights_to_forecasts_ewma_differs_for_different_weight_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dda_common_new,
        "fetch_training_weekly_paid_conversions",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "week_start": [pd.Timestamp("2021-10-04").date(), pd.Timestamp("2021-10-11").date()],
                "paid_conversions": [100.0, 100.0],
            }
        ),
    )
    monkeypatch.setattr(
        dda_common_new,
        "fetch_training_weekly_spend",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "week_start": [
                    pd.Timestamp("2021-10-04").date(),
                    pd.Timestamp("2021-10-11").date(),
                    pd.Timestamp("2021-10-04").date(),
                    pd.Timestamp("2021-10-11").date(),
                ],
                "media_source": ["gads:search", "gads:search", "tiktok", "tiktok"],
                "weekly_spend": [100.0, 100.0, 900.0, 900.0],
            }
        ),
    )
    monkeypatch.setattr(
        dda_common_new,
        "fetch_holdout_weekly_spend",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "week_start": [pd.Timestamp("2021-12-06").date(), pd.Timestamp("2021-12-06").date()],
                "media_source": ["gads:search", "tiktok"],
                "weekly_spend": [500.0, 500.0],
            }
        ),
    )

    df_a = dda_common_new.translate_weights_to_forecasts_ewma(
        weights={"gads:search": 0.5, "tiktok": 0.5},
        model_name="Markov_DDA",
        client=object(),
        project="p",
        dataset="d",
        fold_id="fold_4",
        train_end="2021-12-01",
        holdout_start="2021-12-01",
        holdout_end="2021-12-13",
        confidence_weight=0,
    )
    df_b = dda_common_new.translate_weights_to_forecasts_ewma(
        weights={"gads:search": 0.1, "tiktok": 0.9},
        model_name="Shapley_DDA",
        client=object(),
        project="p",
        dataset="d",
        fold_id="fold_4",
        train_end="2021-12-01",
        holdout_start="2021-12-01",
        holdout_end="2021-12-13",
        confidence_weight=0,
    )

    target_week = pd.Timestamp("2021-12-06").date()
    conv_a = float(df_a.loc[df_a["forecast_period"] == target_week, "expected_conversions"].iloc[0])
    conv_b = float(df_b.loc[df_b["forecast_period"] == target_week, "expected_conversions"].iloc[0])
    cac_a = float(df_a.loc[df_a["forecast_period"] == target_week, "expected_cac_usd"].iloc[0])
    cac_b = float(df_b.loc[df_b["forecast_period"] == target_week, "expected_cac_usd"].iloc[0])

    assert conv_a != conv_b
    assert cac_a != cac_b
