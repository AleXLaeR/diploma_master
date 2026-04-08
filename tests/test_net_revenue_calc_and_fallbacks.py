from __future__ import annotations

from pathlib import Path

import pandas as pd

from models import dda_common, mmm_common


ROOT = Path(__file__).resolve().parents[1]


def test_build_forecast_row_uses_net_revenue_without_refund_reapplication():
    row = mmm_common.build_forecast_row(
        model_name="Baseline_MMM_Reg",
        forecast_period="2022-01-03",
        country_code="EU_WEST",
        expected_net_revenue=123.456,
        confidence_weight=0,
        fold_id="fold_4",
    )

    assert row["expected_net_revenue_usd"] == 123.46
    assert row["segment"] == "Total_Macro_EU_WEST"


def test_fetch_training_paid_acquisition_stats_uses_attribution_paths(monkeypatch):
    captured_sql: list[str] = []

    def _fake_run_query(_client, sql: str) -> pd.DataFrame:
        captured_sql.append(sql)
        return pd.DataFrame([{"paid_revenue": 200.0, "paid_conversions": 4}])

    monkeypatch.setattr(dda_common, "_run_query", _fake_run_query)

    stats = dda_common.fetch_training_paid_acquisition_stats(
        client=object(),
        project="p",
        dataset="d",
        train_end="2021-12-01",
        fold_id="fold_4",
    )

    sql = captured_sql[0]
    assert "FROM `p.d.attribution_paths` AS ap" in sql
    assert "SUM(ap.conversion_value_usd)" in sql
    assert "order_amount_in_usd - COALESCE(refund_amount_in_usd, 0)" not in sql
    assert stats == {"paid_revenue": 200.0, "paid_conversions": 4, "aov": 50.0}


def test_fetch_attribution_paths_uses_mart_revenue_not_raw_sum(monkeypatch):
    captured_sql: list[str] = []

    def _fake_run_query(_client, sql: str) -> pd.DataFrame:
        captured_sql.append(sql)
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "journey": "tiktok > gads:search",
                    "is_converted": True,
                    "conversion_value_usd": 17.5,
                }
            ]
        )

    monkeypatch.setattr(dda_common, "_run_query", _fake_run_query)

    df = dda_common.fetch_attribution_paths(
        client=object(),
        project="p",
        dataset="d",
        train_end="2021-12-01",
    )

    sql = captured_sql[0]
    assert "`p.d.attribution_paths` AS ap" in sql
    assert "SUM(order_amount_in_usd - COALESCE(refund_amount_in_usd, 0))" not in sql
    assert df.loc[0, "conversion_value_usd"] == 17.5


def test_sql_join_fallbacks_for_task3():
    channel_weights_sql = (
        ROOT / "sql" / "intermediate" / "create_channel_cpc_weights.sql"
    ).read_text(encoding="utf-8")
    insights_spend_sql = (
        ROOT / "sql" / "intermediate" / "create_insights_channel_spend.sql"
    ).read_text(encoding="utf-8")
    consolidate_dda_sql = (
        ROOT / "sql" / "consolidation" / "consolidate_dda.sql"
    ).read_text(encoding="utf-8")

    assert "LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed" in channel_weights_sql
    assert "LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw" in channel_weights_sql
    assert "COALESCE(ua_imputed.country_code, ua_raw.country_code)" in channel_weights_sql
    assert "COALESCE(ua_imputed.media_source, ua_raw.media_source)" in channel_weights_sql

    assert "LEFT JOIN\n        `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed" in insights_spend_sql
    assert "LEFT JOIN\n        `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw" in insights_spend_sql
    assert "COALESCE(ua_imputed.country_code, ua_raw.country_code)" in insights_spend_sql
    assert "COALESCE(ua_imputed.media_source, ua_raw.media_source, tl.media_source)" in insights_spend_sql

    assert "LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution_imputed` AS ua_imputed" in consolidate_dda_sql
    assert "LEFT JOIN `{{ project }}.{{ dataset }}.users_attribution` AS ua_raw" in consolidate_dda_sql
    assert "COALESCE(ua_imputed.media_source, ua_raw.media_source, 'legacy_untracked')" in consolidate_dda_sql
