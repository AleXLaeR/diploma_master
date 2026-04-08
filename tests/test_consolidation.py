import re
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent.parent / "sql" / "consolidation"


def test_survival_segment_regex():
    """
    Segment parsing for consolidate_survival.sql must support:
    - ISO week prefixes      (2021-W14_...)
    - Date prefixes          (2021-04-05_...)
    - Legacy week prefixes   (2021_Week-14_...)
    - Full fallback prefixes (ALL_...)
    """
    acq_pattern = re.compile(
        r"^(.+?)_(SUB_WEEKLY|SUB_MONTHLY|SUB_3_MONTH)_.+$"
    )
    product_pattern = re.compile(
        r"(SUB_WEEKLY|SUB_MONTHLY|SUB_3_MONTH)"
    )
    country_pattern = re.compile(
        r"(?:SUB_WEEKLY|SUB_MONTHLY|SUB_3_MONTH)_(.+)$"
    )

    cases = [
        ("2021-W14_SUB_MONTHLY_NA_US", "2021-W14", "SUB_MONTHLY", "NA_US"),
        ("2021-04-05_SUB_3_MONTH_EU_WEST", "2021-04-05", "SUB_3_MONTH", "EU_WEST"),
        ("2021_Week-14_SUB_WEEKLY_ALL", "2021_Week-14", "SUB_WEEKLY", "ALL"),
        ("ALL_SUB_WEEKLY_ALL", "ALL", "SUB_WEEKLY", "ALL"),
        ("ALL_SUB_3_MONTH_ALL", "ALL", "SUB_3_MONTH", "ALL"),
    ]

    for segment, expected_acq, expected_prod, expected_country in cases:
        acq_match = acq_pattern.search(segment)
        assert acq_match is not None, f"Acquisition-key pattern failed on {segment}"
        assert acq_match.group(1) == expected_acq

        # Test product extraction
        prod_match = product_pattern.search(segment)
        assert prod_match is not None, f"Product pattern failed on {segment}"
        assert prod_match.group(1) == expected_prod

        # Test country extraction
        country_match = country_pattern.search(segment)
        assert country_match is not None, f"Country pattern failed on {segment}"
        assert country_match.group(1) == expected_country


def test_consolidation_sql_rendering():
    """
    Test that consolidation SQL scripts exist and can be rendered
    with the target project and dataset.
    """
    project = "test_project"
    dataset = "test_dataset"

    scripts = [
        "consolidate_survival.sql",
        "consolidate_dda.sql",
        "consolidate_mmm.sql",
        "consolidate_channel_contribs.sql",
    ]

    for script_name in scripts:
        sql_path = SQL_DIR / script_name
        assert sql_path.exists(), f"Missing SQL script: {script_name}"

        raw_sql = sql_path.read_text(encoding="utf-8")
        rendered_sql = raw_sql.replace("{{ project }}", project).replace(
            "{{ dataset }}", dataset
        )

        assert project in rendered_sql
        assert dataset in rendered_sql
        assert "{{ project }}" not in rendered_sql
        assert "{{ dataset }}" not in rendered_sql

        # Basic sanity checks on the queries
        assert "MERGE" in rendered_sql
        assert "WHEN MATCHED THEN" in rendered_sql
        if script_name == "consolidate_dda.sql":
            assert "eval_dda" in rendered_sql
            assert "actual_conversions" in rendered_sql
            assert "actual_cac_usd" in rendered_sql
            assert "SAFE_DIVIDE" in rendered_sql
        elif script_name == "consolidate_mmm.sql":
            assert "eval_mmm" in rendered_sql
            assert "actual_net_revenue_usd" in rendered_sql
            assert "users_attribution_imputed" in rendered_sql
            assert "users_attribution` AS ua_raw" in rendered_sql
            assert "COALESCE(fu.actual_net_revenue_usd, 0.0)" in rendered_sql
        elif script_name == "consolidate_channel_contribs.sql":
            assert "mmm_channel_contribs" in rendered_sql
            assert "touchpoints_log" in rendered_sql
            assert "insights_channel_spend" in rendered_sql
            assert "actual_contrib_gads_search" in rendered_sql
            assert "actual_contrib_tiktok" in rendered_sql
            assert "COALESCE(fc.actual_contrib_gads_search, 0.0)" in rendered_sql
        else:
            assert "eval_survival" in rendered_sql
            assert "actual_active_users" in rendered_sql
            assert "actual_ltv_usd" in rendered_sql
            assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in rendered_sql


def test_consolidation_queries_drive_updates_from_eval_target_rows():
    """
    Task #4 requirement: consolidation must update predicted rows even when
    factual holdout values are absent (zero-fill behavior).
    """
    dda_sql = (SQL_DIR / "consolidate_dda.sql").read_text(encoding="utf-8")
    mmm_sql = (SQL_DIR / "consolidate_mmm.sql").read_text(encoding="utf-8")
    survival_sql = (SQL_DIR / "consolidate_survival.sql").read_text(encoding="utf-8")
    contrib_sql = (SQL_DIR / "consolidate_channel_contribs.sql").read_text(encoding="utf-8")

    assert "WITH target_rows AS (" in dda_sql
    assert "FROM `{{ project }}.{{ dataset }}.eval_dda` AS e" in dda_sql
    assert "COALESCE(fc.actual_conversions, 0)" in dda_sql

    assert "WITH target_rows AS (" in mmm_sql
    assert "FROM `{{ project }}.{{ dataset }}.eval_mmm` AS e" in mmm_sql
    assert "COALESCE(fu.actual_net_revenue_usd, 0.0)" in mmm_sql

    assert "WITH target_rows AS (" in survival_sql
    assert "FROM `{{ project }}.{{ dataset }}.eval_survival` AS e" in survival_sql
    assert "COALESCE(" in survival_sql

    assert "WITH target_rows AS (" in contrib_sql
    assert "FROM `{{ project }}.{{ dataset }}.mmm_channel_contribs` AS m" in contrib_sql
    assert "LEFT JOIN factual_contribs AS fc" in contrib_sql
    assert "COALESCE(fc.actual_contrib_gads_search, 0.0)" in contrib_sql


def test_channel_contrib_script_uses_required_inputs():
    """
    Task #5 contract: actual MMM channel contributions must be derived from
    purchases + touchpoints_log, with spend-aware weighting via insights_channel_spend.
    """
    contrib_sql = (SQL_DIR / "consolidate_channel_contribs.sql").read_text(encoding="utf-8")

    assert "`{{ project }}.{{ dataset }}.purchases` AS p" in contrib_sql
    assert "`{{ project }}.{{ dataset }}.touchpoints_log` AS tl" in contrib_sql
    assert "`{{ project }}.{{ dataset }}.insights_channel_spend` AS ics" in contrib_sql
    assert "holdout_net_revenue_usd * ucw.channel_weight" in contrib_sql
    assert "Total_Macro_Global" in contrib_sql
