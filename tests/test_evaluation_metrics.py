from __future__ import annotations

import numpy as np
import pandas as pd

from models.evaluation.metrics_dda import (
    aggregate_dda_wape,
    compute_dda_wape,
    compute_spearman_concordance,
)
from models.evaluation.metrics_mmm import calibration_from_ppc, compute_mmm_fold_metrics
from models.evaluation.metrics_survival import (
    build_ltv_curves_table,
    compute_rmse_by_horizon,
    dedupe_survival_rows,
)


def test_compute_dda_wape_and_aggregate() -> None:
    df = pd.DataFrame(
        {
            "fold_id": ["fold_1", "fold_1"],
            "model_name": ["Shapley_DDA", "Shapley_DDA"],
            "expected_conversions": [11.0, 9.0],
            "actual_conversions": [10.0, 10.0],
            "expected_cac_usd": [100.0, 100.0],
            "actual_cac_usd": [90.0, 110.0],
        }
    )
    fold = compute_dda_wape(df)
    assert len(fold) == 1
    assert float(fold["wape_conversions"].iloc[0]) == 0.1
    agg = aggregate_dda_wape(fold)
    assert agg["n_folds"].iloc[0] == 1


def test_compute_mmm_fold_metrics() -> None:
    df = pd.DataFrame(
        {
            "fold_id": ["fold_1", "fold_1"],
            "model_name": ["Baseline_MMM_Reg", "Baseline_MMM_Reg"],
            "expected_net_revenue_usd": [110.0, 90.0],
            "actual_net_revenue_usd": [100.0, 100.0],
        }
    )
    out = compute_mmm_fold_metrics(df)
    assert len(out) == 1
    assert out["wape"].iloc[0] == 0.1
    assert out["wbias"].iloc[0] == 0.0


def test_calibration_from_ppc() -> None:
    ppc_by_fold = {
        "fold_4": np.array(
            [
                [90.0, 100.0, 110.0],
                [95.0, 100.0, 105.0],
                [98.0, 100.0, 102.0],
            ]
        )
    }
    actual_by_fold = {"fold_4": np.array([97.0, 100.0, 104.0])}
    out = calibration_from_ppc(ppc_by_fold, actual_by_fold, nominal_levels=[0.5, 0.9])
    assert {"fold_id", "nominal_coverage", "empirical_coverage"} <= set(out.columns)
    assert (out["fold_id"] == "aggregate").any()


def test_survival_dedupe_rmse_and_ltv_filter() -> None:
    df = pd.DataFrame(
        {
            "fold_id": ["fold_4"] * 8,
            "model_name": [
                "Baseline_Survival",
                "Baseline_Survival",
                "Baseline_Survival",
                "Baseline_Survival",
                "BdW",
                "BdW",
                "BdW",
                "BdW",
            ],
            "segment": ["s1", "s1", "s1", "s1", "s1", "s1", "s1", "s1"],
            "rebill_period_t": [1, 1, 2, 3, 1, 1, 2, 3],
            "forecast_period": pd.to_datetime(
                [
                    "2021-12-06",
                    "2021-12-13",
                    "2021-12-20",
                    "2021-12-27",
                    "2021-12-06",
                    "2021-12-13",
                    "2021-12-20",
                    "2021-12-27",
                ]
            ),
            "expected_active_users": [100, 100, 80, 70, 95, 95, 82, 72],
            "actual_active_users": [100, 100, 78, 68, 100, 100, 78, 68],
            "expected_ltv_usd": [10, 10, 18, 24, 10, 10, 17, 23],
            "actual_ltv_usd": [10, 10, 17, 22, 10, 10, 17, 22],
        }
    )
    dedup = dedupe_survival_rows(df)
    # repeated t=1 rows should be deduped per model+segment
    assert len(dedup[dedup["rebill_period_t"] == 1]) == 2

    rmse = compute_rmse_by_horizon(dedup)
    assert not rmse.empty
    assert {"rmse", "cohort_count"} <= set(rmse.columns)

    ltv = build_ltv_curves_table(dedup)
    assert not ltv.empty
    assert (ltv["rebill_period_t"] >= 1).all()


def test_compute_spearman_concordance() -> None:
    dda = pd.DataFrame(
        {
            "fold_id": ["fold_4"] * 12,
            "model_name": ["Baseline_LastClick"] * 6 + ["Shapley_DDA"] * 6,
            "media_source": [
                "gads:search",
                "gads:youtube",
                "gads:discover",
                "metads:inst",
                "metads:fb",
                "tiktok",
            ]
            * 2,
            "weight": [0.3, 0.2, 0.1, 0.15, 0.15, 0.1, 0.32, 0.18, 0.09, 0.16, 0.15, 0.1],
        }
    )
    mmm = pd.DataFrame(
        {
            "fold_id": ["fold_4"],
            "model_name": ["MMM_BSTS_DDA"],
            "segment": ["Total_Macro_Global"],
            "incr_gads_search": [0.33],
            "incr_gads_youtube": [0.18],
            "incr_gads_discover": [0.1],
            "incr_metads_inst": [0.15],
            "incr_metads_fb": [0.14],
            "incr_tiktok": [0.1],
        }
    )
    out = compute_spearman_concordance(dda, mmm)
    assert not out.empty
    assert {"fold_id", "comparison", "rho"} <= set(out.columns)

