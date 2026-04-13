from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.evaluation.figures_f1_f3 import (
    render_f1_attribution_shift,
    render_f2_markov_transition_heatmap,
    render_f3_posterior_kde_vs_ols,
)
from models.evaluation.figures_f4_f6 import (
    render_f4_wape_wbias_dual_panel,
    render_f5_calibration_coverage,
    render_f6_posterior_timeseries,
)
from models.evaluation.figures_f7_f9 import (
    render_f7_survival_decay_curves,
    render_f8_rmse_divergence,
    render_f9_ltv_small_multiples,
)
from models.evaluation.metrics_survival import dedupe_survival_rows


class _PosteriorVar:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape


class _FakeTrace:
    def __init__(self, beta_values: np.ndarray) -> None:
        self.posterior = {"beta": _PosteriorVar(beta_values)}


def test_all_figure_renderers_smoke(tmp_path: Path) -> None:
    # F1
    dda_weights = pd.DataFrame(
        {
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
            "weight": [0.3, 0.2, 0.1, 0.15, 0.15, 0.1, 0.25, 0.2, 0.12, 0.16, 0.15, 0.12],
        }
    )
    shap_boot = pd.DataFrame(
        {
            "media_source": [
                "gads:search",
                "gads:youtube",
                "gads:discover",
                "metads:inst",
                "metads:fb",
                "tiktok",
            ],
            "weight_std": [0.02, 0.03, 0.01, 0.02, 0.02, 0.02],
        }
    )
    render_f1_attribution_shift(tmp_path / "F1.png", dda_weights, shap_boot)

    # F2
    paths = pd.DataFrame(
        {
            "journey": [
                "tiktok > metads:fb > gads:search",
                "metads:inst > gads:youtube",
                "gads:discover > metads:fb",
            ],
            "is_converted": [True, False, True],
        }
    )
    render_f2_markov_transition_heatmap(tmp_path / "F2.png", paths)

    # F3
    trace = _FakeTrace(np.random.normal(size=(2, 60, 6)))
    ols = pd.DataFrame(
        {
            "regressor": [
                "spend_gads_search",
                "spend_gads_youtube",
                "spend_gads_discover",
                "spend_metads_inst",
                "spend_metads_fb",
                "spend_tiktok",
            ],
            "coefficient": [0.2, 0.1, 0.05, 0.09, 0.08, 0.07],
            "fit_region": ["Global"] * 6,
            "segment": ["Total_Macro_Global"] * 6,
        }
    )
    render_f3_posterior_kde_vs_ols(tmp_path / "F3.png", trace, ols)

    # F4
    mmm_metrics = pd.DataFrame(
        {
            "fold_id": ["fold_1", "fold_2", "fold_1", "fold_2"],
            "model_name": ["Baseline_MMM_Reg", "Baseline_MMM_Reg", "MMM_BSTS_DDA", "MMM_BSTS_DDA"],
            "wape": [0.2, 0.3, 0.1, 0.11],
            "wbias": [0.05, -0.04, 0.01, -0.01],
        }
    )
    render_f4_wape_wbias_dual_panel(tmp_path / "F4.png", mmm_metrics)

    # F5
    calibration = pd.DataFrame(
        {
            "fold_id": ["fold_4", "fold_4", "aggregate", "aggregate"],
            "nominal_coverage": [0.5, 0.9, 0.5, 0.9],
            "empirical_coverage": [0.45, 0.85, 0.48, 0.88],
        }
    )
    render_f5_calibration_coverage(tmp_path / "F5.png", calibration)

    # F6
    train = pd.DataFrame(
        {"date_week": pd.to_datetime(["2021-11-01", "2021-11-08"]), "actual_net_revenue_usd": [100, 110]}
    )
    holdout = pd.DataFrame(
        {"date_week": pd.to_datetime(["2021-12-06", "2021-12-13", "2021-12-20"]), "actual_net_revenue_usd": [120, 130, 125]}
    )
    ppc = np.random.normal(loc=np.array([121, 129, 126]), scale=2.0, size=(100, 3))
    render_f6_posterior_timeseries(tmp_path / "F6.png", train, holdout, ppc, "2021-12-01")

    # Survival input (F7-F9)
    eval_survival = pd.DataFrame(
        {
            "fold_id": ["fold_4"] * 12,
            "model_name": ["Baseline_Survival"] * 6 + ["BdW"] * 6,
            "segment": ["cohort_a"] * 6 + ["cohort_a"] * 6,
            "rebill_period_t": [1, 1, 2, 3, 4, 5, 1, 1, 2, 3, 4, 5],
            "forecast_period": pd.to_datetime(
                [
                    "2021-12-06",
                    "2021-12-13",
                    "2021-12-20",
                    "2021-12-27",
                    "2022-01-03",
                    "2022-01-10",
                ]
                * 2
            ),
            "expected_active_users": [100, 100, 80, 70, 64, 58, 98, 98, 82, 73, 66, 60],
            "actual_active_users": [100, 100, 79, 71, 65, 59, 100, 100, 79, 71, 65, 59],
            "expected_ltv_usd": [10, 10, 18, 25, 31, 36, 10, 10, 17, 24, 30, 35],
            "actual_ltv_usd": [10, 10, 17, 23, 28, 33, 10, 10, 17, 23, 28, 33],
        }
    )
    dedup = dedupe_survival_rows(eval_survival)
    render_f7_survival_decay_curves(tmp_path / "F7.png", dedup)

    rmse = pd.DataFrame(
        {
            "fold_id": ["fold_4"] * 6,
            "model_name": ["Baseline_Survival"] * 3 + ["BdW"] * 3,
            "rebill_period_t": [1, 2, 3, 1, 2, 3],
            "rmse": [5.0, 9.0, 13.0, 4.0, 6.0, 8.0],
            "cohort_count": [8, 7, 6, 8, 7, 6],
        }
    )
    render_f8_rmse_divergence(tmp_path / "F8.png", rmse)
    render_f9_ltv_small_multiples(tmp_path / "F9.png", dedup)

    for figure_id in range(1, 10):
        assert (tmp_path / f"F{figure_id}.png").exists()
