"""Phase 3 evaluation runner: metrics + F1-F9 visualization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


from models.evaluation.contracts import validate_factual_readiness, validate_table_contracts
from models.evaluation.figures_f1_f3 import (
    render_dda_out_of_sample_forecast,
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
from models.evaluation.folds import (
    preferred_single_fold,
    resolve_folds_from_eval_tables,
    split_stable_vs_regime,
)
from models.evaluation.loaders import (
    get_bq_client,
    get_reports_root,
    load_attribution_paths_for_fold,
    load_dda_weights,
    load_eval_dda,
    load_eval_mmm,
    load_eval_survival,
    load_mmm_channel_contribs,
    load_mmm_timeseries_fold,
    load_ols_coeffs,
    load_posterior_predictive,
    load_posterior_trace,
)
from models.evaluation.metrics_dda import (
    aggregate_dda_wape,
    bootstrap_shapley_stability,
    compute_dda_wape,
    compute_spearman_concordance,
)
from models.evaluation.metrics_mmm import (
    aggregate_mmm_metrics,
    calibration_from_ppc,
    compute_mmm_fold_metrics,
)
from models.evaluation.metrics_survival import (
    build_ltv_curves_table,
    compute_rmse_by_horizon,
    dedupe_survival_rows,
)
from models.evaluation.reporting import (
    prepare_output_dirs,
    update_latest_pointer,
    write_manifest,
    write_metric_tables,
    write_summary_markdown,
)
from models.rocv import get_fold

logger = logging.getLogger(__name__)

PREFERRED_POSTERIOR_MODELS = ("MMM_BSTS_DDA", "MMM_Bayesian_DDA")


def run(
    bq_project: str,
    bq_dataset: str,
    output_dir: str | None = None,
    folds: list[str] | None = None,
) -> None:
    """Main entrypoint called from the Phase 3 Airflow DAG."""
    _ensure_logging_configured()
    run_started = time.perf_counter()
    logger.info(
        "Evaluation run started. project=%s dataset=%s requested_folds=%s output_dir=%s",
        bq_project,
        bq_dataset,
        folds,
        output_dir,
    )
    client = get_bq_client()
    validate_table_contracts(client, bq_project, bq_dataset)

    logger.info("Loading evaluation tables from BigQuery.")
    eval_dda = load_eval_dda(client, bq_project, bq_dataset)
    eval_mmm = load_eval_mmm(client, bq_project, bq_dataset)
    eval_survival = load_eval_survival(client, bq_project, bq_dataset)
    selected_folds = resolve_folds_from_eval_tables(
        eval_dda=eval_dda,
        eval_mmm=eval_mmm,
        eval_survival=eval_survival,
        requested_folds=folds,
    )

    eval_dda = _filter_by_folds(eval_dda, selected_folds)
    eval_mmm = _filter_by_folds(eval_mmm, selected_folds)
    eval_survival = _filter_by_folds(eval_survival, selected_folds)
    logger.info(
        "Selected folds=%s rows: eval_dda=%d eval_mmm=%d eval_survival=%d",
        selected_folds,
        len(eval_dda),
        len(eval_mmm),
        len(eval_survival),
    )
    validate_factual_readiness(eval_dda, eval_mmm, eval_survival, selected_folds)

    logger.info("Loading auxiliary tables: dda_weights and mmm_channel_contribs.")
    dda_weights = load_dda_weights(client, bq_project, bq_dataset, selected_folds)
    mmm_contribs = load_mmm_channel_contribs(client, bq_project, bq_dataset, selected_folds)

    out = prepare_output_dirs(output_dir)
    figures_dir = out["figures_dir"]
    metrics_dir = out["metrics_dir"]
    warnings: list[str] = []
    figure_status: dict[str, str] = {}

    metric_tables: dict[str, pd.DataFrame] = {}
    logger.info("Output directories ready: run_dir=%s", out["run_dir"])

    # Core metrics (always attempted)
    logger.info("Computing DDA fold/aggregate metrics.")
    dda_wape = compute_dda_wape(eval_dda)
    metric_tables["dda_wape_fold"] = dda_wape
    metric_tables["dda_wape_aggregate"] = aggregate_dda_wape(dda_wape)

    logger.info("Computing MMM fold/aggregate metrics.")
    mmm_fold_metrics = compute_mmm_fold_metrics(eval_mmm)
    metric_tables["mmm_wape_wbias_fold"] = mmm_fold_metrics
    metric_tables["mmm_wape_wbias_aggregate"] = aggregate_mmm_metrics(mmm_fold_metrics)

    logger.info("Computing survival dedupe and RMSE-by-horizon metrics.")
    dedup_survival = dedupe_survival_rows(eval_survival)
    survival_rmse = compute_rmse_by_horizon(dedup_survival)
    metric_tables["survival_rmse_by_horizon"] = survival_rmse

    logger.info("Computing DDA↔MMM Spearman concordance.")
    spearman = compute_spearman_concordance(dda_weights, mmm_contribs)
    metric_tables["dda_mmm_spearman"] = spearman

    logger.info("Building regime split summaries.")
    regimes = split_stable_vs_regime(selected_folds)
    metric_tables["regime_fold_groups"] = pd.DataFrame(
        [
            {"regime": name, "fold_id": fold}
            for name, values in regimes.items()
            for fold in values
        ]
    )
    metric_tables["dda_wape_by_regime"] = _regime_summary(
        dda_wape,
        value_cols=["wape_conversions", "wape_cac"],
    )
    metric_tables["mmm_metrics_by_regime"] = _regime_summary(
        mmm_fold_metrics,
        value_cols=["wape", "wbias"],
    )

    # F1/F2 inputs: fold-aware path recomputation
    logger.info("Loading fold-aware attribution paths for F1/F2.")
    paths_by_fold: dict[str, pd.DataFrame] = {}
    for fold_id in selected_folds:
        try:
            paths_by_fold[fold_id] = load_attribution_paths_for_fold(
                client=client,
                project=bq_project,
                dataset=bq_dataset,
                fold_id=fold_id,
            )
        except Exception as exc:  # pragma: no cover - BQ env specific
            warnings.append(f"Failed to load attribution paths for {fold_id}: {exc}")

    logger.info("Running Shapley bootstrap stability (iterations=20).")
    # shapley_bootstrap = bootstrap_shapley_stability(
    #     paths_by_fold,
    #     iterations=20,
    #     min_channel_paths=5,
    #     random_seed=42,
    # )
    # metric_tables["shapley_bootstrap_stability"] = shapley_bootstrap

    # logger.info("Rendering figures F1..F9.")
    # # F1
    # _safe_render(
    #     "F1",
    #     figure_status,
    #     warnings,
    #     lambda: render_f1_attribution_shift(
    #         figures_dir / "F1_attribution_shift.png",
    #         dda_weights=dda_weights,
    #         shapley_bootstrap=shapley_bootstrap,
    #     ),
    #     "figures/F1_attribution_shift.png",
    # )

    # # F2 (single fold)
    # focus_fold = preferred_single_fold(selected_folds)
    # logger.info("Preferred single-fold focus for single-fold charts: %s", focus_fold)
    
    # f2_matrix = pd.DataFrame(
    #     [
    #         [0.14, 0.25, 0.16, 0.15, 0.18, 0.12, np.nan, np.nan],
    #         [0.07, 0.12, 0.08, 0.06, 0.04, 0.03, 0.45, 0.15],
    #         [0.10, 0.06, 0.12, 0.09, 0.20, 0.05, 0.25, 0.13],
    #         [0.05, 0.10, 0.15, 0.15, 0.10, 0.10, 0.20, 0.15],
    #         [0.10, 0.15, 0.10, 0.05, 0.15, 0.10, 0.20, 0.15],
    #         [0.15, 0.22, 0.18, 0.05, 0.08, 0.07, 0.10, 0.15],
    #         [0.15, 0.10, 0.10, 0.05, 0.05, 0.15, 0.15, 0.25],
    #     ],
    #     index=["Старт", "gads:search", "metads:fb", "gads:youtube", "metads:inst", "tiktok", "gads:discover"],
    #     columns=["gads:search", "metads:fb", "gads:youtube", "metads:inst", "tiktok", "gads:discover", "Конверсія", "Відмова"]
    # )
    
    # f2_removal_effects = {
    #     "gads:search": 0.28,
    #     "metads:fb": 0.26,
    #     "gads:youtube": 0.19,
    #     "metads:inst": 0.17,
    #     "tiktok": 0.14,
    #     "gads:discover": 0.09,
    # }
    
    # _safe_render(
    #     "F2",
    #     figure_status,
    #     warnings,
    #     lambda: render_f2_markov_transition_heatmap(
    #         figures_dir / "F2_markov_transition_heatmap.png",
    #         transition_matrix=f2_matrix,
    #         removal_effects=f2_removal_effects,
    #         journeys_count=114250,
    #         base_cr=0.32,
    #     ),
    #     "figures/F2_markov_transition_heatmap.png",
    # )

    # # DDA Out of Sample Actual Plot
    # dda_oos_data = {
    #     "week": [
    #         "2021-12-06", "2021-12-13", "2021-12-20", "2021-12-27", "2022-01-03",
    #         "2022-01-10", "2022-01-17", "2022-01-24", "2022-01-31", "2022-02-07",
    #         "2022-02-14", "2022-02-21", "2022-02-28"
    #     ],
    #     "actual_conv": [1820, 1750, 1680, 3100, 5250, 3800, 3650, 3200, 2800, 2400, 2100, 1350, 180],
    #     "lc_exp": [2450, 2521, 2597, 5791, 9459, 5498, 5594, 4572, 3758, 2872, 3089, 1627, 139],
    #     "sh_exp": [2001, 2059, 2122, 4731, 7727, 4492, 4571, 3734, 3069, 2346, 2523, 1329, 114],
    #     "mk_exp": [2188, 2252, 2321, 5176, 8462, 4915, 5001, 4088, 3360, 2567, 2761, 1454, 125],
    #     "actual_cac": [46.8, 50.1, 53.8, 65.0, 62.7, 50.4, 53.3, 49.7, 46.7, 41.6, 51.2, 41.9, 27.0],
    #     "lc_cac": [34.8]*13,
    #     "sh_cac": [42.6]*13,
    #     "mk_cac": [38.9]*13,
    # }
    # dda_oos_df = pd.DataFrame(dda_oos_data)
    
    # _safe_render(
    #     "DDA_OOS",
    #     figure_status,
    #     warnings,
    #     lambda: render_dda_out_of_sample_forecast(
    #         figures_dir / "DDA_Out_of_Sample.png",
    #         dda_oos_df=dda_oos_df,
    #     ),
    #     "figures/DDA_Out_of_Sample.png",
    # )

    # reports_root = get_reports_root()
    # # F3 (fold_4 preferred)
    # f3_fold = focus_fold
    # f3_model = _pick_model_for_fold(eval_mmm, f3_fold)
    # trace = load_posterior_trace(f3_model, f3_fold, reports_root) if f3_model else None
    # ols = load_ols_coeffs(f3_fold, reports_root)
    # if trace is None or ols is None:
    #     figure_status["F3"] = "skipped (missing posterior trace or OLS coefficients artifact)"
    #     warnings.append(
    #         f"F3 skipped for fold={f3_fold}, model={f3_model}. "
    #         "Expected artifacts: posterior_trace_* and ols_channel_coeffs_*.csv."
    #     )
    # else:
    #     _safe_render(
    #         "F3",
    #         figure_status,
    #         warnings,
    #         lambda: render_f3_posterior_kde_vs_ols(
    #             figures_dir / "F3_posterior_kde_vs_ols.png",
    #             trace=trace,
    #             ols_coeffs=ols,
    #         ),
    #         "figures/F3_posterior_kde_vs_ols.png",
    #     )

    # # F4
    # _safe_render(
    #     "F4",
    #     figure_status,
    #     warnings,
    #     lambda: render_f4_wape_wbias_dual_panel(
    #         figures_dir / "F4_mmm_wape_wbias.png",
    #         mmm_fold_metrics=mmm_fold_metrics,
    #     ),
    #     "figures/F4_mmm_wape_wbias.png",
    # )

    # # F5/F6 require PPC artifacts
    # logger.info("Loading per-fold PPC artifacts for F5/F6.")
    # ppc_by_fold: dict[str, np.ndarray] = {}
    # actual_by_fold: dict[str, np.ndarray] = {}
    # for fold_id in selected_folds:
    #     model_name = _pick_model_for_fold(eval_mmm, fold_id)
    #     if model_name is None:
    #         continue
    #     ppc = load_posterior_predictive(model_name, fold_id, reports_root)
    #     if ppc is None:
    #         warnings.append(f"PPC artifact missing for fold={fold_id}, model={model_name}.")
    #         continue
    #     ppc_by_fold[fold_id] = ppc
    #     fold_actual = (
    #         eval_mmm[
    #             (eval_mmm["fold_id"] == fold_id) & (eval_mmm["model_name"] == model_name)
    #         ]
    #         .groupby("forecast_period", as_index=False)["actual_net_revenue_usd"]
    #         .sum()
    #         .sort_values("forecast_period")
    #     )
    #     actual_by_fold[fold_id] = fold_actual["actual_net_revenue_usd"].astype(float).to_numpy()

    # calibration = calibration_from_ppc(ppc_by_fold, actual_by_fold)
    # metric_tables["mmm_calibration_coverage"] = calibration
    # if calibration.empty:
    #     figure_status["F5"] = "skipped (no fold-level PPC artifacts available)"
    # else:
    #     _safe_render(
    #         "F5",
    #         figure_status,
    #         warnings,
    #         lambda: render_f5_calibration_coverage(
    #             figures_dir / "F5_calibration_coverage.png",
    #             calibration_df=calibration,
    #         ),
    #         "figures/F5_calibration_coverage.png",
    #     )

    # # F6
    # f6_model = _pick_model_for_fold(eval_mmm, focus_fold)
    # f6_ppc = load_posterior_predictive(f6_model, focus_fold, reports_root) if f6_model else None
    # if f6_ppc is None or f6_model is None:
    #     figure_status["F6"] = "skipped (missing focus-fold PPC artifact)"
    # else:
    #     holdout_actual = (
    #         eval_mmm[
    #             (eval_mmm["fold_id"] == focus_fold) & (eval_mmm["model_name"] == f6_model)
    #         ]
    #         .groupby("forecast_period", as_index=False)["actual_net_revenue_usd"]
    #         .sum()
    #         .rename(columns={"forecast_period": "date_week"})
    #     )
    #     fold_spec = get_fold(focus_fold)
    #     mmm_ts = load_mmm_timeseries_fold(client, bq_project, bq_dataset, focus_fold)
    #     train_actual = (
    #         mmm_ts[pd.to_datetime(mmm_ts["date_week"]) < pd.to_datetime(fold_spec["holdout_start"])]
    #         .groupby("date_week", as_index=False)["total_net_revenue_usd"]
    #         .sum()
    #         .rename(columns={"total_net_revenue_usd": "actual_net_revenue_usd"})
    #     )
    #     _safe_render(
    #         "F6",
    #         figure_status,
    #         warnings,
    #         lambda: render_f6_posterior_timeseries(
    #             figures_dir / "F6_posterior_predictive_timeseries.png",
    #             train_series=train_actual,
    #             holdout_actual=holdout_actual,
    #             ppc=f6_ppc,
    #             holdout_start=fold_spec["holdout_start"],
    #         ),
    #         "figures/F6_posterior_predictive_timeseries.png",
    #     )

    # F7/F8/F9
    _safe_render(
        "F7",
        figure_status,
        warnings,
        lambda: render_f7_survival_decay_curves(
            figures_dir / "F7_survival_decay_curves.png",
            dedup_survival=dedup_survival,
        ),
        "figures/F7_survival_decay_curves.png",
    )
    # _safe_render(
    #     "F8",
    #     figure_status,
    #     warnings,
    #     lambda: render_f8_rmse_divergence(
    #         figures_dir / "F8_rmse_divergence.png",
    #         rmse_by_horizon=survival_rmse,
    #     ),
    #     "figures/F8_rmse_divergence.png",
    # )

    # ltv_curves = build_ltv_curves_table(dedup_survival)
    # metric_tables["survival_ltv_curves_d90_plus"] = ltv_curves
    # if ltv_curves.empty:
    #     figure_status["F9"] = "skipped (no D90-eligible cohorts with actual LTV)"
    # else:
    #     _safe_render(
    #         "F9",
    #         figure_status,
    #         warnings,
    #         lambda: render_f9_ltv_small_multiples(
    #             figures_dir / "F9_ltv_extrapolation_bias.png",
    #             ltv_curves=ltv_curves,
    #         ),
    #         "figures/F9_ltv_extrapolation_bias.png",
    #     )

    # metric_files = write_metric_tables(metrics_dir, metric_tables)
    # logger.info("Metric tables written: %s", metric_files)
    # write_summary_markdown(
    #     out["run_dir"] / "summary.md",
    #     selected_folds=selected_folds,
    #     figure_status=figure_status,
    #     warnings=warnings,
    # )
    # manifest = write_manifest(
    #     out["run_dir"],
    #     selected_folds=selected_folds,
    #     figure_status=figure_status,
    #     metric_files=metric_files,
    #     warnings=warnings,
    # )
    # update_latest_pointer(out["base_dir"], manifest)

    # elapsed = time.perf_counter() - run_started
    # logger.info(
    #     "Evaluation complete in %.2fs. folds=%s run_dir=%s rendered=%s warnings=%d",
    #     elapsed,
    #     selected_folds,
    #     out["run_dir"],
    #     {key: val for key, val in figure_status.items() if val.startswith("rendered")},
    #     len(warnings),
    # )


def _safe_render(
    figure_id: str,
    figure_status: dict[str, str],
    warnings: list[str],
    renderer,
    relative_path: str,
) -> None:
    try:
        logger.info("Rendering %s...", figure_id)
        renderer()
        figure_status[figure_id] = f"rendered ({relative_path})"
        logger.info("%s rendered successfully -> %s", figure_id, relative_path)
    except Exception as exc:  # pragma: no cover - plotting edge cases
        figure_status[figure_id] = f"skipped ({exc})"
        warnings.append(f"{figure_id} skipped: {exc}")
        logger.warning("%s skipped: %s", figure_id, exc)


def _pick_model_for_fold(eval_mmm: pd.DataFrame, fold_id: str) -> str | None:
    subset = eval_mmm[eval_mmm["fold_id"] == fold_id]
    available = set(subset["model_name"].astype(str).tolist())
    for model_name in PREFERRED_POSTERIOR_MODELS:
        if model_name in available:
            return model_name
    return None


def _filter_by_folds(frame: pd.DataFrame, folds: list[str]) -> pd.DataFrame:
    if frame.empty or "fold_id" not in frame.columns:
        return frame.copy()
    return frame[frame["fold_id"].isin(folds)].copy()


def _regime_summary(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        cols = ["regime", "model_name", "n_folds", *[f"{col}_mean" for col in value_cols]]
        return pd.DataFrame(columns=cols)
    def to_regime(fold_id: str) -> str:
        if fold_id in {"fold_1", "fold_2"}:
            return "stable_period"
        if fold_id in {"fold_3", "fold_4"}:
            return "regime_change_period"
        return "other"

    working = frame.copy()
    working["regime"] = working["fold_id"].astype(str).map(to_regime)
    agg_map = {f"{column}_mean": (column, "mean") for column in value_cols}
    return (
        working.groupby(["regime", "model_name"], as_index=False)
        .agg(n_folds=("fold_id", "nunique"), **agg_map)
        .sort_values(["regime", "model_name"])
        .reset_index(drop=True)
    )


def _ensure_logging_configured() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1-F9 evaluation/visualization pipeline.")
    parser.add_argument("--bq-project", required=True, help="BigQuery project id")
    parser.add_argument("--bq-dataset", required=True, help="BigQuery dataset id")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional base directory for reports/evaluation runs",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        default=None,
        help="Optional fold ids (example: --folds fold_1 fold_2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        bq_project=args.bq_project,
        bq_dataset=args.bq_dataset,
        output_dir=args.output_dir,
        folds=args.folds,
    )
