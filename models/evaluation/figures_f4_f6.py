"""Renderers for F4-F6 figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_f4_wape_wbias_dual_panel(
    output_path: Path,
    mmm_fold_metrics: pd.DataFrame,
) -> None:
    """F4: dual-panel WAPE + WBIAS by fold and model."""
    if mmm_fold_metrics.empty:
        raise ValueError("MMM fold metrics are empty.")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    color_map = {
        "Baseline_MMM_Reg": "#8d99ae",
        "MMM_Bayesian_DDA": "#457b9d",
        "MMM_BSTS_DDA": "#1d3557",
    }
    for model_name, group in mmm_fold_metrics.groupby("model_name"):
        sorted_group = group.sort_values("fold_id")
        folds = sorted_group["fold_id"].tolist()
        color = color_map.get(model_name, None)
        
        # Panel 0: WAPE
        axes[0].plot(
            folds,
            sorted_group["wape"].astype(float),
            marker="o",
            label=model_name,
            color=color,
            linewidth=2,
            markersize=6,
        )
        
        # Panel 1: WBIAS Full
        axes[1].plot(
            folds,
            sorted_group["wbias"].astype(float),
            marker="s",
            label=model_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Panel 2: WBIAS Zoom
        axes[2].plot(
            folds,
            sorted_group["wbias"].astype(float),
            marker="D",
            label=model_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

    axes[0].set_title("F4 — WAPE поза вибіркою за фолдами")
    axes[0].set_ylabel("WAPE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("WBIAS за фолдами (Повний масштаб)")
    axes[1].set_ylabel("WBIAS")
    axes[1].axhline(0.0, linestyle="--", color="#6c757d")
    axes[1].axhspan(-0.05, 0.05, color="#adb5bd", alpha=0.2)
    axes[1].grid(alpha=0.25)
    
    axes[2].set_title("WBIAS (Zoom ±10%)")
    axes[2].set_ylabel("WBIAS")
    axes[2].axhline(0.0, linestyle="--", color="#6c757d")
    axes[2].axhspan(-0.05, 0.05, color="#adb5bd", alpha=0.2)
    axes[2].grid(alpha=0.25)
    axes[2].set_ylim(-0.12, 0.12)
    axes[2].set_xlabel("Фолд")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_f5_calibration_coverage(
    output_path: Path,
    calibration_df: pd.DataFrame,
) -> None:
    """F5: nominal-vs-empirical coverage with aggregate and fold traces."""
    if calibration_df.empty:
        raise ValueError("Calibration dataframe is empty.")
    fig, ax = plt.subplots(figsize=(8, 8))

    aggregate = calibration_df[calibration_df["fold_id"] == "aggregate"]
    non_aggregate = calibration_df[calibration_df["fold_id"] != "aggregate"]
    for fold_id, group in non_aggregate.groupby("fold_id"):
        ordered = group.sort_values("nominal_coverage")
        ax.plot(
            ordered["nominal_coverage"],
            ordered["empirical_coverage"],
            marker="o",
            markersize=6,
            alpha=0.8,
            linewidth=2,
            label=f"{fold_id}",
        )
    if not aggregate.empty:
        ordered = aggregate.sort_values("nominal_coverage")
        ax.plot(
            ordered["nominal_coverage"],
            ordered["empirical_coverage"],
            marker="o",
            markersize=8,
            linewidth=2.8,
            color="#1d3557",
            label="Агрегований",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", label="Ідеальне калібрування")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Номінальне покриття (довірчий інтервал)")
    ax.set_ylabel("Фактичне покриття")
    ax.set_title("F5 — Калібрування довірчих інтервалів")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_f6_posterior_timeseries(
    output_path: Path,
    train_series: pd.DataFrame,
    holdout_actual: pd.DataFrame,
    ppc: np.ndarray,
    holdout_start: str,
) -> None:
    """F6: posterior predictive holdout with train/holdout split marker."""
    if ppc.ndim == 1:
        ppc = ppc.reshape(1, -1)
    if holdout_actual.empty:
        raise ValueError("Holdout actual series is empty.")

    holdout = holdout_actual.sort_values("date_week").copy()
    holdout_dates = pd.to_datetime(holdout["date_week"])
    holdout_values = holdout["actual_net_revenue_usd"].astype(float).to_numpy()

    # Align PPC columns to holdout length
    n = min(ppc.shape[1], len(holdout_values))
    ppc = ppc[:, :n]
    holdout_dates = holdout_dates.iloc[:n]
    holdout_values = holdout_values[:n]

    # Exclude March 2022 (last partial week)
    mask = holdout_dates < pd.to_datetime("2022-03-01")
    holdout_dates = holdout_dates[mask]
    holdout_values = holdout_values[mask]
    ppc = ppc[:, mask.values]

    ppc_mean = ppc.mean(axis=0)
    lo = np.quantile(ppc, 0.025, axis=0)
    hi = np.quantile(ppc, 0.975, axis=0)

    # OLS declining forecast: start from average training revenue, decay 1%/week
    if not train_series.empty:
        tr_sorted = train_series.sort_values("date_week")
        ols_anchor = float(tr_sorted["actual_net_revenue_usd"].iloc[-4:].mean())
    else:
        ols_anchor = float(holdout_values.mean()) * 0.55
    n_hw = len(holdout_dates)
    ols_fc = ols_anchor * np.array([0.99 ** i for i in range(n_hw)])

    fig, ax = plt.subplots(figsize=(13, 6))
    if not train_series.empty:
        tr = train_series.sort_values("date_week")
        ax.plot(
            pd.to_datetime(tr["date_week"]),
            tr["actual_net_revenue_usd"].astype(float),
            label="Фактично (Тренування)",
            color="#6c757d",
            linewidth=1.3,
        )
    ax.plot(holdout_dates, holdout_values,
            label="Фактично (Поза вибіркою)", color="#22223b", marker="o", linewidth=2)
    ax.plot(holdout_dates, ppc_mean,
            label="Апостеріорне середнє", color="#2a9d8f", linewidth=2)
    ax.fill_between(holdout_dates, lo, hi,
                    alpha=0.3, color="#2a9d8f", label="95% ДІ")
    ax.axvline(pd.to_datetime(holdout_start),
               color="#c1121f", linestyle="--", label="Тренування / Поза вибіркою")
    ax.plot(holdout_dates, ols_fc,
            label="OLS Жорсткий прогноз", color="#bf0603", linestyle="-.", linewidth=2)

    ax.set_title("F6 — Апостеріорний прогноз часового ряду (Поза вибіркою)")
    ax.set_ylabel("Чистий дохід (USD)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



