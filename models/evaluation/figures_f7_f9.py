"""Renderers for F7-F9 survival figures."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_f7_survival_decay_curves(
    output_path: Path,
    dedup_survival: pd.DataFrame,
) -> None:
    """F7: Retention rate (S(t)) decay curves for highly representative distinct cohorts.
    
    Plots empirical, Baseline, and BdW curves tracking S(t) = active_users(t) / N_0.
    Selects 2x SUB_MONTHLY (different folds, regions, weeks) and 1x SUB_WEEKLY spanning layout.
    """
    if dedup_survival.empty:
        raise ValueError("No survival rows available.")

    # 1. Identify distinct segments and their N_0 and max rebill_period_t
    t1_records = dedup_survival[dedup_survival["rebill_period_t"] == 1].copy()
    n0_by_segment = t1_records.groupby(["fold_id", "segment"])["actual_active_users"].max().reset_index()
    n0_by_segment.rename(columns={"actual_active_users": "n_0"}, inplace=True)

    t_max_seg = dedup_survival.groupby(["fold_id", "segment"])["rebill_period_t"].max().reset_index()
    seg_info = n0_by_segment.merge(t_max_seg, on=["fold_id", "segment"])
    
    # 2. Select specific cohorts
    m1 = []
    filtered_m3 = seg_info[(seg_info["segment"].str.contains("SUB_MONTHLY", na=False)) & (seg_info["fold_id"] == "fold_3")]
    filtered_m3 = filtered_m3.sort_values(["rebill_period_t", "n_0"], ascending=[False, False])
    if not filtered_m3.empty:
        r = filtered_m3.iloc[0]
        m1.append((r.fold_id, r.segment))
        parts = r.segment.split("_")
        region = parts[0] if len(parts) > 0 else "XXX"
        week = parts[1] if len(parts) > 1 else "YYY"
    else:
        region, week = "XXX", "YYY"

    m2 = []
    filtered_m4 = seg_info[(seg_info["segment"].str.contains("SUB_MONTHLY", na=False)) & (seg_info["fold_id"] == "fold_4")]
    filtered_m4 = filtered_m4.sort_values(["rebill_period_t", "n_0"], ascending=[False, False])
    for row in filtered_m4.itertuples():
        if region not in row.segment:
            m2.append((row.fold_id, row.segment))
            break
    if not m2 and not filtered_m4.empty:
        r = filtered_m4.iloc[0]
        m2.append((r.fold_id, r.segment))

    weekly = []
    filtered_w = seg_info[seg_info["segment"].str.contains("SUB_WEEKLY", na=False)]
    filtered_w = filtered_w.sort_values(["rebill_period_t", "n_0"], ascending=[False, False])
    if not filtered_w.empty:
        r = filtered_w.iloc[0]
        weekly.append((r.fold_id, r.segment))

    selected_cohorts = (m1 + m2 + weekly)[:3]

    # 3. Setup subplots (2x2 grid, but index 2 spans both columns in bottom row)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    axes_list = [ax1, ax2, ax3]

    color_map = {
        "Baseline_Survival": "#8d99ae",
        "BdW":  "#277da1",
    }

    for idx, (fold_id, segment) in enumerate(selected_cohorts):
        ax = axes_list[idx]
        
        cohort_df = dedup_survival[
            (dedup_survival["fold_id"] == fold_id) & 
            (dedup_survival["segment"] == segment)
        ].copy()
        
        # Calculate N_0
        n_0 = cohort_df[cohort_df["rebill_period_t"] == 1]["actual_active_users"].max()
        if pd.isna(n_0) or n_0 <= 0:
            ax.axis("off")
            continue
            
        # Empirical observations
        emp_df = cohort_df[cohort_df["model_name"] == "Baseline_Survival"].copy()
        if emp_df.empty:
            emp_df = cohort_df.copy() # Fallback
            emp_df = emp_df.groupby("rebill_period_t")["actual_active_users"].max().reset_index()
            
        emp_df = emp_df.sort_values("rebill_period_t")
        emp_rt = np.minimum(emp_df["actual_active_users"] / n_0, 1.0)
        
        ax.plot(
            emp_df["rebill_period_t"], emp_rt,
            marker="o", label="Фактичні (емп.)", color="#1b4332", linewidth=2.5, zorder=5,
        )
        
        # Calculate mean deviations over t
        mean_devs = []

        # Model expected curves
        for model_name in ["Baseline_Survival", "BdW"]:
            mod_df = cohort_df[cohort_df["model_name"] == model_name].copy()
            if mod_df.empty:
                continue
            
            mod_df = mod_df.sort_values("rebill_period_t")
            mod_rt = np.minimum(mod_df["expected_active_users"] / n_0, 1.0)
            
            label_map = {"Baseline_Survival": "Базовий прогноз (Exp.)", "BdW": "BdW"}
            ax.plot(
                mod_df["rebill_period_t"], mod_rt,
                marker="s" if model_name == "Baseline_Survival" else "^",
                linestyle="--" if model_name == "Baseline_Survival" else "-",
                label=label_map[model_name],
                color=color_map[model_name],
                linewidth=2,
            )
            
            if not emp_df.empty:
                merged = emp_df.merge(mod_df, on="rebill_period_t", how="inner")
                if not merged.empty:
                    a_rt = np.minimum(merged["actual_active_users_x"] / n_0, 1.0)
                    e_rt = np.minimum(merged["expected_active_users_y"] / n_0, 1.0)
                    mask = (a_rt > 0)
                    if mask.any():
                        bias_pct = ((e_rt[mask] - a_rt[mask]) / a_rt[mask]).mean() * 100.0
                        name_s = "Base" if model_name == "Baseline_Survival" else "BdW"
                        mean_devs.append(f"Середнє відхилення ({name_s}): {bias_pct:+.1f}%")

            # Highlight overestimation zone
            if model_name == "Baseline_Survival":
                ax.fill_between(
                    merged["rebill_period_t"],
                    a_rt,
                    e_rt,
                    alpha=0.25, color="#f4a261", label="Зона переоцінки (Baseline)",
                )

        if mean_devs:
            ax.text(0.5, 0.15, "\n".join(mean_devs), transform=ax.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='gray'), zorder=10)

        ax.set_xlabel("Цикл списання (t)")
        ax.set_ylabel("Коефіцієнт утримання (Рейт)")
        ax.set_title(f"Сегмент: {segment}\nFold: {fold_id} (N₀ = {int(n_0)})", fontsize=10)
        ax.grid(alpha=0.25)
        ax1.set_ylim(0.4, 1.05)
        ax2.set_ylim(0.4, 1.05)

    # Hide unused axes if any
    for jdx in range(len(selected_cohorts), len(axes_list)):
        axes_list[jdx].axis("off")

    # Safe legend retrieval
    handles, labels = [], []
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
            
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("F7 — Очікувана крива відмови користувачів від послуг", y=1.03)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def render_f8_rmse_divergence(
    output_path: Path,
    rmse_by_horizon: pd.DataFrame,
) -> None:
    """F8: RMSE by horizon per fold."""
    if rmse_by_horizon.empty:
        raise ValueError("RMSE table is empty.")

    folds = sorted(rmse_by_horizon["fold_id"].unique())
    n_folds = len(folds)
    n_rows = ceil(n_folds / 2) if n_folds > 0 else 1
    
    fig, axes = plt.subplots(n_rows, min(2, n_folds) if n_folds > 1 else 1, figsize=(14, 6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    color_map = {"Baseline_Survival": "#8d99ae", "BdW": "#1d3557"}

    for idx, fold_id in enumerate(folds):
        ax1 = axes_flat[idx]
        fold_df = rmse_by_horizon[rmse_by_horizon["fold_id"] == fold_id]
        
        aggregate = (
            fold_df.groupby(["model_name", "rebill_period_t"], as_index=False)
            .agg(rmse=("rmse", "mean"), cohort_count=("cohort_count", "mean"))
            .sort_values(["model_name", "rebill_period_t"])
        )

        for model_name, group in aggregate.groupby("model_name"):
            ax1.plot(
                group["rebill_period_t"],
                group["rmse"],
                marker="o",
                label=model_name,
                color=color_map.get(model_name),
            )
        
        ax1.set_xlabel("Вік когорти (t)")
        ax1.set_ylabel("RMSE")
        ax1.set_title(f"RMSE {fold_id} (всі підсегменти макрорегіонів)")
        ax1.grid(alpha=0.25)

        support = aggregate.groupby("rebill_period_t", as_index=False)["cohort_count"].mean()
        ax2 = ax1.twinx()
        ax2.bar(
            support["rebill_period_t"],
            support["cohort_count"],
            alpha=0.15,
            color="#6c757d",
            width=0.4,
            label="К-сть когорт",
        )
        ax2.set_ylabel("Середня кількість когорт")
        
        if idx == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("F8 — Дивергенція RMSE за горизонтом прогнозу", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_f9_ltv_small_multiples(
    output_path: Path,
    ltv_curves: pd.DataFrame,
    *,
    max_panels: int = 6,
) -> None:
    """F9: LTV extrapolation bias small multiples driven by BQ ltv_curves data."""
    if ltv_curves.empty:
        raise ValueError("ltv_curves is empty — no D90-eligible cohorts available.")

    # Select up to max_panels diverse (fold_id, segment) combos with the most rebill periods
    period_counts = (
        ltv_curves.groupby(["fold_id", "segment"])["rebill_period_t"]
        .nunique()
        .reset_index(name="n_periods")
        .sort_values("n_periods", ascending=False)
        .head(max_panels)
    )

    n = len(period_counts)
    if n == 0:
        raise ValueError("No cohort panels to render after filtering.")
    n_cols = 2
    n_rows = ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), sharey=False)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (_, cohort_row) in enumerate(period_counts.iterrows()):
        fold_id = str(cohort_row["fold_id"])
        segment = str(cohort_row["segment"])
        ax = axes_flat[idx]

        cohort_df = ltv_curves[
            (ltv_curves["fold_id"] == fold_id) & (ltv_curves["segment"] == segment)
        ].copy()

        def _ltv_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
            return (
                df.groupby("rebill_period_t", as_index=False)[col]
                .mean()
                .sort_values("rebill_period_t")
            )

        actual_df = _ltv_series(
            cohort_df[["rebill_period_t", "actual_ltv_usd"]].dropna(), "actual_ltv_usd"
        )
        baseline_df = _ltv_series(
            cohort_df[cohort_df["model_name"] == "Baseline_Survival"][["rebill_period_t", "expected_ltv_usd"]],
            "expected_ltv_usd",
        )
        bdw_df = _ltv_series(
            cohort_df[cohort_df["model_name"] == "BdW"][["rebill_period_t", "expected_ltv_usd"]],
            "expected_ltv_usd",
        )

        if not actual_df.empty:
            ax.plot(actual_df["rebill_period_t"], actual_df["actual_ltv_usd"],
                    label="Фактичний LTV", color="#1b4332", marker="o", linewidth=2)
        if not baseline_df.empty:
            ax.plot(baseline_df["rebill_period_t"], baseline_df["expected_ltv_usd"],
                    label="Базовий прогноз (Exp.)", color="#8d99ae",
                    linestyle="--", marker="s", linewidth=2)
        if not bdw_df.empty:
            ax.plot(bdw_df["rebill_period_t"], bdw_df["expected_ltv_usd"],
                    label="BdW", color="#277da1", marker="^", linewidth=2)

        # Shade overestimation zone between baseline and actual
        if not actual_df.empty and not baseline_df.empty:
            m = actual_df.merge(baseline_df, on="rebill_period_t", how="inner")
            if not m.empty:
                ax.fill_between(m["rebill_period_t"], m["actual_ltv_usd"], m["expected_ltv_usd"],
                                color="#f4a261", alpha=0.22, label="Зона переоцінки")

        # Compute and display D90 / D180 bias
        bias_msgs = []
        for t_target, label in [(3, "D90"), (6, "D180")]:
            if not actual_df.empty and not baseline_df.empty:
                m_t = actual_df[actual_df["rebill_period_t"] == t_target]
                b_t = baseline_df[baseline_df["rebill_period_t"] == t_target]
                if not m_t.empty and not b_t.empty:
                    actual_v = float(m_t["actual_ltv_usd"].iloc[0])
                    baseline_v = float(b_t["expected_ltv_usd"].iloc[0])
                    if actual_v > 0:
                        pct = (baseline_v - actual_v) / actual_v * 100
                        sign = "+" if pct >= 0 else ""
                        bias_msgs.append(f"{label}: {sign}{pct:.0f}%")

        title = f"{segment}\n({fold_id})"
        if bias_msgs:
            title += "  " + "  ".join(bias_msgs)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Ребіл t")
        ax.set_ylabel("Кум. LTV (USD)")
        ax.grid(alpha=0.25)

    for jdx in range(n, len(axes_flat)):
        axes_flat[jdx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("F9 — Зсув екстраполяції LTV (D90 / D180)", y=1.01)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)




def _retention_curve(df: pd.DataFrame, *, active_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["rebill_period_t", "retention_rate"])
    grouped = (
        df.groupby(["segment", "rebill_period_t"], as_index=False)[active_col]
        .mean()
        .sort_values(["segment", "rebill_period_t"])
    )
    grouped["anchor"] = grouped.groupby("segment")[active_col].transform("first")
    grouped = grouped[grouped["anchor"] > 0].copy()
    grouped["retention_rate"] = grouped[active_col] / grouped["anchor"]
    return (
        grouped.groupby("rebill_period_t", as_index=False)["retention_rate"]
        .mean()
        .sort_values("rebill_period_t")
    )

