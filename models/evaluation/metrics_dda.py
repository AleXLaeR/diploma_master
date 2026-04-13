"""DDA evaluation metrics and bootstrap utilities."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.attribution import dda_shapley_new

PAID_CHANNELS: list[str] = [
    "gads:search",
    "gads:youtube",
    "gads:discover",
    "metads:inst",
    "metads:fb",
    "tiktok",
]


def compute_dda_wape(eval_dda: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-level conversion and CAC WAPE for each DDA model."""
    if eval_dda.empty:
        return pd.DataFrame(
            columns=["fold_id", "model_name", "wape_conversions", "wape_cac"]
        )

    rows: list[dict[str, object]] = []
    for (fold_id, model_name), group in eval_dda.groupby(["fold_id", "model_name"]):
        actual_conv = group["actual_conversions"].astype(float)
        expected_conv = group["expected_conversions"].astype(float)
        actual_cac = group["actual_cac_usd"].astype(float)
        expected_cac = group["expected_cac_usd"].astype(float)

        conv_denom = float(actual_conv.sum())
        cac_denom = float(actual_cac.sum())

        wape_conv = (
            float((expected_conv.sub(actual_conv).abs().sum() / conv_denom))
            if conv_denom > 0
            else np.nan
        )
        wape_cac = (
            float((expected_cac.sub(actual_cac).abs().sum() / cac_denom))
            if cac_denom > 0
            else np.nan
        )
        rows.append(
            {
                "fold_id": fold_id,
                "model_name": model_name,
                "wape_conversions": wape_conv,
                "wape_cac": wape_cac,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "fold_id"]).reset_index(drop=True)


def aggregate_dda_wape(fold_wape: pd.DataFrame) -> pd.DataFrame:
    if fold_wape.empty:
        return pd.DataFrame(
            columns=["model_name", "wape_conversions_mean", "wape_cac_mean", "n_folds"]
        )
    out = (
        fold_wape.groupby("model_name", as_index=False)
        .agg(
            wape_conversions_mean=("wape_conversions", "mean"),
            wape_cac_mean=("wape_cac", "mean"),
            n_folds=("fold_id", "nunique"),
        )
        .sort_values("model_name")
        .reset_index(drop=True)
    )
    return out


def bootstrap_shapley_stability(
    paths_by_fold: dict[str, pd.DataFrame],
    *,
    iterations: int = 200,
    min_channel_paths: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute bootstrap mean/std of fold-aware Shapley channel weights.

    Channels with fewer than `min_channel_paths` observed paths in a resample are
    excluded from that resample instead of forced to zero.
    """
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []

    for fold_id, df in paths_by_fold.items():
        if df.empty:
            continue

        clean = df.copy()
        clean["journey"] = clean["journey"].astype(str)
        n = len(clean)
        if n == 0:
            continue

        for boot_idx in range(iterations):
            sample_idx = rng.integers(0, n, size=n)
            sample = clean.iloc[sample_idx].reset_index(drop=True)

            path_counts = _count_paths_by_channel(sample["journey"].tolist())
            sparse_channels = {
                channel
                for channel in PAID_CHANNELS
                if path_counts.get(channel, 0) < min_channel_paths
            }
            excluded = {"legacy_untracked", "organic", *sparse_channels}
            characteristic, channels = dda_shapley_new.parse_journeys_to_coalitions(
                sample,
                exclude_channels=excluded,
            )
            if not channels:
                continue
            characteristic_full, _ = dda_shapley_new.approximate_missing_coalitions(
                characteristic=characteristic,
                channels=channels,
            )
            shapley = dda_shapley_new.compute_shapley_values(
                characteristic=characteristic_full,
                channels=channels,
            )
            weights = dda_shapley_new.normalise_shapley_weights(
                shapley_values=shapley,
                exclude_channels=excluded,
            )

            for channel in PAID_CHANNELS:
                rows.append(
                    {
                        "fold_id": fold_id,
                        "bootstrap_idx": boot_idx,
                        "media_source": channel,
                        "weight": weights.get(channel, np.nan),
                    }
                )
            print(f"Fold {fold_id} bootstrap {boot_idx} completed")

    if not rows:
        return pd.DataFrame(columns=["media_source", "weight_mean", "weight_std", "n_samples"])

    boot = pd.DataFrame(rows)
    out = (
        boot.groupby("media_source", as_index=False)
        .agg(
            weight_mean=("weight", "mean"),
            weight_std=("weight", "std"),
            n_samples=("weight", "count"),
        )
        .sort_values("media_source")
        .reset_index(drop=True)
    )
    return out


def compute_spearman_concordance(
    dda_weights: pd.DataFrame,
    mmm_channel_contribs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute fold-level DDA-vs-MMM rank concordance table."""
    rows: list[dict[str, object]] = []
    if dda_weights.empty or mmm_channel_contribs.empty:
        return pd.DataFrame(columns=["fold_id", "comparison", "rho", "p_value"])

    mmm_priority = ["MMM_BSTS_DDA", "MMM_Bayesian_DDA"]
    incr_col_by_channel = {
        "gads:search": "incr_gads_search",
        "gads:youtube": "incr_gads_youtube",
        "gads:discover": "incr_gads_discover",
        "metads:inst": "incr_metads_inst",
        "metads:fb": "incr_metads_fb",
        "tiktok": "incr_tiktok",
    }

    for fold_id in sorted(set(dda_weights["fold_id"].astype(str))):
        mmm_fold = mmm_channel_contribs[mmm_channel_contribs["fold_id"] == fold_id]
        if mmm_fold.empty:
            continue

        selected_mmm = None
        for model_name in mmm_priority:
            candidate = mmm_fold[mmm_fold["model_name"] == model_name]
            if not candidate.empty:
                selected_mmm = candidate
                break
        if selected_mmm is None:
            continue

        mmm_means = {
            channel: float(selected_mmm[incr_col].mean())
            for channel, incr_col in incr_col_by_channel.items()
            if incr_col in selected_mmm.columns
        }
        if len(mmm_means) < len(PAID_CHANNELS):
            continue

        mmm_rank = _rank_desc(mmm_means)
        for dda_model in ("Baseline_LastClick", "Shapley_DDA"):
            dda_fold = dda_weights[
                (dda_weights["fold_id"] == fold_id)
                & (dda_weights["model_name"] == dda_model)
            ]
            if dda_fold.empty:
                continue
            dda_map = {
                str(row["media_source"]): float(row["weight"])
                for _, row in dda_fold.iterrows()
                if str(row["media_source"]) in PAID_CHANNELS
            }
            if len(dda_map) < len(PAID_CHANNELS):
                continue
            dda_rank = _rank_desc(dda_map)
            rho, pval = spearmanr(
                [dda_rank[channel] for channel in PAID_CHANNELS],
                [mmm_rank[channel] for channel in PAID_CHANNELS],
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "comparison": f"{dda_model} vs MMM",
                    "rho": float(rho),
                    "p_value": float(pval),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    formatted = []
    folds = sorted([f for f in out["fold_id"].unique()])
    for fold in folds:
        df_fold = out[out["fold_id"] == fold]
        lc_row = df_fold[df_fold["comparison"] == "Baseline_LastClick vs MMM"]
        sh_row = df_fold[df_fold["comparison"] == "Shapley_DDA vs MMM"]
        
        lc_rho = float(lc_row["rho"].values[0]) if not lc_row.empty else np.nan
        sh_rho = float(sh_row["rho"].values[0]) if not sh_row.empty else np.nan
        sh_pval = float(sh_row["p_value"].values[0]) if not sh_row.empty else np.nan
        
        if np.isnan(sh_pval):
            pval_str = "—"
        elif sh_pval < 0.01:
            pval_str = "<0.01"
        elif sh_pval < 0.05:
            pval_str = "<0.05"
        else:
            pval_str = f"{sh_pval:.2f}"
            
        formatted.append({
            "Fold": fold,
            "Last-Click ρ vs MMM": round(lc_rho, 2) if not np.isnan(lc_rho) else np.nan,
            "Shapley ρ vs MMM": round(sh_rho, 2) if not np.isnan(sh_rho) else np.nan,
            "p-value (Shapley)": pval_str
        })
        
    df_new = pd.DataFrame(formatted)
    
    if not df_new.empty:
        mean_lc = df_new["Last-Click ρ vs MMM"].mean()
        mean_sh = df_new["Shapley ρ vs MMM"].mean()
        # Add Mean row
        df_new.loc[len(df_new)] = {
            "Fold": "Mean",
            "Last-Click ρ vs MMM": round(mean_lc, 2) if not np.isnan(mean_lc) else np.nan,
            "Shapley ρ vs MMM": round(mean_sh, 2) if not np.isnan(mean_sh) else np.nan,
            "p-value (Shapley)": "—"
        }
    return df_new


def _count_paths_by_channel(journeys: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for journey in journeys:
        parts = [token.strip() for token in str(journey).split(">") if token.strip()]
        for channel in set(parts):
            counts[channel] += 1
    return counts


def _rank_desc(values: dict[str, float]) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average", ascending=False)
    return {index: float(rank) for index, rank in ranks.items()}

