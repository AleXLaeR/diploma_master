"""Fold resolution and regime grouping helpers."""

from __future__ import annotations

import re

import pandas as pd


def resolve_folds_from_eval_tables(
    eval_dda: pd.DataFrame,
    eval_mmm: pd.DataFrame,
    eval_survival: pd.DataFrame,
    requested_folds: list[str] | None = None,
) -> list[str]:
    """Resolve authoritative fold list from fold_id in output-contract tables."""
    fold_ids: set[str] = set()
    for frame in (eval_dda, eval_mmm, eval_survival):
        if "fold_id" in frame.columns and not frame.empty:
            fold_ids.update(str(value) for value in frame["fold_id"].dropna().unique())

    ordered = sorted(fold_ids, key=_fold_sort_key)
    if requested_folds is None:
        if not ordered:
            raise ValueError("No fold_id values found in eval tables.")
        return ordered

    requested_set = {fold for fold in requested_folds}
    filtered = [fold for fold in ordered if fold in requested_set]
    if not filtered:
        raise ValueError(
            f"Requested folds {requested_folds} are absent in eval tables. "
            f"Available folds: {ordered}"
        )
    return filtered


def preferred_single_fold(folds: list[str]) -> str:
    """Prefer fold_4 when present, otherwise highest fold number."""
    if not folds:
        raise ValueError("No folds available.")
    if "fold_4" in folds:
        return "fold_4"
    return sorted(folds, key=_fold_sort_key)[-1]


def split_stable_vs_regime(folds: list[str]) -> dict[str, list[str]]:
    """Return ROCV regime groups used by comparison framework."""
    stable = [fold for fold in folds if fold in {"fold_1", "fold_2"}]
    regime = [fold for fold in folds if fold in {"fold_3", "fold_4"}]
    return {"stable_period": stable, "regime_change_period": regime}


def _fold_sort_key(fold_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", fold_id)
    if not match:
        return (-1, fold_id)
    return (int(match.group(1)), fold_id)

