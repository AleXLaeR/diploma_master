"""
Shapley DDA model with EWMA-based holdout projection.

Spec references:
    docs/algorithms/dda_models.md §2.3, §3
"""

from __future__ import annotations

import logging
from itertools import combinations
from math import factorial

import pandas as pd

logger = logging.getLogger(__name__)


def parse_journeys_to_coalitions(
    df: pd.DataFrame,
    exclude_channels: set[str] | None = None,
) -> tuple[dict[frozenset[str], float], set[str]]:
    """
    Build coalition characteristic function v(S) from journeys.
    """
    excluded = exclude_channels or set()
    coalition_conversions: dict[frozenset[str], float] = {}
    coalition_occurrences: dict[frozenset[str], float] = {}
    channels: set[str] = set()

    for _, row in df.iterrows():
        channel_list = [
            token.strip()
            for token in str(row["journey"]).split(">")
            if token.strip() and token.strip() not in excluded
        ]
        if not channel_list:
            continue

        coalition = frozenset(channel_list)
        coalition_occurrences[coalition] = coalition_occurrences.get(coalition, 0.0) + 1.0
        if bool(row["is_converted"]):
            coalition_conversions[coalition] = coalition_conversions.get(coalition, 0.0) + 1.0
        channels.update(channel_list)

    characteristic: dict[frozenset[str], float] = {}
    for coalition, occurrences in coalition_occurrences.items():
        characteristic[coalition] = coalition_conversions.get(coalition, 0.0) / occurrences

    return characteristic, channels


def approximate_missing_coalitions(
    characteristic: dict[frozenset[str], float],
    channels: set[str],
) -> tuple[dict[frozenset[str], float], bool]:
    """
    Approximate unobserved coalitions via mean of immediate sub-coalitions.
    """
    channel_list = sorted(channels)
    count = len(channel_list)
    full_characteristic = dict(characteristic)
    used_fallback = False

    for coalition_size in range(1, count + 1):
        for combo in combinations(channel_list, coalition_size):
            coalition = frozenset(combo)
            if coalition in full_characteristic:
                continue

            if coalition_size == 1:
                full_characteristic[coalition] = 0.0
            else:
                sub_values = []
                for channel in combo:
                    sub_coalition = coalition - {channel}
                    if sub_coalition in full_characteristic:
                        sub_values.append(full_characteristic[sub_coalition])
                full_characteristic[coalition] = (
                    sum(sub_values) / len(sub_values) if sub_values else 0.0
                )
            used_fallback = True

    full_characteristic[frozenset()] = 0.0
    return full_characteristic, used_fallback


def compute_shapley_values(
    characteristic: dict[frozenset[str], float],
    channels: set[str],
) -> dict[str, float]:
    """
    Compute exact Shapley values for all channels.
    """
    ordered_channels = sorted(channels)
    n = len(ordered_channels)
    n_factorial = factorial(n)
    values: dict[str, float] = {}

    for channel in ordered_channels:
        phi = 0.0
        others = [candidate for candidate in ordered_channels if candidate != channel]
        for k in range(0, len(others) + 1):
            coefficient = factorial(k) * factorial(n - k - 1) / n_factorial
            for combo in combinations(others, k):
                coalition = frozenset(combo)
                coalition_with_channel = coalition | {channel}
                marginal = characteristic.get(coalition_with_channel, 0.0) - characteristic.get(
                    coalition,
                    0.0,
                )
                phi += coefficient * marginal
        values[channel] = phi

    return values


def normalise_shapley_weights(
    shapley_values: dict[str, float],
    exclude_channels: set[str] | None = None,
) -> dict[str, float]:
    """
    Normalize shapley values to positive weights summing to 1.

    If negatives are present, apply affine shift with +5% baseline range.
    """
    if exclude_channels:
        shapley_values = {
            channel: value
            for channel, value in shapley_values.items()
            if channel not in exclude_channels
        }
    if not shapley_values:
        return {}

    min_value = min(shapley_values.values())
    max_value = max(shapley_values.values())

    # affine shift to positive values
    if min_value <= 0:
        epsilon = (max_value - min_value) * 0.05
        if epsilon == 0:
            epsilon = 1e-4
        shifted = {
            channel: (value - min_value) + epsilon
            for channel, value in shapley_values.items()
        }
    else:
        shifted = dict(shapley_values)

    total = sum(shifted.values())
    if total <= 0:
        count = len(shifted)
        return {channel: 1.0 / count for channel in shifted} if count else {}
    return {channel: value / total for channel, value in shifted.items()}


def run(
    bq_project: str,
    bq_dataset: str,
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
) -> None:
    """
    End-to-end Shapley DDA pipeline.
    """
    from models.attribution.dda_common_new import (
        fetch_attribution_paths,
        get_bq_client,
        translate_weights_to_forecasts_ewma,
        write_forecasts_to_bq,
        write_weights_to_bq,
    )

    excluded_channels = {"legacy_untracked", "organic"}
    client = get_bq_client()
    paths = fetch_attribution_paths(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        train_end=train_end,
    )
    if paths.empty:
        logger.error("Shapley_DDA: attribution paths are empty, aborting fold=%s.", fold_id)
        return

    characteristic, channels = parse_journeys_to_coalitions(
        df=paths,
        exclude_channels=excluded_channels,
    )
    characteristic_full, used_fallback = approximate_missing_coalitions(
        characteristic=characteristic,
        channels=channels,
    )
    shapley_values = compute_shapley_values(
        characteristic=characteristic_full,
        channels=channels,
    )
    weights = normalise_shapley_weights(
        shapley_values=shapley_values,
        exclude_channels=excluded_channels,
    )

    forecast_df = translate_weights_to_forecasts_ewma(
        weights=weights,
        model_name="Shapley_DDA",
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
        train_end=train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        confidence_weight=-1 if used_fallback else 0,
    )
    write_forecasts_to_bq(
        client=client,
        forecast_df=forecast_df,
        project=bq_project,
        dataset=bq_dataset,
        model_name="Shapley_DDA",
    )
    write_weights_to_bq(
        client=client,
        weights=weights,
        project=bq_project,
        dataset=bq_dataset,
        model_name="Shapley_DDA",
        fold_id=fold_id,
    )
    logger.info("Shapley_DDA fold=%s complete.", fold_id)

