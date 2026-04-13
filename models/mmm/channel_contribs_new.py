"""Modeled channel contribution helpers for MMM v2 (task #9)."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import pandas as pd

from models.mmm.common_new import CHANNEL_NAMES, segment_for_region
from models.mmm.persistence_new import build_channel_contrib_row

logger = logging.getLogger(__name__)


def normalize_channel_totals(
    channel_totals: Mapping[str, float],
    fold_id: str,
    model_name: str,
    segment: str,
) -> dict[str, float]:
    """Normalize modeled channel totals to shares that sum to 1 or all zeros."""
    clipped = {channel: max(float(channel_totals.get(channel, 0.0)), 0.0) for channel in CHANNEL_NAMES}
    total = float(sum(clipped.values()))

    if total <= 0.0:
        logger.warning(
            "Zero modeled paid contribution for fold=%s model=%s segment=%s; storing zeros.",
            fold_id,
            model_name,
            segment,
        )
        return {channel: 0.0 for channel in CHANNEL_NAMES}

    return {channel: clipped[channel] / total for channel in CHANNEL_NAMES}

def shares_from_weekly_contribs(
    weekly_contribs: pd.DataFrame,
    fold_id: str,
    model_name: str,
    segment: str,
) -> dict[str, float]:
    """Aggregate weekly modeled channel contributions to normalized incr shares."""
    totals: dict[str, float] = {}
    for channel in CHANNEL_NAMES:
        if channel in weekly_contribs.columns:
            totals[channel] = float(weekly_contribs[channel].sum())
        else:
            totals[channel] = 0.0

    return normalize_channel_totals(totals, fold_id, model_name, segment)


def build_contrib_rows(
    fold_id: str,
    model_name: str,
    shares_by_segment: Mapping[str, Mapping[str, float]],
) -> list[dict]:
    """Create mmm_channel_contribs row payloads for all segments."""
    rows: list[dict] = []
    for segment in sorted(shares_by_segment.keys()):
        rows.append(
            build_channel_contrib_row(
                fold_id=fold_id,
                model_name=model_name,
                segment=segment,
                shares_by_channel=dict(shares_by_segment[segment]),
            )
        )
    return rows


def replicate_global_shares_to_regions(
    global_shares: Mapping[str, float],
    regions: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Replicate one Global modeled share vector to fallback region segments."""
    return {
        segment_for_region(region): {channel: float(global_shares.get(channel, 0.0)) for channel in CHANNEL_NAMES}
        for region in regions
    }
