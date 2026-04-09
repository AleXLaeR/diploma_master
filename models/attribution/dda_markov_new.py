"""
Markov DDA model (2nd-order) with EWMA-based holdout projection.

Spec references:
    docs/algorithms/dda_models.md §2.2, §3
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_START = "__START__"
_CONVERSION = "__CONVERSION__"
_NULL = "__NULL__"
_SPARSITY_MIN_STATE_OBS = 5


def parse_journeys_to_transitions(
    df: pd.DataFrame,
    order: int = 2,
    check_sparsity: bool = True,
) -> tuple[dict[tuple, dict[tuple, int]], bool]:
    """
    Parse journey rows into N-order transition counts.
    """
    transitions = _build_transitions_for_order(df, order=order)
    is_sparse = False

    if check_sparsity and order > 1:
        # Sparse state criterion:
        # a state is sparse when its total observed outbound transitions
        # are below K observations.
        for edges in transitions.values():
            state_observations = sum(edges.values())
            if state_observations < _SPARSITY_MIN_STATE_OBS:
                is_sparse = True
                break
        if is_sparse:
            logger.warning(
                "Markov sparsity detected at order=%d (state observations < %d). "
                "Applying add-1 smoothing and setting confidence_weight=-1.",
                order,
                _SPARSITY_MIN_STATE_OBS,
            )

    return transitions, is_sparse


def _build_transitions_for_order(
    df: pd.DataFrame,
    order: int,
) -> dict[tuple, dict[tuple, int]]:
    transitions: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        journey_raw = str(row.get("journey", ""))
        if not journey_raw or journey_raw == "nan":
            channels = []
        else:
            channels = [part.strip() for part in journey_raw.split(">") if part.strip()]
        absorbing = _CONVERSION if bool(row["is_converted"]) else _NULL

        padded = [_START] * order + channels
        for idx in range(len(padded) - order):
            state = tuple(padded[idx : idx + order])
            target_state = tuple(padded[idx + 1 : idx + order + 1])
            transitions[state][target_state] += 1

        final_state = tuple(padded[-order:])
        transitions[final_state][(absorbing,)] += 1

    return dict(transitions)


def build_transition_matrix(
    transitions: dict[tuple, dict[tuple, int]],
    alpha: float = 1.0,
) -> tuple[np.ndarray, dict[Any, int]]:
    """
    Build a row-stochastic transition matrix with additive smoothing.
    """
    all_states: set[Any] = set()
    for state, edges in transitions.items():
        all_states.add(state)
        for target in edges:
            all_states.add(target)
    all_states.add((_CONVERSION,))
    all_states.add((_NULL,))

    state_list = sorted(all_states, key=str)
    state_index = {state: idx for idx, state in enumerate(state_list)}
    size = len(state_list)
    matrix = np.zeros((size, size), dtype=float)

    for state, edges in transitions.items():
        row_idx = state_index[state]
        for target, count in edges.items():
            col_idx = state_index[target]
            matrix[row_idx, col_idx] += count

    absorbing_rows = {state_index[(_CONVERSION,)], state_index[(_NULL,)]}
    if alpha > 0:
        for row_idx in range(size):
            if row_idx not in absorbing_rows:
                matrix[row_idx, :] += alpha

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = matrix / row_sums

    for absorbing_state in ((_CONVERSION,), (_NULL,)):
        idx = state_index[absorbing_state]
        matrix[idx, :] = 0.0
        matrix[idx, idx] = 1.0

    return matrix, state_index


def compute_conversion_rate(
    matrix: np.ndarray,
    state_index: dict[Any, int],
    order: int,
) -> float:
    """
    Compute conversion absorption probability from start state.
    """
    conv_idx = state_index[(_CONVERSION,)]
    null_idx = state_index[(_NULL,)]

    absorbing_indices = {conv_idx, null_idx}
    transient_indices = sorted(
        idx for idx in range(len(state_index)) if idx not in absorbing_indices
    )
    if not transient_indices:
        return 0.0

    q = matrix[np.ix_(transient_indices, transient_indices)]
    absorb_order = sorted(absorbing_indices)
    r = matrix[np.ix_(transient_indices, absorb_order)]

    identity = np.eye(len(transient_indices))
    try:
        fundamental = np.linalg.inv(identity - q)
    except np.linalg.LinAlgError:
        logger.error("Markov matrix inversion failed for conversion rate.")
        return 0.0

    absorption = fundamental @ r
    start_state = tuple([_START] * order)
    if start_state not in state_index:
        return 0.0
    start_global_idx = state_index[start_state]
    if start_global_idx not in transient_indices:
        return 0.0

    start_local_idx = transient_indices.index(start_global_idx)
    conv_local_idx = absorb_order.index(conv_idx)
    return float(absorption[start_local_idx, conv_local_idx])


def _get_all_channels(transitions: dict[tuple, dict[tuple, int]]) -> set[str]:
    special = {_START, _CONVERSION, _NULL}
    channels: set[str] = set()
    for state in transitions:
        for token in state:
            if token not in special:
                channels.add(token)
    for edges in transitions.values():
        for target in edges:
            for token in target:
                if token not in special:
                    channels.add(token)
    return channels


def _redirect_channel_to_null(
    transitions: dict[tuple, dict[tuple, int]],
    channel: str,
) -> dict[tuple, dict[tuple, int]]:
    redirected: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    for state, edges in transitions.items():
        for target_state, count in edges.items():
            if channel in state or channel in target_state:
                redirected[state][(_NULL,)] += count
            else:
                redirected[state][target_state] += count
    return dict(redirected)


def compute_removal_effects(
    transitions: dict[tuple, dict[tuple, int]],
    base_cr: float,
    order: int,
    alpha: float = 1.0,
) -> dict[str, float]:
    """
    Compute removal effects RE_x = 1 - CR_-x / CR_total.
    """
    effects: dict[str, float] = {}
    for channel in _get_all_channels(transitions):
        modified = _redirect_channel_to_null(transitions=transitions, channel=channel)
        matrix_mod, index_mod = build_transition_matrix(modified, alpha=alpha)
        cr_mod = compute_conversion_rate(matrix_mod, index_mod, order=order)
        effect = 1.0 - (cr_mod / base_cr) if base_cr > 0 else 0.0
        effects[channel] = max(float(effect), 0.0)
    return effects


def compute_channel_weights(
    removal_effects: dict[str, float],
    exclude_channels: set[str] | None = None,
) -> dict[str, float]:
    """
    Normalize removal effects to attribution weights.
    """
    if exclude_channels:
        removal_effects = {
            channel: value
            for channel, value in removal_effects.items()
            if channel not in exclude_channels
        }

    total = float(sum(removal_effects.values()))
    if total <= 0:
        count = len(removal_effects)
        return {channel: 1.0 / count for channel in removal_effects} if count else {}
    return {channel: value / total for channel, value in removal_effects.items()}


def run(
    bq_project: str,
    bq_dataset: str,
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
) -> None:
    """
    End-to-end Markov DDA pipeline.
    """
    from models.attribution.dda_common_new import (
        fetch_attribution_paths,
        get_bq_client,
        translate_weights_to_forecasts_ewma,
        write_forecasts_to_bq,
        write_weights_to_bq,
    )

    client = get_bq_client()
    paths = fetch_attribution_paths(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        train_end=train_end,
    )
    if paths.empty:
        logger.error("Markov_DDA: attribution paths are empty, aborting fold=%s.", fold_id)
        return

    order = 2
    transitions, is_sparse = parse_journeys_to_transitions(paths, order=order)
    smoothing_alpha = 1.0 if is_sparse else 0.0
    matrix, state_index = build_transition_matrix(
        transitions=transitions,
        alpha=smoothing_alpha,
    )
    base_cr = compute_conversion_rate(matrix=matrix, state_index=state_index, order=order)
    if base_cr <= 0:
        logger.error("Markov_DDA: base conversion rate is zero, aborting fold=%s.", fold_id)
        return

    effects = compute_removal_effects(
        transitions=transitions,
        base_cr=base_cr,
        order=order,
        alpha=smoothing_alpha,
    )
    weights = compute_channel_weights(
        removal_effects=effects,
        exclude_channels={"legacy_untracked", "organic"},
    )

    forecast_df = translate_weights_to_forecasts_ewma(
        weights=weights,
        model_name="Markov_DDA",
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
        train_end=train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        confidence_weight=-1 if is_sparse else 0,
    )
    write_forecasts_to_bq(
        client=client,
        forecast_df=forecast_df,
        project=bq_project,
        dataset=bq_dataset,
        model_name="Markov_DDA",
    )
    write_weights_to_bq(
        client=client,
        weights=weights,
        project=bq_project,
        dataset=bq_dataset,
        model_name="Markov_DDA",
        fold_id=fold_id,
    )
    logger.info("Markov_DDA fold=%s complete.", fold_id)
