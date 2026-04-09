from __future__ import annotations

import pandas as pd

from models.attribution import dda_markov_new


def test_parse_journeys_to_transitions_not_sparse_for_dense_paths() -> None:
    df = pd.DataFrame(
        [
            {"journey": "tiktok > gads:search > metads:fb", "is_converted": True}
            for _ in range(20)
        ]
    )

    _, is_sparse = dda_markov_new.parse_journeys_to_transitions(df=df, order=2, check_sparsity=True)

    assert is_sparse is False


def test_parse_journeys_to_transitions_sparse_for_low_observation_states() -> None:
    df = pd.DataFrame(
        [
            {"journey": "tiktok > gads:search", "is_converted": True},
            {"journey": "metads:fb > gads:youtube", "is_converted": False},
        ]
    )

    _, is_sparse = dda_markov_new.parse_journeys_to_transitions(df=df, order=2, check_sparsity=True)

    assert is_sparse is True

