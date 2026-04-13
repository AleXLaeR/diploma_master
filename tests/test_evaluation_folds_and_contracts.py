from __future__ import annotations

import pandas as pd
import pytest

from models.evaluation.contracts import validate_factual_readiness
from models.evaluation.folds import preferred_single_fold, resolve_folds_from_eval_tables


def test_resolve_folds_from_eval_tables_prefers_authoritative_fold_ids() -> None:
    eval_dda = pd.DataFrame({"fold_id": ["fold_2", "fold_1"]})
    eval_mmm = pd.DataFrame({"fold_id": ["fold_4"]})
    eval_survival = pd.DataFrame({"fold_id": ["fold_3"]})

    folds = resolve_folds_from_eval_tables(eval_dda, eval_mmm, eval_survival)
    assert folds == ["fold_1", "fold_2", "fold_3", "fold_4"]
    assert preferred_single_fold(folds) == "fold_4"


def test_validate_factual_readiness_raises_on_null_actuals() -> None:
    eval_dda = pd.DataFrame(
        {
            "fold_id": ["fold_4"],
            "actual_conversions": [None],
            "actual_cac_usd": [1.0],
        }
    )
    eval_mmm = pd.DataFrame(
        {
            "fold_id": ["fold_4"],
            "actual_net_revenue_usd": [1.0],
        }
    )
    eval_survival = pd.DataFrame(
        {
            "fold_id": ["fold_4"],
            "actual_active_users": [1.0],
            "actual_ltv_usd": [1.0],
        }
    )
    with pytest.raises(ValueError):
        validate_factual_readiness(eval_dda, eval_mmm, eval_survival, ["fold_4"])

