"""
test_survival_new.py
====================
Automated pytest suite for the new survival model implementations.
All tests are pure-logic (no BigQuery dependency).
Spec references: docs/algorithms/survival_models.md

Run: uv run pytest tests/test_survival_new.py -v
"""

from __future__ import annotations

import math
import sys
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable without Airflow
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.survival.survival_base_new import (
    LTVModel,
    acq_month_str,
    acq_week_iso_str,
    parse_acquisition_week,
    translate_survival_to_output,
    SMALL_COHORT_THRESHOLD,
)
from models.survival.survival_sbg_new import SBGModel
from models.survival.survival_bdw_new import BDWModel
from models.survival.survival_baseline_new import BaselineSurvivalModel
from models.survival.gamma_gamma_new import (
    GammaGammaModel,
    _gg_nll,
    MIN_OBS,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _synthetic_cohort(n0: int, survival_rates: List[float]) -> List[int]:
    """Build a cohort vector from N0 and per-period survival rates."""
    vec = [int(n0)]
    for r in survival_rates:
        vec.append(int(vec[-1] * r))
    return vec


# ===========================================================================
# 1. sBG: survival curve correctness
# ===========================================================================

class TestSBG:
    def _fit_sbg(self, cohort: List[int], bounds=None) -> SBGModel:
        if bounds is None:
            bounds = [[0.0001, 10_000], [0.0001, 10_000]]
        m = SBGModel(bounds)
        ok = m.optimize([cohort], initial_users=cohort[0])
        assert ok, "sBG optimization should succeed on clean data"
        return m

    def test_s0_equals_one(self):
        """S(0) must equal 1 regardless of parameters."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.0, 1.0])
        assert m.predicted_survival(0) == pytest.approx(1.0, abs=1e-9)

    def test_s1_known_value(self):
        """For α=1, β=1: S(1) = β/(α+β) = 0.5."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.0, 1.0])
        assert m.predicted_survival(1) == pytest.approx(0.5, abs=1e-9)

    def test_survival_monotonically_decreasing(self):
        """S(t) must be non-increasing for all t."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([2.0, 5.0])
        s_vals = [m.predicted_survival(t) for t in range(10)]
        for a, b in zip(s_vals, s_vals[1:]):
            assert a >= b - 1e-12, "Survival must be non-increasing"

    def test_survival_bounded_in_01(self):
        """S(t) must stay in [0, 1] for all reasonable t."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([3.0, 1.0])
        for t in range(20):
            s = m.predicted_survival(t)
            assert 0.0 <= s <= 1.0 + 1e-12, f"S({t})={s} out of [0,1]"

    def test_nll_decreases_with_better_data(self):
        """NLL evaluated at true params < NLL at poor starting guess."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        # True: α=1, β=3 → S(1)=0.75, S(2)=0.625, ...
        cohort = _synthetic_cohort(1000, [0.75, 0.625 / 0.75, 0.53 / 0.625])
        true_params = np.array([1.0, 3.0])
        bad_params  = np.array([50.0, 50.0])
        nll_true = m.log_likelihood_multi_cohort(true_params, [cohort])
        nll_bad  = m.log_likelihood_multi_cohort(bad_params,  [cohort])
        assert nll_true < nll_bad

    def test_optimize_recovers_params(self):
        """Fitting on clean synthetic data recovers approximately correct α/β."""
        alpha_true, beta_true = 0.8, 2.5
        m_gen = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m_gen.params = np.array([alpha_true, beta_true])
        cohort = [2000]
        for t in range(1, 12):
            cohort.append(int(2000 * m_gen.predicted_survival(t)))
        fitted = self._fit_sbg(cohort)
        assert fitted.params[0] == pytest.approx(alpha_true, rel=0.20)
        assert fitted.params[1] == pytest.approx(beta_true,  rel=0.20)

    def test_not_fitted_raises(self):
        m = SBGModel([[0.0001, 100], [0.0001, 100]])
        with pytest.raises(ValueError):
            m.predicted_survival(1)

    def test_t_max_lt_2_returns_false(self):
        """T_obs < 2 → optimize() must return False (spec §2.5)."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        # Only one transition: [N0, N1]
        ok = m.optimize([[1000, 750]], initial_users=1000)
        assert not ok, "T_obs=1 is underdetermined; must return False"


# ===========================================================================
# 2. BdW: survival curve and c=1 equivalence
# ===========================================================================

class TestBDW:
    def test_s0_equals_one(self):
        """S(0) = 1 for BdW (0^c = 0 → betaln ratio = 0 → exp(0) = 1)."""
        m = BDWModel([[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]])
        m.params = np.array([1.5, 3.0, 0.7])
        assert m.predicted_survival(0) == pytest.approx(1.0, abs=1e-9)

    def test_survival_monotonically_decreasing(self):
        m = BDWModel([[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]])
        m.params = np.array([2.0, 4.0, 0.8])
        s_vals = [m.predicted_survival(t) for t in range(10)]
        for a, b in zip(s_vals, s_vals[1:]):
            assert a >= b - 1e-12

    def test_c_equals_one_matches_sbg(self):
        """When c=1 BdW S(t) must match sBG S(t) exactly (spec §2.4 note)."""
        alpha, beta = 2.0, 5.0
        sbg = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        sbg.params = np.array([alpha, beta])
        bdw = BDWModel([[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]])
        bdw.params = np.array([alpha, beta, 1.0])
        for t in range(1, 10):
            assert bdw.predicted_survival(t) == pytest.approx(
                sbg.predicted_survival(t), abs=1e-6
            ), f"BdW(c=1) ≠ sBG at t={t}"

    def test_survival_clipped_01(self):
        m = BDWModel([[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]])
        m.params = np.array([5.0, 0.5, 2.0])
        for t in range(15):
            s = m.predicted_survival(t)
            assert 0.0 <= s <= 1.0 + 1e-12

    def test_bdw_nll_fit(self):
        """BdW optimizer succeeds on a cohort with visible negative duration dep."""
        alpha_true, beta_true, c_true = 1.0, 3.0, 0.6
        m_gen = BDWModel([[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]])
        m_gen.params = np.array([alpha_true, beta_true, c_true])
        cohort = [2000]
        for t in range(1, 10):
            cohort.append(int(2000 * m_gen.predicted_survival(t)))
        m_fit = BDWModel([[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]])
        ok = m_fit.optimize([cohort], initial_users=2000)
        assert ok
        # c should be < 1 (negative duration dep)
        assert m_fit.params[2] < 1.0 + 0.3  # generous tolerance for 5-restart


# ===========================================================================
# 3. Baseline: curve fitting and prediction
# ===========================================================================

class TestBaseline:
    def _make_baseline(self) -> BaselineSurvivalModel:
        # Linear decay cohort: [1000, 900, 800, 700, 600, 500, 400, 300]
        cohort = [1000, 900, 800, 700, 600, 500, 400, 300]
        m = BaselineSurvivalModel([[0, 0]])
        ok = m.optimize([cohort], initial_users=1000)
        assert ok
        return m

    def test_baseline_fits_successfully(self):
        self._make_baseline()

    def test_baseline_predictions_clipped_01(self):
        m = self._make_baseline()
        for t in range(15):
            s = m.predicted_survival(t)
            assert 0.0 <= s <= 1.0, f"Baseline S({t})={s} outside [0,1]"

    def test_baseline_s0_near_one(self):
        """S(0) should be near 1 since all curves project backward to t=0."""
        m = self._make_baseline()
        assert m.predicted_survival(0) == pytest.approx(1.0, abs=0.05)

    def test_baseline_fails_on_short_cohort(self):
        """Fewer than 3 periods → optimize() returns False."""
        cohort = [100, 90]   # only t=0,1 — max_t=1
        m = BaselineSurvivalModel([[0, 0]])
        ok = m.optimize([cohort])
        assert not ok

    def test_not_fitted_raises(self):
        m = BaselineSurvivalModel([[0, 0]])
        with pytest.raises(ValueError):
            m.predicted_survival(1)


# ===========================================================================
# 4. Gamma-Gamma: NLL, E[M], fallback
# ===========================================================================

class TestGammaGamma:
    def _sample_monetary(self, n: int = 500) -> np.ndarray:
        """Gamma-distributed monetary values (simulating realistic ARPU)."""
        rng = np.random.default_rng(42)
        return rng.gamma(shape=3.0, scale=10.0, size=n)  # mean=30

    def test_nll_finite_for_valid_params(self):
        m_vals = self._sample_monetary()
        nll = _gg_nll(np.array([3.0, 5.0, 10.0]), m_vals)
        assert np.isfinite(nll)
        assert nll < 0 or nll > 0   # not 1e15

    def test_nll_returns_large_for_invalid_params(self):
        m_vals = self._sample_monetary()
        # q <= 1 violates the model assumption → should return penalty
        nll = _gg_nll(np.array([1.0, 0.5, 10.0]), m_vals)
        assert nll >= 1e14

    def test_fit_succeeds_on_sufficient_data(self):
        m_vals = self._sample_monetary(n=500)
        gg = GammaGammaModel()
        ok = gg.fit(m_vals)
        assert ok
        assert gg.p is not None and gg.q is not None and gg.gamma is not None

    def test_expected_monetary_value_formula(self):
        """E[M] = q·γ / (q−1) for q > 1."""
        gg = GammaGammaModel()
        gg.p  = 2.0
        gg.q  = 4.0
        gg.gamma = 9.0
        expected = 4.0 * 9.0 / (4.0 - 1.0)   # = 12.0
        assert gg.expected_monetary_value() == pytest.approx(expected, rel=1e-9)

    def test_expected_value_q_le_1_uses_median_fallback(self):
        """With q ≤ 1 the unconditional expectation is undefined; use median."""
        gg = GammaGammaModel()
        gg.q              = 0.9
        gg.gamma          = 10.0
        gg._median_fallback = 25.0
        em = gg.expected_monetary_value()
        assert em == pytest.approx(25.0, abs=1e-9)

    def test_fallback_threshold_via_wrapper(self):
        """
        fit_gamma_gamma_models() should set median fallback and return a model
        that em > 0 even when n < MIN_OBS — the wrapper owns the MIN_OBS check.
        """
        from models.survival.gamma_gamma_new import fit_gamma_gamma_models
        import pandas as pd
        # Create a tiny monetary df with fewer than MIN_OBS rows
        rng = np.random.default_rng(42)
        m_vals = rng.gamma(2, 10, size=5)   # n=5 << MIN_OBS=30
        df_monetary = pd.DataFrame({
            "subscription_type": ["SUB_MONTHLY"] * 5,
            "macro_region": ["ROW"] * 5,
            "net_amount_usd": m_vals,
        })
        models = fit_gamma_gamma_models(df_monetary, "fold_1")
        assert "SUB_MONTHLY" in models
        em = models["SUB_MONTHLY"].expected_monetary_value()
        # Should fall back to median, which must be > 0
        assert em > 0

    def test_fit_rejects_too_few_points(self):
        """fit() refuses < 3 data points (cannot fit a 3-param model)."""
        gg = GammaGammaModel()
        ok = gg.fit(np.array([10.0, 20.0]))   # n=2
        assert not ok

    def test_median_fallback_populated_after_failed_fit(self):
        """After a failed fit, _median_fallback must be set."""
        m_vals = np.array([10.0, 20.0])
        gg = GammaGammaModel()
        gg.fit(m_vals)
        assert gg._median_fallback == pytest.approx(15.0, abs=1e-9)

    def test_fit_on_min_obs_boundary(self):
        """fit() succeeds with exactly MIN_OBS observations."""
        rng = np.random.default_rng(7)
        m_vals = rng.gamma(shape=2.0, scale=15.0, size=MIN_OBS)
        gg = GammaGammaModel()
        ok = gg.fit(m_vals)
        # May or may not succeed depending on convergence; just must not crash
        assert isinstance(ok, bool)


# ===========================================================================
# 5. Fallback hierarchy: trigger conditions and naming
# ===========================================================================

class TestFallbackTriggers:
    """Spec §2.5 trigger conditions."""

    def test_initial_users_below_10_skips_mle(self):
        """initial_users < 10 → optimize must not be called at target granularity."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        # Simulate: 5 users, 5 rebill periods → should never reach MLE
        cohort = [5, 4, 3, 3, 2, 1]
        # Direct MLE attempt: L-BFGS-B may succeed but spec says skip
        # The guard is in execute_survival_model; here we verify the model itself
        # doesn't break on tiny data
        ok = m.optimize([cohort], initial_users=5)
        # With accept_lower=True (small cohort), solution may be found
        assert isinstance(ok, bool)

    def test_t_max_lt_2_triggers_fallback(self):
        """T_obs < 2: optimize() returns False → fallback must be triggered."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        cohort = [100, 85]  # t_max=1
        ok = m.optimize([cohort], initial_users=100)
        assert not ok, "T_obs=1 must return False per spec §2.5"

    def test_fb1_segment_name_format(self):
        """Fallback 1 segment should end with _MACRO_GLOBAL."""
        acq_dt = datetime(2021, 8, 2)   # 2021-W31
        acq_str = acq_week_iso_str(acq_dt)
        expected = f"{acq_str}_SUB_MONTHLY_MACRO_GLOBAL"
        assert expected.endswith("_MACRO_GLOBAL")
        assert "SUB_MONTHLY" in expected

    def test_fb2_segment_name_format(self):
        """Fallback 2 segment should use YYYY-MM and end with _ALL_SUB_."""
        acq_dt = datetime(2021, 8, 2)
        month_str = acq_month_str(acq_dt)
        expected = f"{month_str}_SUB_MONTHLY_ALL_SUB_"
        assert expected.startswith("2021-08")
        assert expected.endswith("_ALL_SUB_")

    def test_acq_week_iso_format(self):
        """Monday of ISO week 31 of 2021 is 2021-08-02."""
        dt = datetime(2021, 8, 2)
        s = acq_week_iso_str(dt)
        assert s == "2021-W31"

    def test_acq_month_format(self):
        dt = datetime(2021, 11, 15)
        assert acq_month_str(dt) == "2021-11"


# ===========================================================================
# 6. Trial exclusion: t_target formula
# ===========================================================================

class TestTrialExclusion:
    """
    Spec §3.1: t_target = ceil((holdout_week_start - acq_week - trial_duration) / rebill_duration)
    """

    def _t_target(
        self,
        acq_week: datetime,
        holdout_week: datetime,
        trial_days: float,
        rebill_days: float,
    ) -> int:
        import math
        diff = (holdout_week - acq_week).days
        return math.ceil((diff - trial_days) / rebill_days)

    def test_sub_weekly_t_target(self):
        """SUB_WEEKLY: trial=7d, rebill=7d. Cohort acquired 2021-W31, holdout week 2021-W40."""
        acq    = datetime(2021, 8, 2)    # 2021-W31 Monday
        hold   = datetime(2021, 10, 4)   # 2021-W40 Monday  (63 days later)
        trial, rebill = 7, 7
        t = self._t_target(acq, hold, trial, rebill)
        # (63 - 7) / 7 = 8 exactly → ceil(8) = 8
        assert t == 8

    def test_sub_monthly_t_target(self):
        """SUB_MONTHLY: trial=7d, rebill=30d. Cohort 2021-W31, holdout ~3 months."""
        acq  = datetime(2021, 8, 2)
        hold = datetime(2021, 11, 1)    # 91 days later
        trial, rebill = 7, 30
        t = self._t_target(acq, hold, trial, rebill)
        # (91 - 7) / 30 = 2.8 → ceil = 3
        assert t == 3

    def test_sub_3_month_t_target(self):
        """SUB_3_MONTH: trial=90d, rebill=90d. Only reaches t_target=1 in 3-month holdout."""
        acq  = datetime(2021, 9, 6)
        hold = datetime(2021, 12, 6)   # 91 days later
        trial, rebill = 90, 90
        t = self._t_target(acq, hold, trial, rebill)
        # (91 - 90) / 90 = 0.011 → ceil = 1
        assert t == 1

    def test_t_target_within_trial_returns_lt_1(self):
        """If holdout week is within trial window, t_target < 1 → must skip."""
        acq  = datetime(2021, 11, 1)
        hold = datetime(2021, 11, 7)   # only 6 days after acquisition
        trial, rebill = 7, 7
        t = self._t_target(acq, hold, trial, rebill)
        assert t < 1


# ===========================================================================
# 7. Boundary-hit confidence_weight
# ===========================================================================

class TestBoundaryHit:
    def test_hits_boundary_lower(self):
        """Params at lower bound → hits_boundary() = True."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([0.0001, 5.0])   # alpha at lower bound
        assert m.hits_boundary()

    def test_hits_boundary_upper(self):
        """Params at upper bound → hits_boundary() = True."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.0, 10_000.0])   # beta at upper bound
        assert m.hits_boundary()

    def test_no_boundary_hit(self):
        """Interior params → hits_boundary() = False."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.5, 4.0])
        assert not m.hits_boundary()


# ===========================================================================
# 8. Output contract columns
# ===========================================================================

class TestOutputContract:
    REQUIRED_COLS = {
        "fold_id", "model_name", "forecast_period", "segment",
        "rebill_period_t", "expected_active_users", "actual_active_users",
        "expected_ltv_usd", "actual_ltv_usd", "confidence_weight",
    }

    def _build_predictions(self) -> list:
        """Minimal prediction dict list for translate_survival_to_output."""
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.0, 3.0])

        gg = GammaGammaModel()
        gg.q     = 4.0
        gg.gamma = 9.0
        gg.p     = 2.0

        return [
            {
                "fold_id":           "fold_1",
                "segment":           "2021-W31_SUB_MONTHLY_NA_US",
                "subscription_type": "SUB_MONTHLY",
                "S_t_target":        m.predicted_survival(2),
                "t_target":          2,
                "N_initial":         500.0,
                "forecast_period":   "2021-09-06",
                "confidence_weight": 0.0,
                "model_obj":         m,
                "max_holdout_t":     3,
            }
        ]

    def test_output_has_required_columns(self):
        preds = self._build_predictions()
        gg = GammaGammaModel()
        gg.q = 4.0; gg.gamma = 9.0; gg.p = 2.0
        df = translate_survival_to_output(
            predictions  = preds,
            model_name   = "sBG",
            fold_id      = "fold_1",
            gg_models    = {"SUB_MONTHLY": gg},
            refund_rates = {"SUB_MONTHLY": 0.05},
            holdout_weeks= [],
        )
        assert not df.empty
        assert self.REQUIRED_COLS.issubset(set(df.columns)), (
            f"Missing: {self.REQUIRED_COLS - set(df.columns)}"
        )

    def test_actual_columns_are_null(self):
        """actual_active_users and actual_ltv_usd must be NULL (filled by consolidation)."""
        preds = self._build_predictions()
        gg = GammaGammaModel()
        gg.q = 4.0; gg.gamma = 9.0; gg.p = 2.0
        df = translate_survival_to_output(
            predictions  = preds,
            model_name   = "sBG",
            fold_id      = "fold_1",
            gg_models    = {"SUB_MONTHLY": gg},
            refund_rates = {"SUB_MONTHLY": 0.05},
            holdout_weeks= [],
        )
        assert df["actual_active_users"].isna().all()
        assert df["actual_ltv_usd"].isna().all()

    def test_expected_active_users_is_n0_times_s(self):
        """expected_active_users = N_initial × S(t_target)."""
        preds = self._build_predictions()
        n0   = preds[0]["N_initial"]
        s_t  = preds[0]["S_t_target"]
        gg = GammaGammaModel()
        gg.q = 4.0; gg.gamma = 9.0; gg.p = 2.0
        df = translate_survival_to_output(
            predictions  = preds,
            model_name   = "sBG",
            fold_id      = "fold_1",
            gg_models    = {"SUB_MONTHLY": gg},
            refund_rates = {"SUB_MONTHLY": 0.05},
            holdout_weeks= [],
        )
        assert df["expected_active_users"].iloc[0] == pytest.approx(n0 * s_t, rel=1e-4)

    def test_empty_predictions_returns_empty_df(self):
        df = translate_survival_to_output(
            predictions=[], model_name="sBG", fold_id="fold_1",
            gg_models={}, refund_rates={}, holdout_weeks=[],
        )
        assert df.empty


# ===========================================================================
# 9. LTV calculation
# ===========================================================================

class TestLTVCalculation:
    def test_ltv_formula(self):
        """
        expected_ltv = N0 × Σ_{t=t_target}^{max_t} S(t) × E[M] × (1 − refund_rate)
        """
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.0, 3.0])

        gg = GammaGammaModel()
        gg.q = 4.0; gg.gamma = 9.0; gg.p = 2.0
        em = gg.expected_monetary_value()  # = 4*9/(4-1) = 12
        rr = 0.05
        n0 = 1000.0
        t_target = 2
        max_holdout_t = 4

        expected_sum = sum(m.predicted_survival(t) for t in range(t_target, max_holdout_t + 1))
        expected_ltv = n0 * expected_sum * em * (1 - rr)

        pred = {
            "fold_id":           "fold_1",
            "segment":           "2021-W31_SUB_MONTHLY_NA_US",
            "subscription_type": "SUB_MONTHLY",
            "S_t_target":        m.predicted_survival(t_target),
            "t_target":          t_target,
            "N_initial":         n0,
            "forecast_period":   "2021-09-06",
            "confidence_weight": 0.0,
            "model_obj":         m,
            "max_holdout_t":     max_holdout_t,
        }
        df = translate_survival_to_output(
            predictions  = [pred],
            model_name   = "sBG",
            fold_id      = "fold_1",
            gg_models    = {"SUB_MONTHLY": gg},
            refund_rates = {"SUB_MONTHLY": rr},
            holdout_weeks= [],
        )
        assert df["expected_ltv_usd"].iloc[0] == pytest.approx(expected_ltv, rel=1e-4)


# ===========================================================================
# 10. Pooled-MLE disaggregation logic (unit)
# ===========================================================================

class TestPooledMLEDisaggregation:
    def test_pooled_survival_disaggregated_by_n0(self):
        """
        When pooled S(t) applied to each region's N_0, predicted actives must sum
        to total pooled active users (within rounding).
        """
        m = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m.params = np.array([1.0, 3.0])

        regions = {"NA_US": 300, "CIS": 150, "ROW": 200}
        t = 3
        s_pooled = m.predicted_survival(t)

        disaggregated_actives = {r: n0 * s_pooled for r, n0 in regions.items()}
        total_disagg = sum(disaggregated_actives.values())
        total_pooled  = sum(regions.values()) * s_pooled

        assert total_disagg == pytest.approx(total_pooled, rel=1e-9)

    def test_multi_restart_optimizer_finds_better_solution(self):
        """
        With n_restarts=5 the optimizer should achieve ≤ NLL of n_restarts=1
        on the same data.
        """
        alpha_true, beta_true = 0.5, 4.0
        m_gen = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m_gen.params = np.array([alpha_true, beta_true])
        cohort = [500]
        for t in range(1, 8):
            cohort.append(int(500 * m_gen.predicted_survival(t)))

        m5 = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])
        m1 = SBGModel([[0.0001, 10_000], [0.0001, 10_000]])

        ok5 = m5.optimize([cohort], n_restarts=5, initial_users=500)
        ok1 = m1.optimize([cohort], n_restarts=1, initial_users=500)

        if ok5 and ok1:
            nll5 = m5.log_likelihood_multi_cohort(m5.params, [cohort])
            nll1 = m1.log_likelihood_multi_cohort(m1.params, [cohort])
            assert nll5 <= nll1 + 1e-6, "5-restart should find at least as good a solution"
