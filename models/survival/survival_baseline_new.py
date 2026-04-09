"""
survival_baseline_new.py
========================
Naive curve-fitting baseline for Survival Analysis.
Spec reference: docs/algorithms/survival_models.md §2.2

Fits quadratic, exponential, and power-law curves to aggregate observed
retention rates, then projects forward on the holdout horizon.  The mean of
the three predictions is taken as the final estimate.

Purpose: demonstrate that deterministic curve-fitting (no heterogeneity) is
systematically inferior to the sBG / BdW frailty models.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List

from models.survival.survival_base_new import LTVModel

logger = logging.getLogger(__name__)


class BaselineSurvivalModel(LTVModel):
    """
    Naive regression baseline.

    Overrides ``optimize()`` entirely — does not use the MLE machinery.
    ``bounds`` is accepted for API compatibility but is unused.
    ``log_likelihood_multi_cohort`` is a no-op stub.
    """

    def log_likelihood_multi_cohort(
        self, params: np.ndarray, data: List[List[int]]
    ) -> float:
        """Not used — satisfies abstract interface only."""
        return 0.0

    # ------------------------------------------------------------------
    def optimize(self, data: List[List[int]], initial_users: int = 0, **kwargs) -> bool:
        """
        Fit quadratic, exponential, and power-law curves to the weighted-average
        aggregate survival rate observed at training periods.

        Returns True on success, False if there is insufficient data.
        """
        if not data:
            return False

        max_t = max(len(cohort) - 1 for cohort in data)
        if max_t < 2:
            return False  # need at least 3 periods (t=0,1,2)

        # Weighted aggregation: larger cohorts contribute proportionally
        totals   = np.zeros(max_t + 1, dtype=float)
        initials = np.zeros(max_t + 1, dtype=float)
        for cohort in data:
            n0 = float(cohort[0])
            if n0 <= 0:
                continue
            for t, val in enumerate(cohort):
                totals[t]   += float(val)
                initials[t] += n0

        S_points: List[float] = []
        for t in range(max_t + 1):
            if initials[t] > 0:
                S_points.append(totals[t] / initials[t])
            else:
                break

        if len(S_points) < 3:
            logger.warning("Baseline: fewer than 3 observation periods — cannot fit.")
            return False

        t_vals = np.arange(len(S_points), dtype=float)
        y_vals = np.array(S_points, dtype=float)

        try:
            # Quadratic: exact algebraic polynomial fit (degree 2)
            coeffs_quad = np.polyfit(t_vals, y_vals, 2)

            # Exponential: log-linearisation  ln(y) = a + b·t
            y_log = np.log(np.maximum(y_vals, 1e-15))
            coeffs_exp = np.polyfit(t_vals, y_log, 1)

            # Power-law: ln(y) = a + b·ln(t+1) → y = exp(a)·(t+1)^b
            t_log = np.log(t_vals + 1.0)
            coeffs_pow = np.polyfit(t_log, y_log, 1)

            # Layout: [quad_a, quad_b, quad_c, exp_a, exp_b, pow_a, pow_b]
            self.params = np.array([
                coeffs_quad[2], coeffs_quad[1], coeffs_quad[0],  # a,b,c
                coeffs_exp[1],  coeffs_exp[0],                   # a,b
                coeffs_pow[1],  coeffs_pow[0],                   # a,b
            ], dtype=float)
            return True

        except Exception as exc:
            logger.warning("Baseline curve fitting failed: %s", exc)
            self.params = None
            return False

    # ------------------------------------------------------------------
    def predicted_survival(self, t: int) -> float:
        """Average of quadratic, exponential, and power-law predictions, clipped to [0, 1]."""
        if self.params is None:
            raise ValueError("BaselineSurvivalModel not fitted — call optimize() first.")

        q_a, q_b, q_c, e_a, e_b, p_a, p_b = self.params

        y_quad = float(q_a + q_b * t + q_c * (t ** 2))
        y_exp  = float(np.exp(e_a + e_b * t))
        y_pow  = float(np.exp(p_a) * ((t + 1.0) ** p_b))

        return float(np.clip((y_quad + y_exp + y_pow) / 3.0, 0.0, 1.0))
