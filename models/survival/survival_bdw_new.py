"""
survival_bdw_new.py
===================
Beta-Discrete-Weibull (BdW) survival model.
Spec reference: docs/algorithms/survival_models.md §2.4

Extension of sBG with duration-dependence parameter c.
c < 1 → negative duration dependence (churn hazard decreases over time).
c = 1 → exactly degenerates to sBG.
c > 1 → positive duration dependence (rare in subscriptions).

Key formulas
------------
S(t|α,β,c) = B(α, β+t^c) / B(α, β)
           = exp(betaln(α, β+t^c) − betaln(α, β))   [numerically stable]
S(0)       = 1  (since 0^c = 0)
P(T=t)     = S(t-1) − S(t)
NLL        = −[ Σ s_t ln P(T=t) + n_{T_obs} ln S(T_obs) ]
"""

from __future__ import annotations

import numpy as np
from typing import List
from scipy.special import betaln

from models.survival.survival_base_new import LTVModel


class BDWModel(LTVModel):
    """
    Beta-Discrete-Weibull model.

    Parameters : [alpha, beta, c]
    Bounds     : [[0.0001, 10_000], [0.0001, 10_000], [0.0001, 3.0]]
    """

    @staticmethod
    def _s(alpha: float, beta: float, c: float, t: int) -> float:
        """
        Numerically stable S(t) via log Beta-ratio.
        S(0) = 1 (explicit guard avoids 0^c edge cases).
        """
        if t == 0:
            return 1.0
        val = np.exp(betaln(alpha, beta + float(t) ** c) - betaln(alpha, beta))
        return float(np.clip(val, 0.0, 1.0))

    # ------------------------------------------------------------------
    def log_likelihood_multi_cohort(
        self, params: np.ndarray, data: List[List[int]]
    ) -> float:
        """Negative log-likelihood across all cohorts."""
        alpha, beta, c = params
        if alpha <= 0 or beta <= 0 or c <= 0:
            return 1e15

        max_t = max(len(cohort) - 1 for cohort in data) if data else 0
        if max_t == 0:
            return 1e15

        S = np.array([self._s(alpha, beta, c, t) for t in range(max_t + 1)])
        P = np.empty(max_t + 1, dtype=float)
        P[0] = 0.0
        for t in range(1, max_t + 1):
            P[t] = max(S[t - 1] - S[t], 1e-15)

        total_n = sum(cohort[0] for cohort in data) or 1
        ll = 0.0
        for cohort in data:
            t_obs = len(cohort) - 1
            if t_obs < 1:
                continue
            w_c = cohort[0] / total_n
            cohort_ll = 0.0
            for t in range(1, t_obs + 1):
                churn_t = max(0, cohort[t - 1] - cohort[t])
                if churn_t > 0:
                    cohort_ll += churn_t * np.log(P[t])
            n_right = cohort[-1]
            if n_right > 0:
                cohort_ll += n_right * np.log(max(S[t_obs], 1e-15))
            ll += w_c * cohort_ll

        return -ll

    # ------------------------------------------------------------------
    def predicted_survival(self, t: int) -> float:
        """S(t) via numerically stable Beta-function ratio."""
        if self.params is None:
            raise ValueError("BDWModel not fitted — call optimize() first.")
        alpha, beta, c = self.params
        return self._s(alpha, beta, c, t)
