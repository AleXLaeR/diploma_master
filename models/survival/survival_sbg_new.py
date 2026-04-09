"""
survival_sbg_new.py
===================
Shifted-Beta-Geometric (sBG) survival model.
Spec reference: docs/algorithms/survival_models.md §2.3

Model assumption
----------------
Each subscriber has a constant per-period churn probability p drawn from
Beta(α, β) (heterogeneity across the cohort). Discrete renewal cycles.

Key formulas
------------
P(T=1)   = α / (α + β)
P(T=t)   = (β + t − 2) / (α + β + t − 1) × P(T=t-1)   for t > 1
S(t)     = ∏_{i=0}^{t-1} (β+i) / (α+β+i)               S(0)=1
NLL      = −[ Σ s_t ln P(T=t) + n_{T_obs} ln S(T_obs) ]
"""

from __future__ import annotations

import numpy as np
from typing import List

from models.survival.survival_base_new import LTVModel


class SBGModel(LTVModel):
    """
    Shifted-Beta-Geometric model.

    Parameters : [alpha, beta]
    Bounds     : [[0.0001, 10_000], [0.0001, 10_000]]
    """

    def _compute_survival_array(
        self, alpha: float, beta: float, t_max: int
    ) -> np.ndarray:
        """Vectorised S[0..t_max]. S[0]=1, S[t] = S[t-1]*(β+t-1)/(α+β+t-1)."""
        S = np.empty(t_max + 1, dtype=float)
        S[0] = 1.0
        for t in range(1, t_max + 1):
            S[t] = S[t - 1] * (beta + t - 1) / (alpha + beta + t - 1)
        return S

    # ------------------------------------------------------------------
    def log_likelihood_multi_cohort(
        self, params: np.ndarray, data: List[List[int]]
    ) -> float:
        """
        Negative log-likelihood across all cohorts.

        data[c][0]  = N_initial (active_users_at_t=0, i.e. trial subscribers)
        data[c][t]  = active_users_at_t  for t = 1 … T_obs
        """
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e15

        max_t = max(len(c) - 1 for c in data) if data else 0
        if max_t == 0:
            return 1e15

        S = self._compute_survival_array(alpha, beta, max_t)
        # P[t] = S[t-1] - S[t], floored
        P = np.empty(max_t + 1, dtype=float)
        P[0] = 0.0
        for t in range(1, max_t + 1):
            P[t] = max(S[t - 1] - S[t], 1e-15)

        total_n = sum(c[0] for c in data) or 1
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
        """S(t) = ∏_{i=0}^{t-1} (β+i)/(α+β+i).  S(0) = 1."""
        if self.params is None:
            raise ValueError("SBGModel not fitted — call optimize() first.")
        alpha, beta = self.params
        S = 1.0
        for i in range(t):
            S *= (beta + i) / (alpha + beta + i)
        return float(S)
