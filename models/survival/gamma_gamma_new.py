"""
gamma_gamma_new.py
==================
Discrete Gamma-Gamma monetary model for subscription LTV estimation.
Spec reference: docs/algorithms/survival_models.md §3.2

Adaptation
----------
The classical Gamma-Gamma model (Fader & Hardie, 2005) is adapted here for a
discrete, contractual setting.  Frequency is already captured by sBG/BdW
survival models.  This module models **per-renewal monetary value only**.

Mathematical Core
-----------------
Per-user average transaction value M̄_i ~ Gamma(p, ν_i),
where ν_i ~ Gamma(q, γ) → Gamma-Gamma mixture.

MLE: minimise NLL over (p, q, γ) using L-BFGS-B.

Expected value (unconditional, cohort-level projection):
    E[M] = q·γ / (q − 1)   [valid when q > 1]

If q ≤ 1 after fitting, fall back to median observed monetary value.

Threshold: fallback to pooled estimates when n_obs < MIN_OBS.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.optimize import minimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)

MIN_OBS = 30          # minimum monetary observations to attempt full MLE
_BOUNDS = [(1e-4, 1e4), (1.001, 1e4), (1e-4, 1e4)]   # (p, q, γ) — q > 1 enforced
_N_RESTARTS = 5


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------

def _gg_nll(params: np.ndarray, m_values: np.ndarray) -> float:
    """
    Negative log-likelihood of the Gamma-Gamma model for observed per-renewal
    monetary values m_values (1-D array of positive floats).

    The marginal distribution of a single observed transaction m_i given
    population parameters (p, q, γ) is:

        f(m | p, q, γ) = [Γ(p+q) / (Γ(p)·Γ(q))] · (m/γ)^p · (1 + m/γ)^-(p+q) · (1/m)

    This is the Beta-prime / Lomax PDF evaluated at m/γ, scaled by 1/γ.
    Log-form for numerical stability:
        ln f = ln Γ(p+q) − ln Γ(p) − ln Γ(q) + p·(ln m − ln γ) − (p+q)·ln(1 + m/γ) − ln m
    """
    p, q, gamma = params
    if p <= 0 or q <= 1.001 or gamma <= 0:
        return 1e15
    m = m_values
    if np.any(m <= 0):
        return 1e15
    try:
        ll = (
            gammaln(p + q) - gammaln(p) - gammaln(q)
            + p * (np.log(m) - np.log(gamma))
            - (p + q) * np.log(1.0 + m / gamma)
            - np.log(m)
        )
        return -np.sum(ll)
    except Exception:
        return 1e15


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class GammaGammaModel:
    """
    Gamma-Gamma monetary model, fitted per subscription_type.
    Stores (p, q, γ) and derives E[M].
    """

    def __init__(self) -> None:
        self.p: Optional[float] = None
        self.q: Optional[float] = None
        self.gamma: Optional[float] = None
        self._median_fallback: Optional[float] = None

    # ------------------------------------------------------------------
    def fit(self, m_values: np.ndarray) -> bool:
        """
        Fit (p, q, γ) via L-BFGS-B with multiple random restarts.
        Caches median as fallback for q ≤ 1 solutions.

        Returns True on successful convergence, False otherwise.
        """
        m_clean = m_values[np.isfinite(m_values) & (m_values > 0)]
        self._median_fallback = float(np.median(m_clean)) if len(m_clean) > 0 else 0.0

        if len(m_clean) < 3:
            return False

        best_res = None
        best_nll = np.inf
        rng = np.random.default_rng(42)

        # Initial guess: method-of-moments approximations
        m_mean = float(np.mean(m_clean))
        m_var  = float(np.var(m_clean)) if float(np.var(m_clean)) > 0 else m_mean
        # Rough MOM: p ~ mean^2/var, q ~ 2.0, γ ~ mean
        x0_mom = np.array([max(m_mean ** 2 / m_var, 0.1), 2.0, max(m_mean, 0.1)])
        candidates = [x0_mom]
        for _ in range(_N_RESTARTS - 1):
            candidates.append(rng.uniform([0.1, 1.05, 0.1], [10.0, 20.0, m_mean * 5]))

        for x0 in candidates:
            res = minimize(
                _gg_nll,
                x0,
                args=(m_clean,),
                bounds=_BOUNDS,
                method="L-BFGS-B",
            )
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_res = res

        if best_res is None:
            logger.warning("GammaGamma: all restarts failed; using median fallback.")
            return False

        self.p, self.q, self.gamma = best_res.x
        return True

    # ------------------------------------------------------------------
    def expected_monetary_value(self) -> float:
        """
        Unconditional E[M] = q·γ / (q − 1)   [cohort-level projection].
        Falls back to median if q ≤ 1 or model not fitted.
        """
        if self.q is not None and self.q > 1.0 and self.gamma is not None:
            return float(self.q * self.gamma / (self.q - 1.0))
        if self._median_fallback is not None:
            logger.warning(
                "GammaGamma: q=%.4f ≤ 1 or not fitted; using median fallback (%.2f).",
                self.q or 0.0,
                self._median_fallback,
            )
            return float(self._median_fallback)
        return 0.0


# ---------------------------------------------------------------------------
# Fetch monetary data from BigQuery
# ---------------------------------------------------------------------------

def fetch_monetary_data(
    client: bigquery.Client,
    project: str,
    dataset: str,
    train_end: str,
) -> pd.DataFrame:
    """
    Load per-renewal monetary values (rebill_number >= 1) from the training
    window.  Returns a DataFrame with columns:
        subscription_type, macro_region, net_amount_usd
    """
    sql = f"""
        SELECT
            p.product_id            AS subscription_type,
            COALESCE(c.region, 'ROW') AS macro_region,
            p.order_amount_in_usd - COALESCE(p.refund_amount_in_usd, 0) AS net_amount_usd
        FROM `{project}.{dataset}.purchases` AS p
        LEFT JOIN `{project}.{dataset}.users_attribution_imputed` AS ua_imputed
            ON p.user_id = ua_imputed.user_id
        LEFT JOIN `{project}.{dataset}.users_attribution` AS ua_raw
            ON p.user_id = ua_raw.user_id
        LEFT JOIN `{project}.{dataset}.countries` AS c
            ON COALESCE(ua_imputed.country_code, ua_raw.country_code) = c.country_code
        WHERE
            p.rebill_number >= 1
            AND p.order_status IN ('approved', 'settled_ok', 'refunded')
            AND CAST(p.order_date AS DATE) < '{train_end}'
    """
    df = client.query(sql).to_dataframe()
    df["net_amount_usd"] = pd.to_numeric(df["net_amount_usd"], errors="coerce").fillna(0)
    return df[df["net_amount_usd"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fit per subscription_type + persist monetary params
# ---------------------------------------------------------------------------

def fit_gamma_gamma_models(
    df_monetary: pd.DataFrame,
    fold_id: str,
    expected_subscription_types: Optional[list[str]] = None,
) -> dict[str, GammaGammaModel]:
    """
    Fit one GG model per subscription_type.

    Fallback: if a (subscription_type, macro_region) pair has fewer than
    MIN_OBS observations, use the subscription_type-level pooled estimates.

    Returns:
        {subscription_type: GammaGammaModel}  — pooled fits per sub type.
    """
    models: dict[str, GammaGammaModel] = {}

    for sub_type, group in df_monetary.groupby("subscription_type"):
        m_values = group["net_amount_usd"].values
        model = GammaGammaModel()
        if len(m_values) >= MIN_OBS:
            success = model.fit(m_values)
        else:
            logger.warning(
                "GammaGamma fold=%s sub_type=%s: only %d obs (< %d) — fallback median.",
                fold_id, sub_type, len(m_values), MIN_OBS,
            )
            success = False

        if not success:
            # Fallback: use median as expected value
            model._median_fallback = float(np.median(m_values)) if len(m_values) else 0.0

        models[str(sub_type)] = model
        logger.info(
            "GammaGamma fold=%s sub_type=%s: p=%.4f q=%.4f γ=%.4f E[M]=%.2f",
            fold_id, sub_type,
            model.p or float("nan"),
            model.q or float("nan"),
            model.gamma or float("nan"),
            model.expected_monetary_value(),
        )

    if expected_subscription_types:
        missing = sorted(set(expected_subscription_types) - set(models.keys()))
        for sub_type in missing:
            logger.warning(
                "GammaGamma fold=%s sub_type=%s: no eligible monetary training rows "
                "(rebill_number>=1, net_amount_usd>0, order_date < train_end).",
                fold_id,
                sub_type,
            )

    return models


# ---------------------------------------------------------------------------
# Persist to survival_monetary_params table
# ---------------------------------------------------------------------------

def persist_monetary_params(
    client: bigquery.Client,
    models: dict[str, GammaGammaModel],
    fold_id: str,
    project: str,
    dataset: str,
    segment_to_sub_type: Optional[dict[str, str]] = None,
) -> None:
    """Idempotent write of GG params to survival_monetary_params."""
    table_fqn = f"{project}.{dataset}.survival_monetary_params"

    # Fold-scoped idempotency
    client.query(
        f"DELETE FROM `{table_fqn}` WHERE fold_id = '{fold_id}'"
    ).result()

    rows = []
    # Contract-aligned mode: persist one row per evaluated/fallback segment.
    if segment_to_sub_type is not None:
        for segment, sub_type in sorted(segment_to_sub_type.items()):
            model = models.get(sub_type)
            if model is None:
                logger.warning(
                    "GammaGamma fold=%s segment=%s sub_type=%s: no fitted monetary model; "
                    "persisting NULL params with expected_arpu=0.0",
                    fold_id,
                    segment,
                    sub_type,
                )
                rows.append({
                    "fold_id":       fold_id,
                    "segment":       segment,
                    "p":             None,
                    "q":             None,
                    "gamma":         None,
                    "expected_arpu": 0.0,
                })
                continue
            rows.append({
                "fold_id":               fold_id,
                "segment":               segment,
                "p":                     model.p,
                "q":                     model.q,
                "gamma":                 model.gamma,
                "expected_arpu":         round(model.expected_monetary_value(), 4),
            })
    else:
        # Backward-compatible mode (legacy): one row per subscription_type.
        for sub_type, model in models.items():
            rows.append({
                "fold_id":               fold_id,
                "segment":               sub_type,
                "p":                     model.p,
                "q":                     model.q,
                "gamma":                 model.gamma,
                "expected_arpu":         round(model.expected_monetary_value(), 4),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    schema = [
        bigquery.SchemaField("fold_id",       "STRING",  mode="REQUIRED"),
        bigquery.SchemaField("segment",        "STRING",  mode="REQUIRED"),
        bigquery.SchemaField("p",              "FLOAT64"),
        bigquery.SchemaField("q",              "FLOAT64"),
        bigquery.SchemaField("gamma",          "FLOAT64"),
        bigquery.SchemaField("expected_arpu",  "FLOAT64"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
    )
    client.load_table_from_dataframe(df, table_fqn, job_config=job_config).result()
    logger.info("GammaGamma fold=%s: wrote %d rows to %s", fold_id, len(df), table_fqn)
