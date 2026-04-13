"""BSTS MMM rewrite for task #9."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from models.mmm.channel_contribs_new import (
    build_contrib_rows,
    replicate_global_shares_to_regions,
    shares_from_weekly_contribs,
)
from models.mmm.common_new import (
    ALPHA_MEAN_BY_TIER,
    CHANNEL_NAMES,
    EXOG_COLUMNS,
    EXOG_PRIORS,
    FUNNEL_TIERS,
    SPEND_COLUMNS,
    aggregate_to_global,
    check_fallback_needed,
    disaggregate_global_eval_rows,
    fetch_shapley_weights,
    fit_spend_scalers,
    get_bq_client,
    get_ppc_artifact_path,
    get_trace_artifact_path,
    hill,
    load_mmm_data,
    segment_for_region,
    transform_spend_with_scalers,
)
from models.mmm.persistence_new import (
    build_eval_mmm_row,
    write_eval_mmm_to_bq,
    write_mmm_channel_contribs_to_bq,
)

logger = logging.getLogger(__name__)

MODEL_NAME_A = "MMM_BSTS_Heuristic"
MODEL_NAME_B = "MMM_BSTS_DDA"

_PILOT_DRAWS = 250
_PILOT_TUNE = 250
_PILOT_CHAINS = 2
_PILOT_MAX_DIVERGENCES = 20

_DRAWS = 400
_TUNE = 500
_CHAINS = 2
_TARGET_ACCEPT = 0.95
_RHAT_THRESHOLD = 1.1


@dataclass
class FitResult:
    eval_rows: list[dict[str, Any]]
    channel_shares: dict[str, float]
    converged: bool


def _alpha_means_for_scenario(scenario: str) -> list[float]:
    """Return pre-compute AdStock alpha means per channel and scenario."""
    if scenario == "A":
        return [0.5] * len(CHANNEL_NAMES)

    means: list[float] = []
    for channel in CHANNEL_NAMES:
        tier = FUNNEL_TIERS.get(channel, "mid")
        means.append(float(ALPHA_MEAN_BY_TIER.get(tier, 0.5)))
    return means


def _build_model(
    x_media: np.ndarray,
    x_exog: np.ndarray,
    y_z: np.ndarray,
    scenario: str,
    shapley_weights: dict[str, float] | None,
    c_scale_z: float,
) -> pm.Model:
    """Construct BSTS model with LLT latent state + MMM regression block."""
    t_steps, n_channels = x_media.shape

    coords = {
        "channel": CHANNEL_NAMES,
        "exog": EXOG_COLUMNS,
        "time": np.arange(t_steps),
    }

    gamma_mu = np.array([EXOG_PRIORS[column][0] for column in EXOG_COLUMNS], dtype=float)
    gamma_sigma = np.array([EXOG_PRIORS[column][1] for column in EXOG_COLUMNS], dtype=float)

    with pm.Model(coords=coords) as model:
        # Strongly regularized LLT priors to avoid posterior funnels in short panels.
        sigma_level = pm.HalfNormal("sigma_level", sigma=0.15)
        sigma_slope = pm.HalfNormal("sigma_slope", sigma=0.01)

        mu_init = pm.Normal("mu_init", mu=0.0, sigma=0.5)
        nu_init = pm.Normal("nu_init", mu=0.0, sigma=0.03)

        slope_raw = pm.Normal("slope_raw", mu=0.0, sigma=1.0, shape=t_steps - 1)
        slope_innov = slope_raw * sigma_slope
        nu = pm.Deterministic(
            "nu",
            pm.math.concatenate([
                pm.math.stack([nu_init]),
                nu_init + pm.math.cumsum(slope_innov),
            ]),
            dims="time",
        )

        level_raw = pm.Normal("level_raw", mu=0.0, sigma=1.0, shape=t_steps - 1)
        level_innov = level_raw * sigma_level
        mu_steps = nu[:-1] + level_innov
        mu = pm.Deterministic(
            "mu",
            pm.math.concatenate([
                pm.math.stack([mu_init]),
                mu_init + pm.math.cumsum(mu_steps),
            ]),
            dims="time",
        )

        k = pm.Beta("K", alpha=2, beta=2, dims="channel")
        s = pm.Gamma("S", alpha=3, beta=1, dims="channel")

        x_media_data = pm.Data("X_media", x_media)
        x_exog_data = pm.Data("X_exog", x_exog)

        hill_transformed = 1.0 / (1.0 + (k / x_media_data) ** s)

        if scenario == "B" and shapley_weights is not None:
            beta_mu = np.array([
                float(shapley_weights.get(channel, 1.0 / len(CHANNEL_NAMES))) * c_scale_z
                for channel in CHANNEL_NAMES
            ])
            beta = pm.TruncatedNormal(
                "beta",
                mu=beta_mu,
                sigma=0.75,
                lower=0.0,
                dims="channel",
            )
        else:
            beta = pm.HalfNormal("beta", sigma=1.0, dims="channel")

        gamma = pm.Normal("gamma", mu=gamma_mu, sigma=gamma_sigma, dims="exog")

        media_contrib = pm.math.dot(hill_transformed, beta)
        exog_contrib = pm.math.dot(x_exog_data, gamma)

        expected_y = mu + media_contrib + exog_contrib
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)

        y_obs = pm.Data("y_obs", y_z)
        pm.Normal("Y_obs", mu=expected_y, sigma=sigma_obs, observed=y_obs, dims="time")

    return model


def _thin_trace(trace: az.InferenceData, max_total_samples: int = 1000) -> az.InferenceData:
    """Deterministically thin inference data before artifact persistence."""
    draws = int(trace.posterior.sizes.get("draw", 0))
    chains = int(trace.posterior.sizes.get("chain", 1))
    if draws <= 0:
        return trace

    target_draws_per_chain = max(1, max_total_samples // max(chains, 1))
    if draws <= target_draws_per_chain:
        return trace

    stride = int(math.ceil(draws / target_draws_per_chain))
    return trace.isel(draw=slice(0, None, stride))


def _rhat_ok(trace: az.InferenceData) -> bool:
    """Return True when convergence diagnostics pass threshold."""
    summary = az.summary(
        trace,
        var_names=["beta", "K", "S", "sigma_level", "sigma_slope"],
        round_to=4,
    )
    if summary.empty or "r_hat" not in summary.columns:
        return False

    finite_rhat = summary["r_hat"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_rhat.empty:
        return False

    max_rhat = float(finite_rhat.max())
    if max_rhat > _RHAT_THRESHOLD:
        logger.warning("Convergence failure: max r_hat=%.4f > %.2f", max_rhat, _RHAT_THRESHOLD)
        return False
    return True


def _count_divergences(trace: az.InferenceData) -> int:
    """Return total divergence count from sample stats."""
    if "diverging" not in trace.sample_stats:
        return 0
    return int(np.asarray(trace.sample_stats["diverging"].values, dtype=int).sum())


def _prepare_media_features(train_spend: np.ndarray, holdout_spend: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Apply AdStock and train-window scaling to media features."""
    scalers = fit_spend_scalers(train_spend=train_spend, alphas=alphas)
    x_train = transform_spend_with_scalers(train_spend, alphas=alphas, scalers=scalers)
    x_holdout = transform_spend_with_scalers(holdout_spend, alphas=alphas, scalers=scalers)
    return x_train, x_holdout


def _posterior_outputs(
    trace: az.InferenceData,
    x_media_holdout: np.ndarray,
    x_exog_holdout: np.ndarray,
    y_location: float,
    y_scale: float,
) -> dict[str, Any]:
    """Compute holdout posterior means for BSTS forecast and decomposition."""
    posterior = trace.posterior
    n_channels = len(CHANNEL_NAMES)

    beta = posterior["beta"].values.reshape(-1, n_channels)
    k = posterior["K"].values.reshape(-1, n_channels)
    s = posterior["S"].values.reshape(-1, n_channels)
    gamma = posterior["gamma"].values.reshape(-1, len(EXOG_COLUMNS))

    sigma_level = posterior["sigma_level"].values.reshape(-1)
    sigma_slope = posterior["sigma_slope"].values.reshape(-1)
    sigma_obs = posterior["sigma_obs"].values.reshape(-1)

    mu_last = posterior["mu"].isel(time=-1).values.reshape(-1)
    nu_last = posterior["nu"].isel(time=-1).values.reshape(-1)

    n_samples = mu_last.shape[0]
    t_holdout = x_media_holdout.shape[0]

    rng = np.random.default_rng(42)
    mu_forecast = np.empty((n_samples, t_holdout), dtype=float)

    mu_curr = mu_last.copy()
    nu_curr = nu_last.copy()
    for t_idx in range(t_holdout):
        zeta = rng.normal(loc=0.0, scale=sigma_slope)
        eta = rng.normal(loc=0.0, scale=sigma_level)
        mu_next = mu_curr + nu_curr + eta
        nu_next = nu_curr + zeta

        mu_forecast[:, t_idx] = mu_next
        mu_curr = mu_next
        nu_curr = nu_next

    hill_values = hill(
        x_media_holdout[np.newaxis, :, :],
        k[:, np.newaxis, :],
        s[:, np.newaxis, :],
    )
    media_channel = beta[:, np.newaxis, :] * hill_values
    media_total = media_channel.sum(axis=2)

    exog_contrib = np.einsum("te,se->st", x_exog_holdout, gamma)
    base_component = mu_forecast + exog_contrib

    mu = base_component + media_total
    ppc = rng.normal(loc=mu, scale=sigma_obs[:, np.newaxis])

    ppc_raw = y_location + y_scale * ppc
    expected_mean_raw = ppc_raw.mean(axis=0)
    base_mean_raw = y_location + y_scale * base_component.mean(axis=0)
    media_weekly_raw = y_scale * media_channel.mean(axis=0)

    return {
        "ppc": ppc_raw,
        "expected_mean": expected_mean_raw,
        "base_mean": base_mean_raw,
        "media_weekly_channel_mean": media_weekly_raw,
        "mean_saturation_point": float(k.mean(axis=1).mean()),
    }


def _fit_segment(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    region_label: str,
    fold_id: str,
    confidence_weight: int,
    scenario: str,
    prior_source: str,
    model_name: str,
    shapley_weights: dict[str, float] | None,
) -> FitResult | None:
    """Fit one segment and return eval rows + modeled channel shares."""
    spend_train = train_df[SPEND_COLUMNS].to_numpy(dtype=float)
    spend_holdout = holdout_df[SPEND_COLUMNS].to_numpy(dtype=float)
    exog_train = train_df[EXOG_COLUMNS].to_numpy(dtype=float)
    exog_holdout = holdout_df[EXOG_COLUMNS].to_numpy(dtype=float)
    y_train_raw = train_df["total_net_revenue_usd"].to_numpy(dtype=float)
    y_location = float(np.mean(y_train_raw))
    y_scale = max(float(np.std(y_train_raw)), 1.0)
    y_train_z = (y_train_raw - y_location) / y_scale

    # Translate Scenario-B scaling constant into standardized target space.
    c_scale_raw = y_location / max(len(CHANNEL_NAMES), 1)
    c_scale_z = c_scale_raw / y_scale

    alphas = _alpha_means_for_scenario(scenario)
    x_media_train, x_media_holdout = _prepare_media_features(spend_train, spend_holdout, alphas)

    model = _build_model(
        x_media=x_media_train,
        x_exog=exog_train,
        y_z=y_train_z,
        scenario=scenario,
        shapley_weights=shapley_weights,
        c_scale_z=c_scale_z,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with model:
                pilot_trace = pm.sample(
                    draws=_PILOT_DRAWS,
                    tune=_PILOT_TUNE,
                    chains=_PILOT_CHAINS,
                    target_accept=_TARGET_ACCEPT,
                    random_seed=42,
                    progressbar=False,
                    return_inferencedata=True,
                    cores=min(_PILOT_CHAINS, 2),
                    compute_convergence_checks=False,
                )
        pilot_div = _count_divergences(pilot_trace)
        pilot_ok = _rhat_ok(pilot_trace) and pilot_div <= _PILOT_MAX_DIVERGENCES
        if not pilot_ok:
            logger.warning(
                "[%s][%s] Pilot diagnostics failed (divergences=%d, max_allowed=%d); falling back.",
                region_label,
                model_name,
                pilot_div,
                _PILOT_MAX_DIVERGENCES,
            )
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with model:
                trace = pm.sample(
                    draws=_DRAWS,
                    tune=_TUNE,
                    chains=_CHAINS,
                    target_accept=_TARGET_ACCEPT,
                    random_seed=42,
                    progressbar=False,
                    return_inferencedata=True,
                    cores=min(_CHAINS, 2),
                    compute_convergence_checks=False,
                )
    except Exception as exc:  # pragma: no cover - sampler runtime path
        logger.error("[%s][%s] sampling failed: %s", region_label, model_name, exc)
        return None

    divergences = _count_divergences(trace)
    converged = _rhat_ok(trace)
    if divergences > _PILOT_MAX_DIVERGENCES:
        logger.warning(
            "[%s][%s] Full sampling divergences=%d > %d; treating as non-converged.",
            region_label,
            model_name,
            divergences,
            _PILOT_MAX_DIVERGENCES,
        )
        converged = False

    posterior = _posterior_outputs(
        trace,
        x_media_holdout=x_media_holdout,
        x_exog_holdout=exog_holdout,
        y_location=y_location,
        y_scale=y_scale,
    )

    # Artifact persistence: always overwrite (no stale-file guard).
    trace_path = get_trace_artifact_path(model_name=model_name, fold_id=fold_id)
    ppc_path = get_ppc_artifact_path(model_name=model_name, fold_id=fold_id)
    _thin_trace(trace).to_netcdf(trace_path)
    np.save(ppc_path, posterior["ppc"])

    segment = segment_for_region(region_label)
    eval_rows: list[dict[str, Any]] = []
    holdout_dates = holdout_df["date_week"].reset_index(drop=True)

    for idx, date_week in enumerate(holdout_dates):
        eval_rows.append(
            build_eval_mmm_row(
                fold_id=fold_id,
                model_name=model_name,
                forecast_period=date_week,
                segment=segment,
                expected_net_revenue_usd=max(float(posterior["expected_mean"][idx]), 0.0),
                base_sales_intercept=float(posterior["base_mean"][idx]),
                mean_saturation_point=float(posterior["mean_saturation_point"]),
                prior_source=prior_source,
                confidence_weight=confidence_weight,
            )
        )

    media_weekly = posterior["media_weekly_channel_mean"]
    weekly_contribs = pd.DataFrame(
        {
            "gads:search": media_weekly[:, 0],
            "gads:youtube": media_weekly[:, 1],
            "gads:discover": media_weekly[:, 2],
            "metads:inst": media_weekly[:, 3],
            "metads:fb": media_weekly[:, 4],
            "tiktok": media_weekly[:, 5],
        }
    )
    shares = shares_from_weekly_contribs(
        weekly_contribs=weekly_contribs,
        fold_id=fold_id,
        model_name=model_name,
        segment=segment,
    )

    return FitResult(eval_rows=eval_rows, channel_shares=shares, converged=converged)


def _run_scenario(
    client,
    bq_project: str,
    bq_dataset: str,
    fold_id: str,
    train_all: pd.DataFrame,
    holdout_all: pd.DataFrame,
    scenario: str,
    shapley_weights: dict[str, float] | None,
) -> None:
    """Run one BSTS prior scenario and persist outputs."""
    is_scenario_b = scenario == "B"
    model_name = MODEL_NAME_B if is_scenario_b else MODEL_NAME_A

    if is_scenario_b and shapley_weights is None:
        prior_source = "heuristic_fallback"
    elif is_scenario_b:
        prior_source = "shapley_dda"
    else:
        prior_source = "heuristic"

    scenario_weights = shapley_weights if is_scenario_b else None

    eval_rows: list[dict[str, Any]] = []
    shares_by_segment: dict[str, dict[str, float]] = {}
    fallback_regions: list[str] = []

    regions = sorted(train_all["macro_region"].dropna().unique().tolist())
    for region in regions:
        tr = train_all[train_all["macro_region"] == region].sort_values("date_week")
        ho = holdout_all[holdout_all["macro_region"] == region].sort_values("date_week")

        if ho.empty:
            continue

        if check_fallback_needed(tr):
            fallback_regions.append(region)
            continue

        fit = _fit_segment(
            train_df=tr,
            holdout_df=ho,
            region_label=region,
            fold_id=fold_id,
            confidence_weight=0,
            scenario=scenario,
            prior_source=prior_source,
            model_name=model_name,
            shapley_weights=scenario_weights,
        )
        if fit is None or not fit.converged:
            fallback_regions.append(region)
            continue

        eval_rows.extend(fit.eval_rows)
        shares_by_segment[segment_for_region(region)] = fit.channel_shares

    uncovered = [region for region in fallback_regions if segment_for_region(region) not in shares_by_segment]
    if uncovered:
        global_fit = _fit_segment(
            train_df=aggregate_to_global(train_all),
            holdout_df=aggregate_to_global(holdout_all),
            region_label="Global",
            fold_id=fold_id,
            confidence_weight=-1,
            scenario=scenario,
            prior_source=prior_source,
            model_name=model_name,
            shapley_weights=scenario_weights,
        )

        if global_fit is not None and global_fit.converged:
            disagg_eval = disaggregate_global_eval_rows(
                global_eval_rows=pd.DataFrame(global_fit.eval_rows),
                train_df=train_all,
                target_regions=uncovered,
            )
            eval_rows.extend(disagg_eval.to_dict("records"))
            shares_by_segment.update(
                replicate_global_shares_to_regions(global_fit.channel_shares, uncovered)
            )
        else:
            logger.warning("%s fold=%s: global fallback failed for regions=%s", model_name, fold_id, uncovered)

    if not eval_rows:
        logger.warning("%s fold=%s: no rows generated.", model_name, fold_id)
        return

    eval_df = pd.DataFrame(eval_rows)
    write_eval_mmm_to_bq(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        model_name=model_name,
        fold_id=fold_id,
        eval_rows=eval_df,
    )

    contrib_rows = build_contrib_rows(
        fold_id=fold_id,
        model_name=model_name,
        shares_by_segment=shares_by_segment,
    )
    write_mmm_channel_contribs_to_bq(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        model_name=model_name,
        fold_id=fold_id,
        contrib_rows=pd.DataFrame(contrib_rows),
    )

    logger.info(
        "%s fold=%s complete. eval_rows=%d contrib_rows=%d",
        model_name,
        fold_id,
        len(eval_df),
        len(contrib_rows),
    )


def run(
    bq_project: str,
    bq_dataset: str,
    fold_id: str,
    train_end: str,
    holdout_start: str,
    holdout_end: str,
) -> None:
    """Run BSTS MMM scenarios A/B and persist eval + incr contributions."""
    client = get_bq_client()

    train_all = load_mmm_data(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
        train_end=train_end,
    )
    holdout_all = load_mmm_data(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )

    if train_all.empty:
        logger.error("BSTS MMM fold=%s: empty training data, skipping.", fold_id)
        return

    shapley_weights = fetch_shapley_weights(
        client=client,
        project=bq_project,
        dataset=bq_dataset,
        fold_id=fold_id,
    )

    _run_scenario(
        client=client,
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        fold_id=fold_id,
        train_all=train_all,
        holdout_all=holdout_all,
        scenario="A",
        shapley_weights=None,
    )
    _run_scenario(
        client=client,
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        fold_id=fold_id,
        train_all=train_all,
        holdout_all=holdout_all,
        scenario="B",
        shapley_weights=shapley_weights,
    )
