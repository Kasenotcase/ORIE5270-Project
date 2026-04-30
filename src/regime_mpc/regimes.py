"""Market regime modeling utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from regime_mpc.optimization import (
    DEFAULT_COV_RIDGE,
    nearest_psd,
    solve_markowitz,
    solve_mpc,
    weighted_mean_and_cov,
)


DEFAULT_HMM_STATES = 3
DEFAULT_HMM_MAX_ITER = 200
DEFAULT_HMM_RANDOM_STATE = 7
DEFAULT_HMM_MIN_FEATURE_DAYS = 252
DEFAULT_INITIAL_TRAIN_DAYS = 756
DEFAULT_MPC_HORIZON = 3
DEFAULT_TRANSACTION_COST_BPS = 3.5
DEFAULT_REGIME_COST_MULTIPLIERS = np.array([1.0, 1.5, 3.0])
DEFAULT_REGIME_LABELS = ("calm", "transition", "stress")


def fit_hmm_regime_inputs(
    date: pd.Timestamp,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    hmm_min_feature_days: int = DEFAULT_HMM_MIN_FEATURE_DAYS,
    hmm_states: int = DEFAULT_HMM_STATES,
    hmm_max_iter: int = DEFAULT_HMM_MAX_ITER,
    hmm_random_state: int = DEFAULT_HMM_RANDOM_STATE,
    mpc_horizon: int = DEFAULT_MPC_HORIZON,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    regime_cost_multipliers: np.ndarray = DEFAULT_REGIME_COST_MULTIPLIERS,
    regime_labels: tuple[str, ...] = DEFAULT_REGIME_LABELS,
    cov_ridge: float = DEFAULT_COV_RIDGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit an HMM regime model and build MPC inputs.

    Returns
    -------
    mu_path:
        Expected return path with shape ``(horizon, n_assets)``.
    cov_path:
        Covariance path with shape ``(horizon, n_assets, n_assets)``.
    cost_path:
        Transaction-cost path with shape ``(horizon, n_assets)``.
    q_path:
        Regime probability path with shape ``(horizon, n_states)``.
    meta:
        Dictionary with regime probabilities and diagnostics.
    """
    if hmm_states != 3:
        raise ValueError("This implementation expects exactly 3 HMM states.")

    required_features = {"rv_63", "amihud_log", "vix_log", "hy_spread", "mom_63"}
    missing = required_features.difference(features.columns)
    if missing:
        raise ValueError(f"features is missing required columns: {sorted(missing)}")

    regime_cost_multipliers = np.asarray(regime_cost_multipliers, dtype=float)
    if len(regime_cost_multipliers) != hmm_states:
        raise ValueError("regime_cost_multipliers must have length equal to hmm_states.")

    if len(regime_labels) != hmm_states:
        raise ValueError("regime_labels must have length equal to hmm_states.")

    feature_train = features.loc[:date].tail(initial_train_days).dropna()
    train_returns = returns.loc[:date].tail(initial_train_days)

    if len(feature_train) < hmm_min_feature_days or len(train_returns) < hmm_min_feature_days:
        raise ValueError("Insufficient observations for HMM-MPC target.")

    n_assets = returns.shape[1]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_train.values)

    model = GaussianHMM(
        n_components=hmm_states,
        covariance_type="diag",
        n_iter=hmm_max_iter,
        tol=1e-4,
        random_state=hmm_random_state,
        min_covar=1e-3,
    )
    model.fit(x_scaled)
    probs = model.predict_proba(x_scaled)

    columns = list(feature_train.columns)
    state_means = model.means_

    stress_score = (
        state_means[:, columns.index("rv_63")]
        + state_means[:, columns.index("amihud_log")]
        + state_means[:, columns.index("vix_log")]
        + state_means[:, columns.index("hy_spread")]
        - state_means[:, columns.index("mom_63")]
    )

    order = np.argsort(stress_score)
    transmat = model.transmat_[np.ix_(order, order)]
    current_prob = probs[-1, order]

    next_returns = returns.shift(-1)
    paired_dates = feature_train.index.intersection(next_returns.dropna().index)
    paired_dates = paired_dates[paired_dates < date]

    paired_returns = next_returns.loc[paired_dates]
    paired_probs = pd.DataFrame(probs, index=feature_train.index).loc[paired_dates].values

    unconditional_mu = train_returns.mean().values
    unconditional_cov = LedoitWolf().fit(train_returns.values).covariance_
    unconditional_cov = nearest_psd(unconditional_cov + cov_ridge * np.eye(n_assets))

    mu_states_original = []
    cov_states_original = []
    eff_samples_original = []

    for k in range(hmm_states):
        raw_mu, raw_cov, eff_n = weighted_mean_and_cov(paired_returns.values, paired_probs[:, k])
        alpha = eff_n / (eff_n + 252.0)

        if eff_n < 30:
            alpha = 0.0

        mu_k = alpha * raw_mu + (1.0 - alpha) * unconditional_mu
        cov_k = alpha * raw_cov + (1.0 - alpha) * unconditional_cov

        mu_states_original.append(mu_k)
        cov_states_original.append(nearest_psd(cov_k + cov_ridge * np.eye(n_assets)))
        eff_samples_original.append(eff_n)

    mu_states = np.asarray(mu_states_original)[order]
    cov_states = np.asarray(cov_states_original)[order]
    eff_samples = np.asarray(eff_samples_original)[order]

    q_path = []
    mu_path = []
    cov_path = []
    cost_path = []

    q = current_prob.copy()
    base_cost = transaction_cost_bps / 10000.0

    for _h in range(mpc_horizon):
        q = q @ transmat
        q = np.maximum(q, 0.0)
        q = q / q.sum()

        mu_mix = q @ mu_states
        cov_mix = np.zeros((n_assets, n_assets))

        for k in range(hmm_states):
            diff = (mu_states[k] - mu_mix).reshape(-1, 1)
            cov_mix += q[k] * (cov_states[k] + diff @ diff.T)

        cost_multiplier = float(q @ regime_cost_multipliers)

        q_path.append(q.copy())
        mu_path.append(mu_mix)
        cov_path.append(nearest_psd(cov_mix + cov_ridge * np.eye(n_assets)))
        cost_path.append(np.ones(n_assets) * base_cost * cost_multiplier)

    meta = {
        "date": date,
        "prob_calm": float(current_prob[0]),
        "prob_transition": float(current_prob[1]),
        "prob_stress": float(current_prob[2]),
        "next_prob_calm": float(q_path[0][0]),
        "next_prob_transition": float(q_path[0][1]),
        "next_prob_stress": float(q_path[0][2]),
        "dominant_regime": regime_labels[int(np.argmax(current_prob))],
        "next_cost_multiplier": float(q_path[0] @ regime_cost_multipliers),
        "hmm_log_likelihood": float(model.score(x_scaled)),
        "effective_sample_calm": float(eff_samples[0]),
        "effective_sample_transition": float(eff_samples[1]),
        "effective_sample_stress": float(eff_samples[2]),
    }

    return (
        np.asarray(mu_path),
        np.asarray(cov_path),
        np.asarray(cost_path),
        np.asarray(q_path),
        meta,
    )


def make_hmm_mpc_target(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> tuple[Callable[[pd.Timestamp, np.ndarray], tuple[np.ndarray, np.ndarray]], list[dict]]:
    """Create a target-weight function for the HMM-MPC strategy.

    If HMM or MPC fails at a rebalance date, the function falls back to
    a Markowitz target.
    """
    regime_records: list[dict] = []
    n_assets = returns.shape[1]
    base_cost_vector = np.ones(n_assets) * (transaction_cost_bps / 10000.0)

    def target(date: pd.Timestamp, current_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            mu_path, cov_path, cost_path, _q_path, meta = fit_hmm_regime_inputs(
                date=date,
                returns=returns,
                features=features,
                initial_train_days=initial_train_days,
                transaction_cost_bps=transaction_cost_bps,
            )
            target_weights = solve_mpc(current_weights, mu_path, cov_path, cost_path)
            meta["solver_status"] = "hmm_mpc"
            regime_records.append(meta)
            return target_weights, cost_path[0]

        except Exception as exc:
            train = returns.loc[:date].tail(initial_train_days)
            fallback = solve_markowitz(train)
            regime_records.append(
                {
                    "date": date,
                    "prob_calm": np.nan,
                    "prob_transition": np.nan,
                    "prob_stress": np.nan,
                    "next_prob_calm": np.nan,
                    "next_prob_transition": np.nan,
                    "next_prob_stress": np.nan,
                    "dominant_regime": "fallback",
                    "next_cost_multiplier": 1.0,
                    "hmm_log_likelihood": np.nan,
                    "effective_sample_calm": np.nan,
                    "effective_sample_transition": np.nan,
                    "effective_sample_stress": np.nan,
                    "solver_status": f"fallback_markowitz: {type(exc).__name__}",
                }
            )
            return fallback, base_cost_vector

    return target, regime_records