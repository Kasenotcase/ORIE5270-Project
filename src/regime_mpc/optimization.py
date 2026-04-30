"""Optimization utilities for portfolio construction."""

from __future__ import annotations

from typing import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


DEFAULT_GAMMA = 8.0
DEFAULT_WEIGHT_CAP = 0.25
DEFAULT_COV_RIDGE = 1e-6
DEFAULT_MPC_TURNOVER_CAP = 0.75
DEFAULT_MPC_QUADRATIC_TRADE_PENALTY = 1e-3


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric positive semidefinite approximation of a matrix."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, DEFAULT_COV_RIDGE)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _solve_problem_with_fallbacks(problem: cp.Problem, solvers: Iterable[str]) -> bool:
    """Try multiple CVXPY solvers and return whether one succeeds."""
    for solver in solvers:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue

        if problem.status in {"optimal", "optimal_inaccurate"}:
            return True

    return False


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Clip tiny numerical negatives and normalize weights to sum to one."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 0.0)

    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights) / len(weights)

    return weights / total


def solve_markowitz(
    train_returns: pd.DataFrame,
    gamma: float = DEFAULT_GAMMA,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
    cov_ridge: float = DEFAULT_COV_RIDGE,
    solvers: tuple[str, ...] = ("OSQP", "CLARABEL", "ECOS"),
) -> np.ndarray:
    """Solve a long-only single-period Markowitz portfolio problem.

    The optimization problem is:

        maximize mu^T w - 0.5 * gamma * w^T Sigma w

    subject to:

        sum(w) = 1
        0 <= w <= weight_cap

    If all solvers fail, the function returns equal weights.
    """
    if train_returns.empty:
        raise ValueError("train_returns must not be empty.")

    mu = train_returns.mean().values
    cov = LedoitWolf().fit(train_returns.values).covariance_
    cov = nearest_psd(cov + cov_ridge * np.eye(len(mu)))

    n_assets = len(mu)

    if weight_cap * n_assets < 1.0:
        raise ValueError("weight_cap is too small to allow weights to sum to one.")

    w = cp.Variable(n_assets)

    objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= weight_cap,
    ]

    problem = cp.Problem(objective, constraints)
    solved = _solve_problem_with_fallbacks(problem, solvers)

    if solved and w.value is not None:
        weights = _normalize_weights(w.value)
        if weights.max() <= weight_cap + 1e-6:
            return weights

    return np.ones(n_assets) / n_assets


def weighted_mean_and_cov(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute weighted mean and covariance.

    Parameters
    ----------
    values:
        Array with shape ``(n_observations, n_assets)``.
    weights:
        Nonnegative observation weights.

    Returns
    -------
    mean:
        Weighted mean vector.
    cov:
        Weighted covariance matrix.
    effective_weight:
        Sum of the input weights. This is useful for regime models where
        the total probability mass measures the effective sample size.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")

    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")

    if len(weights) != len(values):
        raise ValueError("weights and values must have the same number of rows.")

    weight_sum = float(weights.sum())

    if weight_sum <= 1e-12:
        mean = np.mean(values, axis=0)
        cov = np.cov(values, rowvar=False)
        return mean, np.atleast_2d(cov), 0.0

    norm_weights = weights / weight_sum
    mean = norm_weights @ values
    centered = values - mean
    cov = (centered.T * norm_weights) @ centered

    return mean, np.atleast_2d(cov), weight_sum


def solve_mpc(
    current_weights: np.ndarray,
    mu_path: np.ndarray,
    cov_path: np.ndarray,
    cost_path: np.ndarray,
    gamma: float = DEFAULT_GAMMA,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
    turnover_cap: float = DEFAULT_MPC_TURNOVER_CAP,
    quadratic_trade_penalty: float = DEFAULT_MPC_QUADRATIC_TRADE_PENALTY,
    solvers: tuple[str, ...] = ("OSQP", "CLARABEL", "ECOS"),
) -> np.ndarray:
    """Solve a long-only multi-period portfolio optimization problem.

    The function returns only the first-period target weights, following
    the receding-horizon MPC convention.
    """
    current_weights = np.asarray(current_weights, dtype=float).reshape(-1)
    mu_path = np.asarray(mu_path, dtype=float)
    cov_path = np.asarray(cov_path, dtype=float)
    cost_path = np.asarray(cost_path, dtype=float)

    if mu_path.ndim != 2:
        raise ValueError("mu_path must have shape (horizon, n_assets).")

    horizon, n_assets = mu_path.shape

    if current_weights.shape != (n_assets,):
        raise ValueError("current_weights must have shape (n_assets,).")

    if cov_path.shape != (horizon, n_assets, n_assets):
        raise ValueError("cov_path must have shape (horizon, n_assets, n_assets).")

    if cost_path.shape != (horizon, n_assets):
        raise ValueError("cost_path must have shape (horizon, n_assets).")

    if weight_cap * n_assets < 1.0:
        raise ValueError("weight_cap is too small to allow weights to sum to one.")

    current_weights = _normalize_weights(current_weights)

    x = cp.Variable((horizon, n_assets))
    buy = cp.Variable((horizon, n_assets), nonneg=True)
    sell = cp.Variable((horizon, n_assets), nonneg=True)

    objective_terms = []
    constraints = []

    previous = current_weights

    for h in range(horizon):
        trade = buy[h] - sell[h]
        turnover = 0.5 * cp.sum(buy[h] + sell[h])

        constraints += [
            x[h] - previous == trade,
            cp.sum(x[h]) == 1.0,
            x[h] >= 0.0,
            x[h] <= weight_cap,
            turnover <= turnover_cap,
        ]

        objective_terms.append(
            mu_path[h] @ x[h]
            - 0.5 * gamma * cp.quad_form(x[h], cp.psd_wrap(nearest_psd(cov_path[h])))
            - cost_path[h] @ (buy[h] + sell[h])
            - 0.5 * quadratic_trade_penalty * cp.sum_squares(trade)
        )

        previous = x[h]

    problem = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints)
    solved = _solve_problem_with_fallbacks(problem, solvers)

    if solved and x.value is not None:
        target = _normalize_weights(x.value[0])
        if target.max() <= weight_cap + 1e-6:
            return target

    raise RuntimeError("MPC solver failed.")