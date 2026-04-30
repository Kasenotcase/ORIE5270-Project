import math

import numpy as np
import pandas as pd
import pytest

from regime_mpc.optimization import (
    nearest_psd,
    solve_markowitz,
    solve_mpc,
    weighted_mean_and_cov,
)


def test_nearest_psd_returns_symmetric_psd_matrix():
    matrix = np.array(
        [
            [1.0, 2.0],
            [2.0, -3.0],
        ]
    )

    result = nearest_psd(matrix)

    assert np.allclose(result, result.T)
    assert np.linalg.eigvalsh(result).min() >= -1e-10


def test_weighted_mean_and_cov_with_positive_weights():
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    weights = np.array([0.2, 0.3, 0.5])

    mean, cov, weight_sum = weighted_mean_and_cov(values, weights)

    expected_mean = weights @ values / weights.sum()

    assert np.allclose(mean, expected_mean)
    assert cov.shape == (2, 2)
    assert math.isclose(weight_sum, 1.0)
    assert np.linalg.eigvalsh(cov).min() >= -1e-10


def test_weighted_mean_and_cov_with_zero_weights_falls_back_to_unweighted():
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )
    weights = np.zeros(3)

    mean, cov, weight_sum = weighted_mean_and_cov(values, weights)

    assert np.allclose(mean, values.mean(axis=0))
    assert cov.shape == (2, 2)
    assert weight_sum == 0.0


def test_weighted_mean_and_cov_rejects_shape_mismatch():
    values = np.ones((3, 2))
    weights = np.ones(2)

    with pytest.raises(ValueError):
        weighted_mean_and_cov(values, weights)


def test_solve_markowitz_returns_valid_long_only_weights():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(loc=0.001, scale=0.01, size=(80, 5)),
        columns=["A", "B", "C", "D", "E"],
    )

    weights = solve_markowitz(returns, weight_cap=0.6)

    assert weights.shape == (5,)
    assert np.all(weights >= -1e-8)
    assert weights.max() <= 0.6 + 1e-6
    assert math.isclose(weights.sum(), 1.0, abs_tol=1e-6)


def test_solve_markowitz_rejects_impossible_weight_cap():
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(size=(20, 4)),
        columns=["A", "B", "C", "D"],
    )

    with pytest.raises(ValueError):
        solve_markowitz(returns, weight_cap=0.2)


def test_solve_mpc_returns_valid_first_period_target():
    current_weights = np.array([0.25, 0.25, 0.25, 0.25])

    mu_path = np.array(
        [
            [0.01, 0.02, 0.015, 0.005],
            [0.01, 0.018, 0.012, 0.004],
        ]
    )

    cov = np.eye(4) * 0.02
    cov_path = np.stack([cov, cov])

    cost_path = np.ones((2, 4)) * 0.001

    target = solve_mpc(
        current_weights=current_weights,
        mu_path=mu_path,
        cov_path=cov_path,
        cost_path=cost_path,
        weight_cap=0.7,
        turnover_cap=0.8,
    )

    assert target.shape == (4,)
    assert np.all(target >= -1e-8)
    assert target.max() <= 0.7 + 1e-6
    assert math.isclose(target.sum(), 1.0, abs_tol=1e-6)


def test_solve_mpc_rejects_bad_shapes():
    current_weights = np.array([0.5, 0.5])
    mu_path = np.ones((2, 2))
    cov_path = np.ones((2, 2, 2))
    cost_path = np.ones((2, 3))

    with pytest.raises(ValueError):
        solve_mpc(current_weights, mu_path, cov_path, cost_path)