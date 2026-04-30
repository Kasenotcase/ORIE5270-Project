import math

import numpy as np
import pandas as pd
import pytest

from regime_mpc.backtest import equal_weight_target, normalize_weights, rebalance_dates, run_backtest


def test_normalize_weights_clips_and_normalizes():
    weights = np.array([0.5, -0.2, 0.7])

    result = normalize_weights(weights)

    assert np.all(result >= 0)
    assert math.isclose(result.sum(), 1.0)


def test_equal_weight_target_matches_dimension():
    current_weights = np.array([0.2, 0.3, 0.5])

    result = equal_weight_target(pd.Timestamp("2020-01-01"), current_weights)

    assert np.allclose(result, np.ones(3) / 3)


def test_rebalance_dates_returns_empty_if_not_enough_dates():
    dates = pd.date_range("2020-01-01", periods=5)

    result = rebalance_dates(dates, initial_train_days=10)

    assert result == set()


def test_run_backtest_outputs_aligned_series():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.full(20, 0.001),
            "B": np.full(20, 0.002),
        },
        index=dates,
    )

    def target_fn(_date, _weights):
        return np.array([0.6, 0.4])

    strategy_returns, turnover, costs, weights = run_backtest(
        returns=returns,
        strategy_name="test_strategy",
        target_weight_fn=target_fn,
        initial_train_days=5,
        rebalance_freq="7D",
        show_progress=False,
    )

    assert len(strategy_returns) == 14
    assert strategy_returns.index.equals(turnover.index)
    assert strategy_returns.index.equals(costs.index)
    assert weights.shape == (14, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_run_backtest_accepts_target_with_cost_vector():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.full(20, 0.001),
            "B": np.full(20, 0.002),
        },
        index=dates,
    )

    def target_fn(_date, _weights):
        return np.array([0.8, 0.2]), np.array([0.001, 0.002])

    strategy_returns, turnover, costs, weights = run_backtest(
        returns=returns,
        strategy_name="cost_strategy",
        target_weight_fn=target_fn,
        initial_train_days=5,
        rebalance_freq="7D",
        show_progress=False,
    )

    assert len(strategy_returns) == 14
    assert costs.sum() >= 0
    assert turnover.sum() >= 0
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_run_backtest_rejects_short_data():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.DataFrame({"A": np.zeros(5), "B": np.zeros(5)}, index=dates)

    with pytest.raises(ValueError):
        run_backtest(
            returns=returns,
            strategy_name="too_short",
            target_weight_fn=equal_weight_target,
            initial_train_days=5,
            show_progress=False,
        )