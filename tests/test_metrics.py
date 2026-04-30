import math

import numpy as np
import pandas as pd

from regime_mpc.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    performance_table,
    sharpe_ratio,
)


def test_max_drawdown_for_simple_path():
    returns = pd.Series([0.10, -0.20, 0.05])

    result = max_drawdown(returns)

    assert math.isclose(result, -0.20)


def test_max_drawdown_empty_returns_is_nan():
    returns = pd.Series(dtype=float)

    result = max_drawdown(returns)

    assert np.isnan(result)


def test_annualized_return_constant_daily_return():
    returns = pd.Series([0.01, 0.01])

    result = annualized_return(returns, periods_per_year=2)

    assert math.isclose(result, 0.0201)


def test_annualized_return_empty_returns_is_nan():
    returns = pd.Series(dtype=float)

    result = annualized_return(returns)

    assert np.isnan(result)


def test_annualized_volatility_constant_returns_is_zero():
    returns = pd.Series([0.01, 0.01, 0.01])

    result = annualized_volatility(returns)

    assert math.isclose(result, 0.0, abs_tol=1e-12)


def test_annualized_volatility_single_observation_is_zero():
    returns = pd.Series([0.01])

    result = annualized_volatility(returns)

    assert result == 0.0


def test_sharpe_ratio_zero_volatility_is_nan():
    returns = pd.Series([0.01, 0.01, 0.01])

    result = sharpe_ratio(returns)

    assert np.isnan(result)


def test_performance_table_contains_expected_columns():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    strategy_returns = pd.DataFrame(
        {
            "strategy_a": [0.01, -0.02, 0.03],
        },
        index=dates,
    )
    turnover = pd.DataFrame(
        {
            "strategy_a": [0.0, 0.5, 0.2],
        },
        index=dates,
    )
    costs = pd.DataFrame(
        {
            "strategy_a": [0.0, 0.001, 0.002],
        },
        index=dates,
    )

    table = performance_table(strategy_returns, turnover, costs)

    assert len(table) == 1
    assert table.loc[0, "strategy"] == "strategy_a"
    assert table.loc[0, "observations"] == 3
    assert math.isclose(table.loc[0, "total_cost"], 0.003)
    assert math.isclose(table.loc[0, "avg_rebalance_turnover"], 0.35)


def test_performance_table_can_write_csv(tmp_path):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    strategy_returns = pd.DataFrame({"strategy_a": [0.01, 0.02]}, index=dates)
    turnover = pd.DataFrame({"strategy_a": [0.0, 0.1]}, index=dates)
    costs = pd.DataFrame({"strategy_a": [0.0, 0.001]}, index=dates)

    output_path = tmp_path / "performance.csv"
    table = performance_table(strategy_returns, turnover, costs, output_path=output_path)

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert saved.shape == table.shape