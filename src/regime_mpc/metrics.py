"""Performance metrics for portfolio backtests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown of a return series.

    Parameters
    ----------
    returns:
        Periodic simple returns, such as daily portfolio returns.

    Returns
    -------
    float
        The most negative drawdown. For example, -0.2 means a 20% drawdown.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan

    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized compound return from periodic returns."""
    r = returns.dropna()
    if r.empty:
        return np.nan

    total_growth = float((1.0 + r).prod())
    return total_growth ** (periods_per_year / len(r)) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized volatility from periodic returns."""
    r = returns.dropna()
    if len(r) <= 1:
        return 0.0

    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute the annualized Sharpe ratio with zero risk-free rate."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)

    if ann_vol <= 0 or np.isnan(ann_vol):
        return np.nan

    return float(ann_ret / ann_vol)


def performance_table(
    strategy_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create a performance summary table for multiple strategies.

    Parameters
    ----------
    strategy_returns:
        DataFrame whose columns are strategy names and rows are dated returns.
    turnover:
        DataFrame of turnover values with matching strategy columns.
    costs:
        DataFrame of transaction costs with matching strategy columns.
    output_path:
        Optional path. If provided, the table is also written as a CSV file.

    Returns
    -------
    pd.DataFrame
        One row per strategy with return, volatility, Sharpe, drawdown,
        wealth, turnover, and transaction cost statistics.
    """
    rows = []

    for name in strategy_returns.columns:
        r = strategy_returns[name].dropna()

        if r.empty:
            rows.append(
                {
                    "strategy": name,
                    "start": None,
                    "end": None,
                    "observations": 0,
                    "annualized_return": np.nan,
                    "annualized_volatility": np.nan,
                    "sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "final_wealth": np.nan,
                    "avg_daily_turnover": np.nan,
                    "avg_rebalance_turnover": np.nan,
                    "total_cost": np.nan,
                }
            )
            continue

        ann_ret = annualized_return(r)
        ann_vol = annualized_volatility(r)

        strategy_turnover = turnover[name] if name in turnover.columns else pd.Series(dtype=float)
        strategy_costs = costs[name] if name in costs.columns else pd.Series(dtype=float)

        positive_turnover = strategy_turnover.loc[strategy_turnover > 0]

        rows.append(
            {
                "strategy": name,
                "start": r.index.min().date().isoformat(),
                "end": r.index.max().date().isoformat(),
                "observations": len(r),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
                "max_drawdown": max_drawdown(r),
                "final_wealth": float((1.0 + r).prod()),
                "avg_daily_turnover": float(strategy_turnover.mean())
                if not strategy_turnover.empty
                else np.nan,
                "avg_rebalance_turnover": float(positive_turnover.mean())
                if not positive_turnover.empty
                else np.nan,
                "total_cost": float(strategy_costs.sum()) if not strategy_costs.empty else np.nan,
            }
        )

    table = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)

    return table