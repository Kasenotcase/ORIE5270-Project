"""Backtesting utilities for portfolio strategies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_INITIAL_TRAIN_DAYS = 756
DEFAULT_REBALANCE_FREQ = "W-FRI"
DEFAULT_TRANSACTION_COST_BPS = 3.5


def rebalance_dates(
    index: pd.DatetimeIndex,
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
) -> set[pd.Timestamp]:
    """Return rebalance dates after the initial training period."""
    eligible = index[initial_train_days:-1]

    if len(eligible) == 0:
        return set()

    dates = pd.Series(eligible, index=eligible).resample(rebalance_freq).last().dropna()
    return set(pd.to_datetime(dates.values))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Clip numerical negatives and normalize weights to sum to one."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 0.0)

    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights) / len(weights)

    return weights / total


def equal_weight_target(_date: pd.Timestamp, current_weights: np.ndarray) -> np.ndarray:
    """Return equal weights with the same dimension as current_weights."""
    n_assets = len(current_weights)
    return np.ones(n_assets) / n_assets


def run_backtest(
    returns: pd.DataFrame,
    strategy_name: str,
    target_weight_fn: Callable[[pd.Timestamp, np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]],
    initial_train_days: int = DEFAULT_INITIAL_TRAIN_DAYS,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    show_progress: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Run a daily portfolio backtest with periodic rebalancing.

    Parameters
    ----------
    returns:
        Asset return DataFrame indexed by date.
    strategy_name:
        Name assigned to the output return, turnover, and cost series.
    target_weight_fn:
        Function called at rebalance dates. It should return either a target
        weight vector or ``(target_weights, cost_vector)``.
    initial_train_days:
        Number of initial observations skipped before starting the backtest.
    rebalance_freq:
        Pandas resampling frequency, such as ``"W-FRI"`` or ``"M"``.
    transaction_cost_bps:
        Default transaction cost in basis points.
    show_progress:
        Whether to show a tqdm progress bar.

    Returns
    -------
    return_series:
        Daily net strategy returns.
    turnover_series:
        Daily turnover.
    cost_series:
        Daily transaction costs.
    weights_df:
        Daily post-return portfolio weights.
    """
    if returns.empty:
        raise ValueError("returns must not be empty.")

    if initial_train_days + 1 >= len(returns):
        raise ValueError("Not enough observations after initial_train_days.")

    dates = pd.DatetimeIndex(returns.index)
    n_assets = returns.shape[1]

    rebalance_set = rebalance_dates(
        dates,
        initial_train_days=initial_train_days,
        rebalance_freq=rebalance_freq,
    )

    weights = np.ones(n_assets) / n_assets

    daily_returns = []
    daily_turnover = []
    daily_costs = []
    weight_records = []

    cost_per_dollar = transaction_cost_bps / 10000.0
    iterator = range(initial_train_days + 1, len(dates))

    if show_progress:
        iterator = tqdm(iterator, desc=strategy_name, leave=False)

    for i in iterator:
        prev_date = dates[i - 1]
        date = dates[i]

        cost = 0.0
        turnover = 0.0

        if prev_date in rebalance_set:
            target_result = target_weight_fn(prev_date, weights)

            if isinstance(target_result, tuple):
                target, cost_vector = target_result
            else:
                target = target_result
                cost_vector = np.ones(n_assets) * cost_per_dollar

            target = normalize_weights(target)
            cost_vector = np.asarray(cost_vector, dtype=float).reshape(-1)

            if cost_vector.shape != (n_assets,):
                raise ValueError("cost_vector must have shape (n_assets,).")

            traded = np.abs(target - weights)
            turnover = 0.5 * float(traded.sum())
            cost = float(cost_vector @ traded)

            weights = target

        asset_return = returns.loc[date].values
        gross_return = float(weights @ asset_return)
        net_return = (1.0 - cost) * (1.0 + gross_return) - 1.0

        denom = 1.0 + gross_return
        if denom > 0:
            weights = weights * (1.0 + asset_return) / denom
            weights = normalize_weights(weights)

        daily_returns.append((date, net_return))
        daily_turnover.append((date, turnover))
        daily_costs.append((date, cost))
        weight_records.append(pd.Series(weights, index=returns.columns, name=date))

    return_series = pd.Series(dict(daily_returns), name=strategy_name).sort_index()
    turnover_series = pd.Series(dict(daily_turnover), name=strategy_name).sort_index()
    cost_series = pd.Series(dict(daily_costs), name=strategy_name).sort_index()

    weights_df = pd.DataFrame(weight_records)
    weights_df.index.name = "date"

    return return_series, turnover_series, cost_series, weights_df