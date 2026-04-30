"""Feature engineering utilities for market regime modeling."""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def build_regime_features(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build market-regime features from price, volume, return, and macro data.

    The features follow the original project logic:

    - 21-day and 63-day market momentum
    - 63-day market drawdown
    - 21-day and 63-day realized volatility
    - Amihud illiquidity
    - VIX level and 21-day VIX change
    - high-yield spread level and 21-day spread change

    Parameters
    ----------
    adj_close:
        Adjusted close prices, indexed by date.
    close:
        Close prices, indexed by date.
    volume:
        Trading volume, indexed by date.
    returns:
        Asset returns, indexed by date.
    macro:
        Macro-financial variables. Must include ``VIXCLS`` and ``BAMLH0A0HYM2``.
    output_path:
        Optional CSV path for saving the generated features.

    Returns
    -------
    pd.DataFrame
        Clean feature matrix indexed by date.
    """
    required_macro_cols = {"VIXCLS", "BAMLH0A0HYM2"}
    missing = required_macro_cols.difference(macro.columns)
    if missing:
        raise ValueError(f"macro is missing required columns: {sorted(missing)}")

    common_index = adj_close.index.intersection(close.index).intersection(volume.index)
    common_index = common_index.intersection(returns.index).intersection(macro.index)

    adj_close = adj_close.loc[common_index]
    close = close.loc[common_index]
    volume = volume.loc[common_index]
    returns = returns.loc[common_index]
    macro = macro.loc[common_index]

    market_price = adj_close.mean(axis=1)
    market_return = market_price.pct_change()
    log_market = np.log(market_price)

    features = pd.DataFrame(index=returns.index)

    features["mom_21"] = log_market.diff(21)
    features["mom_63"] = log_market.diff(63)
    features["drawdown_63"] = market_price / market_price.rolling(63).max() - 1.0
    features["rv_21"] = market_return.rolling(21).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    features["rv_63"] = market_return.rolling(63).std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    dollar_volume = close * volume
    illiq = returns.abs() / dollar_volume.replace(0, np.nan)
    features["amihud_log"] = np.log1p(illiq.mean(axis=1))

    features["vix_log"] = np.log(macro["VIXCLS"])
    features["vix_change_21"] = np.log(macro["VIXCLS"]).diff(21)
    features["hy_spread"] = macro["BAMLH0A0HYM2"]
    features["hy_spread_change_21"] = macro["BAMLH0A0HYM2"].diff(21)

    features = features.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path)

    return features