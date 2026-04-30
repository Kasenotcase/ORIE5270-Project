"""Data loading, downloading, and cleaning utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_ASSETS = (
    "XLE",
    "XLF",
    "XLK",
    "XLY",
    "XLI",
    "XLV",
    "XLP",
    "XLU",
    "XLB",
)

DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_FRED_SERIES = ("VIXCLS", "BAMLH0A0HYM2")


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def field_frame(data: pd.DataFrame, field: str, assets: tuple[str, ...] = DEFAULT_ASSETS) -> pd.DataFrame:
    """Extract one price field from a yfinance-style multi-index DataFrame.

    Parameters
    ----------
    data:
        DataFrame returned by ``yfinance.download`` with a MultiIndex column.
    field:
        Field name such as ``"Adj Close"``, ``"Close"``, or ``"Volume"``.
    assets:
        Asset tickers to keep and order.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date and with one column per asset.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex DataFrame from yfinance.")

    if field in data.columns.get_level_values(1):
        frame = data.xs(field, axis=1, level=1)
    elif field in data.columns.get_level_values(0):
        frame = data.xs(field, axis=1, level=0)
    else:
        raise KeyError(f"Field {field!r} not found in downloaded data.")

    frame = frame.reindex(columns=list(assets))
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame


def fetch_yahoo_chart_data(
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    raw_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download OHLCV data directly from Yahoo Finance chart endpoint.

    This is a fallback for cases where yfinance fails.
    """
    frames: dict[str, pd.DataFrame] = {}
    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    period2 = int(pd.Timestamp(end_date, tz="UTC").timestamp())
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in assets:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        payload = response.json()
        error = payload.get("chart", {}).get("error")
        if error is not None:
            raise RuntimeError(f"Yahoo chart error for {ticker}: {error}")

        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"]["adjclose"][0]["adjclose"]

        index = (
            pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None)
        )

        frames[ticker] = pd.DataFrame(
            {
                "Open": quote["open"],
                "High": quote["high"],
                "Low": quote["low"],
                "Close": quote["close"],
                "Adj Close": adj,
                "Volume": quote["volume"],
            },
            index=index,
        )

    raw = pd.concat(frames, axis=1).sort_index()

    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_dir / "yahoo_chart_ohlcv_raw.csv")

    adj_close = pd.DataFrame({ticker: frames[ticker]["Adj Close"] for ticker in assets})
    close = pd.DataFrame({ticker: frames[ticker]["Close"] for ticker in assets})
    volume = pd.DataFrame({ticker: frames[ticker]["Volume"] for ticker in assets})

    for frame in (adj_close, close, volume):
        frame.index.name = "Date"

    return adj_close.sort_index(), close.sort_index(), volume.sort_index()


def fetch_price_data(
    refresh: bool = False,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download or load adjusted close, close, and volume data."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    ensure_dirs(raw_dir, processed_dir)

    price_path = processed_dir / "adj_close.csv"
    close_path = processed_dir / "close.csv"
    volume_path = processed_dir / "volume.csv"

    if not refresh and price_path.exists() and close_path.exists() and volume_path.exists():
        adj_close = pd.read_csv(price_path, index_col=0, parse_dates=True)
        close = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        return adj_close, close, volume

    try:
        import yfinance as yf

        data = yf.download(
            list(assets),
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data.empty:
            raise RuntimeError("yfinance returned no ETF data.")

        data.to_csv(raw_dir / "yfinance_ohlcv_raw.csv")
        adj_close = field_frame(data, "Adj Close", assets=assets)
        close = field_frame(data, "Close", assets=assets)
        volume = field_frame(data, "Volume", assets=assets)
        (processed_dir / "price_data_source.txt").write_text("yfinance download\n", encoding="utf-8")

    except Exception:
        adj_close, close, volume = fetch_yahoo_chart_data(
            assets=assets,
            start_date=start_date,
            end_date=end_date,
            raw_dir=raw_dir,
        )
        (processed_dir / "price_data_source.txt").write_text(
            "Yahoo Finance chart endpoint fallback\n",
            encoding="utf-8",
        )

    adj_close.to_csv(price_path)
    close.to_csv(close_path)
    volume.to_csv(volume_path)

    return adj_close, close, volume


def fetch_fred_series(
    series_id: str,
    refresh: bool = False,
    raw_dir: str | Path = "data/raw",
) -> pd.Series:
    """Download or load one FRED time series from the public CSV endpoint."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"fred_{series_id}.csv"

    if not refresh and path.exists():
        raw = pd.read_csv(path)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        raw = pd.read_csv(url)
        raw.to_csv(path, index=False)

    date_col = "DATE" if "DATE" in raw.columns else "observation_date"
    raw[date_col] = pd.to_datetime(raw[date_col])
    values = pd.to_numeric(raw[series_id].replace(".", np.nan), errors="coerce")

    series = pd.Series(values.values, index=raw[date_col], name=series_id).sort_index()
    return series


def fetch_macro_data(
    refresh: bool = False,
    series_ids: tuple[str, ...] = DEFAULT_FRED_SERIES,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Download or load macro-financial data from FRED."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    macro_path = processed_dir / "macro.csv"

    if not refresh and macro_path.exists():
        return pd.read_csv(macro_path, index_col=0, parse_dates=True)

    macro = pd.concat(
        [fetch_fred_series(series_id, refresh=refresh, raw_dir=raw_dir) for series_id in series_ids],
        axis=1,
    )
    macro.to_csv(macro_path)
    return macro


def clean_data(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    macro: pd.DataFrame,
    processed_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean and align price, volume, return, and macro data.

    The function removes invalid prices, invalid volumes, computes daily returns,
    forward-fills macro data to trading dates, and aligns all outputs to the same
    return index.
    """
    adj_close = adj_close.replace([np.inf, -np.inf], np.nan)
    close = close.replace([np.inf, -np.inf], np.nan)
    volume = volume.replace([np.inf, -np.inf], np.nan)

    valid = adj_close.notna().all(axis=1) & close.notna().all(axis=1) & volume.notna().all(axis=1)
    valid &= (adj_close > 0).all(axis=1) & (close > 0).all(axis=1) & (volume >= 0).all(axis=1)

    adj_close = adj_close.loc[valid]
    close = close.loc[valid]
    volume = volume.loc[valid]

    returns = adj_close.pct_change().dropna(how="any")

    macro = macro.reindex(adj_close.index).ffill()
    macro = macro.loc[returns.index]
    close = close.loc[returns.index]
    volume = volume.loc[returns.index]
    adj_close = adj_close.loc[returns.index]

    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        adj_close.to_csv(processed_dir / "adj_close_clean.csv")
        close.to_csv(processed_dir / "close_clean.csv")
        volume.to_csv(processed_dir / "volume_clean.csv")
        returns.to_csv(processed_dir / "returns.csv")
        macro.to_csv(processed_dir / "macro_aligned.csv")

    return adj_close, close, volume, returns, macro