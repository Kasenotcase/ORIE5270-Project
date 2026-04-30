from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_mpc.data import clean_data, ensure_dirs, field_frame


def test_ensure_dirs_creates_directories(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b" / "nested"

    ensure_dirs(path_a, path_b)

    assert path_a.exists()
    assert path_b.exists()


def test_field_frame_extracts_field_from_yfinance_multiindex():
    dates = pd.date_range("2020-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["Adj Close", "Close"]])
    data = pd.DataFrame(
        [
            [10.0, 11.0, 20.0, 21.0],
            [12.0, 13.0, 22.0, 23.0],
        ],
        index=dates,
        columns=columns,
    )

    result = field_frame(data, "Adj Close", assets=("AAA", "BBB"))

    assert list(result.columns) == ["AAA", "BBB"]
    assert result.loc[dates[0], "AAA"] == 10.0
    assert result.loc[dates[0], "BBB"] == 20.0


def test_field_frame_rejects_non_multiindex():
    data = pd.DataFrame({"Adj Close": [1.0, 2.0]})

    with pytest.raises(ValueError):
        field_frame(data, "Adj Close")


def test_clean_data_removes_invalid_rows_and_aligns_outputs():
    dates = pd.date_range("2020-01-01", periods=5)

    adj_close = pd.DataFrame(
        {
            "A": [100.0, 101.0, np.inf, 103.0, 104.0],
            "B": [200.0, 202.0, 204.0, 206.0, 208.0],
        },
        index=dates,
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [200.0, 202.0, 204.0, 206.0, 208.0],
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "A": [1000, 1100, 1200, 1300, 1400],
            "B": [2000, 2100, 2200, 2300, 2400],
        },
        index=dates,
    )
    macro = pd.DataFrame(
        {
            "VIXCLS": [20.0, np.nan, 22.0, 23.0, 24.0],
            "BAMLH0A0HYM2": [4.0, 4.1, 4.2, 4.3, 4.4],
        },
        index=dates,
    )

    clean_adj, clean_close, clean_volume, returns, clean_macro = clean_data(
        adj_close,
        close,
        volume,
        macro,
    )

    assert dates[2] not in clean_adj.index
    assert clean_adj.index.equals(returns.index)
    assert clean_close.index.equals(returns.index)
    assert clean_volume.index.equals(returns.index)
    assert clean_macro.index.equals(returns.index)
    assert not returns.isna().any().any()


def test_clean_data_can_write_outputs(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3)
    adj_close = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dates)
    close = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dates)
    volume = pd.DataFrame({"A": [1000, 1100, 1200]}, index=dates)
    macro = pd.DataFrame(
        {"VIXCLS": [20.0, 21.0, 22.0], "BAMLH0A0HYM2": [4.0, 4.1, 4.2]},
        index=dates,
    )

    clean_data(adj_close, close, volume, macro, processed_dir=tmp_path)

    expected_files = [
        "adj_close_clean.csv",
        "close_clean.csv",
        "volume_clean.csv",
        "returns.csv",
        "macro_aligned.csv",
    ]

    for filename in expected_files:
        assert Path(tmp_path / filename).exists()