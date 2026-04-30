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

from regime_mpc.data import fetch_fred_series, fetch_macro_data, fetch_price_data, fetch_yahoo_chart_data


def test_fetch_fred_series_reads_cached_file(tmp_path):
    raw = pd.DataFrame(
        {
            "DATE": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "TESTSERIES": ["1.0", ".", "3.0"],
        }
    )
    raw.to_csv(tmp_path / "fred_TESTSERIES.csv", index=False)

    series = fetch_fred_series("TESTSERIES", raw_dir=tmp_path)

    assert series.name == "TESTSERIES"
    assert len(series) == 3
    assert series.index[0] == pd.Timestamp("2020-01-01")
    assert series.iloc[0] == 1.0
    assert np.isnan(series.iloc[1])
    assert series.iloc[2] == 3.0


def test_fetch_macro_data_reads_cached_file(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    expected = pd.DataFrame(
        {
            "VIXCLS": [20.0, 21.0],
            "BAMLH0A0HYM2": [4.0, 4.1],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )
    expected.to_csv(processed_dir / "macro.csv")

    result = fetch_macro_data(processed_dir=processed_dir)

    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_fetch_price_data_reads_cached_files(tmp_path):
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    processed_dir.mkdir()
    raw_dir.mkdir()

    dates = pd.date_range("2020-01-01", periods=2)
    adj_close = pd.DataFrame({"A": [100.0, 101.0]}, index=dates)
    close = pd.DataFrame({"A": [99.0, 100.0]}, index=dates)
    volume = pd.DataFrame({"A": [1000.0, 1100.0]}, index=dates)

    adj_close.to_csv(processed_dir / "adj_close.csv")
    close.to_csv(processed_dir / "close.csv")
    volume.to_csv(processed_dir / "volume.csv")

    result_adj, result_close, result_volume = fetch_price_data(
        assets=("A",),
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    pd.testing.assert_frame_equal(result_adj, adj_close, check_freq=False)
    pd.testing.assert_frame_equal(result_close, close, check_freq=False)
    pd.testing.assert_frame_equal(result_volume, volume, check_freq=False)


def test_fetch_yahoo_chart_data_with_mocked_response(monkeypatch, tmp_path):
    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            timestamps = [
                int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()),
                int(pd.Timestamp("2020-01-02", tz="UTC").timestamp()),
            ]
            return {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "timestamp": timestamps,
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [10.0, 11.0],
                                        "high": [11.0, 12.0],
                                        "low": [9.0, 10.0],
                                        "close": [10.5, 11.5],
                                        "volume": [1000, 1100],
                                    }
                                ],
                                "adjclose": [
                                    {
                                        "adjclose": [10.4, 11.4],
                                    }
                                ],
                            },
                        }
                    ],
                }
            }

    def mock_get(*args, **kwargs):
        return MockResponse()

    import regime_mpc.data as data_module

    monkeypatch.setattr(data_module.requests, "get", mock_get)

    adj_close, close, volume = fetch_yahoo_chart_data(
        assets=("AAA",),
        start_date="2020-01-01",
        end_date="2020-01-03",
        raw_dir=tmp_path,
    )

    assert list(adj_close.columns) == ["AAA"]
    assert list(close.columns) == ["AAA"]
    assert list(volume.columns) == ["AAA"]
    assert adj_close.iloc[0, 0] == 10.4
    assert close.iloc[1, 0] == 11.5
    assert volume.iloc[1, 0] == 1100
    assert (tmp_path / "yahoo_chart_ohlcv_raw.csv").exists()


def test_fetch_yahoo_chart_data_raises_on_chart_error(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "chart": {
                    "error": {"code": "Not Found"},
                    "result": None,
                }
            }

    def mock_get(*args, **kwargs):
        return MockResponse()

    import regime_mpc.data as data_module

    monkeypatch.setattr(data_module.requests, "get", mock_get)

    with pytest.raises(RuntimeError):
        fetch_yahoo_chart_data(
            assets=("BAD",),
            start_date="2020-01-01",
            end_date="2020-01-03",
        )