import numpy as np
import pandas as pd
import pytest

from regime_mpc.features import build_regime_features


def make_feature_inputs(n_days: int = 90):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    base = np.linspace(100.0, 120.0, n_days)

    adj_close = pd.DataFrame(
        {
            "A": base,
            "B": base * 1.1,
        },
        index=dates,
    )
    close = pd.DataFrame(
        {
            "A": base,
            "B": base * 1.1,
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "A": np.full(n_days, 1_000_000.0),
            "B": np.full(n_days, 2_000_000.0),
        },
        index=dates,
    )
    returns = adj_close.pct_change().dropna()

    macro = pd.DataFrame(
        {
            "VIXCLS": np.linspace(20.0, 30.0, n_days),
            "BAMLH0A0HYM2": np.linspace(4.0, 5.0, n_days),
        },
        index=dates,
    )

    common_index = returns.index
    return (
        adj_close.loc[common_index],
        close.loc[common_index],
        volume.loc[common_index],
        returns,
        macro.loc[common_index],
    )


def test_build_regime_features_has_expected_columns():
    adj_close, close, volume, returns, macro = make_feature_inputs()

    features = build_regime_features(adj_close, close, volume, returns, macro)

    expected_columns = [
        "mom_21",
        "mom_63",
        "drawdown_63",
        "rv_21",
        "rv_63",
        "amihud_log",
        "vix_log",
        "vix_change_21",
        "hy_spread",
        "hy_spread_change_21",
    ]

    assert list(features.columns) == expected_columns
    assert len(features) > 0
    assert not features.isna().any().any()


def test_build_regime_features_handles_zero_volume_without_inf():
    adj_close, close, volume, returns, macro = make_feature_inputs()
    volume.iloc[10:15, 0] = 0.0

    features = build_regime_features(adj_close, close, volume, returns, macro)

    assert np.isfinite(features.to_numpy()).all()


def test_build_regime_features_rejects_missing_macro_columns():
    adj_close, close, volume, returns, macro = make_feature_inputs()
    macro = macro.drop(columns=["VIXCLS"])

    with pytest.raises(ValueError):
        build_regime_features(adj_close, close, volume, returns, macro)


def test_build_regime_features_can_write_csv(tmp_path):
    adj_close, close, volume, returns, macro = make_feature_inputs()

    output_path = tmp_path / "regime_features.csv"
    features = build_regime_features(adj_close, close, volume, returns, macro, output_path=output_path)

    assert output_path.exists()
    saved = pd.read_csv(output_path, index_col=0)
    assert saved.shape == features.shape