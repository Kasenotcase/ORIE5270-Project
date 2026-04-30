import numpy as np
import pandas as pd
import pytest

from regime_mpc.regimes import fit_hmm_regime_inputs, make_hmm_mpc_target


def make_regime_test_data(n_days: int = 80):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n_days, 4)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )

    features = pd.DataFrame(
        {
            "mom_63": rng.normal(size=n_days),
            "rv_63": rng.uniform(0.1, 0.3, size=n_days),
            "amihud_log": rng.uniform(0.0, 0.01, size=n_days),
            "vix_log": rng.uniform(2.5, 3.5, size=n_days),
            "hy_spread": rng.uniform(3.0, 6.0, size=n_days),
        },
        index=dates,
    )

    return returns, features


def test_fit_hmm_regime_inputs_rejects_insufficient_data():
    returns, features = make_regime_test_data(n_days=30)

    with pytest.raises(ValueError):
        fit_hmm_regime_inputs(
            date=returns.index[-1],
            returns=returns,
            features=features,
            initial_train_days=20,
            hmm_min_feature_days=40,
        )


def test_make_hmm_mpc_target_falls_back_to_markowitz_when_hmm_fails():
    returns, features = make_regime_test_data(n_days=80)

    target_fn, records = make_hmm_mpc_target(
        returns=returns,
        features=features,
        initial_train_days=40,
    )

    current_weights = np.ones(4) / 4
    target, cost_vector = target_fn(returns.index[-1], current_weights)

    assert target.shape == (4,)
    assert cost_vector.shape == (4,)
    assert np.all(target >= -1e-8)
    assert np.isclose(target.sum(), 1.0)
    assert len(records) == 1
    assert records[0]["dominant_regime"] == "fallback"