import numpy as np
import pandas as pd
import pytest
import regime_mpc.regimes as regimes_module
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

def make_success_regime_test_data(n_days: int = 120):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(123)

    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n_days, 4)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )

    # Construct three visually different regimes so GaussianHMM has enough structure.
    block = n_days // 3
    mom_63 = np.concatenate(
        [
            np.full(block, 1.0),
            np.full(block, 0.0),
            np.full(n_days - 2 * block, -1.0),
        ]
    )
    rv_63 = np.concatenate(
        [
            np.full(block, 0.10),
            np.full(block, 0.20),
            np.full(n_days - 2 * block, 0.35),
        ]
    )
    amihud_log = np.concatenate(
        [
            np.full(block, 0.001),
            np.full(block, 0.005),
            np.full(n_days - 2 * block, 0.010),
        ]
    )
    vix_log = np.concatenate(
        [
            np.full(block, 2.8),
            np.full(block, 3.1),
            np.full(n_days - 2 * block, 3.5),
        ]
    )
    hy_spread = np.concatenate(
        [
            np.full(block, 3.0),
            np.full(block, 4.5),
            np.full(n_days - 2 * block, 6.0),
        ]
    )

    features = pd.DataFrame(
        {
            "mom_63": mom_63 + rng.normal(0, 0.01, n_days),
            "rv_63": rv_63 + rng.normal(0, 0.005, n_days),
            "amihud_log": amihud_log + rng.normal(0, 0.0001, n_days),
            "vix_log": vix_log + rng.normal(0, 0.01, n_days),
            "hy_spread": hy_spread + rng.normal(0, 0.02, n_days),
        },
        index=dates,
    )

    return returns, features


def test_fit_hmm_regime_inputs_returns_expected_shapes():
    returns, features = make_success_regime_test_data(n_days=120)

    mu_path, cov_path, cost_path, q_path, meta = fit_hmm_regime_inputs(
        date=returns.index[-1],
        returns=returns,
        features=features,
        initial_train_days=100,
        hmm_min_feature_days=40,
        hmm_max_iter=30,
        mpc_horizon=2,
    )

    assert mu_path.shape == (2, 4)
    assert cov_path.shape == (2, 4, 4)
    assert cost_path.shape == (2, 4)
    assert q_path.shape == (2, 3)
    assert np.allclose(q_path.sum(axis=1), 1.0)
    assert meta["dominant_regime"] in {"calm", "transition", "stress"}
    assert "hmm_log_likelihood" in meta


def test_fit_hmm_regime_inputs_rejects_missing_required_features():
    returns, features = make_success_regime_test_data(n_days=80)
    features = features.drop(columns=["rv_63"])

    with pytest.raises(ValueError):
        fit_hmm_regime_inputs(
            date=returns.index[-1],
            returns=returns,
            features=features,
            initial_train_days=60,
            hmm_min_feature_days=30,
        )


def test_fit_hmm_regime_inputs_rejects_non_three_state_model():
    returns, features = make_success_regime_test_data(n_days=80)

    with pytest.raises(ValueError):
        fit_hmm_regime_inputs(
            date=returns.index[-1],
            returns=returns,
            features=features,
            initial_train_days=60,
            hmm_min_feature_days=30,
            hmm_states=2,
        )


def test_fit_hmm_regime_inputs_rejects_bad_cost_multiplier_length():
    returns, features = make_success_regime_test_data(n_days=80)

    with pytest.raises(ValueError):
        fit_hmm_regime_inputs(
            date=returns.index[-1],
            returns=returns,
            features=features,
            initial_train_days=60,
            hmm_min_feature_days=30,
            regime_cost_multipliers=np.array([1.0, 2.0]),
        )


def test_fit_hmm_regime_inputs_rejects_bad_regime_label_length():
    returns, features = make_success_regime_test_data(n_days=80)

    with pytest.raises(ValueError):
        fit_hmm_regime_inputs(
            date=returns.index[-1],
            returns=returns,
            features=features,
            initial_train_days=60,
            hmm_min_feature_days=30,
            regime_labels=("calm", "stress"),
        )


def test_make_hmm_mpc_target_success_branch_with_monkeypatch(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    returns = pd.DataFrame(
        np.random.default_rng(0).normal(0.0, 0.01, size=(20, 4)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    features = pd.DataFrame(
        {
            "mom_63": np.ones(20),
            "rv_63": np.ones(20),
            "amihud_log": np.ones(20),
            "vix_log": np.ones(20),
            "hy_spread": np.ones(20),
        },
        index=dates,
    )

    def fake_fit_hmm_regime_inputs(*args, **kwargs):
        mu_path = np.ones((2, 4)) * 0.001
        cov_path = np.stack([np.eye(4) * 0.01, np.eye(4) * 0.01])
        cost_path = np.ones((2, 4)) * 0.00035
        q_path = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
        meta = {
            "date": dates[-1],
            "prob_calm": 0.7,
            "prob_transition": 0.2,
            "prob_stress": 0.1,
            "next_prob_calm": 0.6,
            "next_prob_transition": 0.3,
            "next_prob_stress": 0.1,
            "dominant_regime": "calm",
            "next_cost_multiplier": 1.0,
            "hmm_log_likelihood": -10.0,
            "effective_sample_calm": 50.0,
            "effective_sample_transition": 30.0,
            "effective_sample_stress": 20.0,
        }
        return mu_path, cov_path, cost_path, q_path, meta

    def fake_solve_mpc(current_weights, mu_path, cov_path, cost_path):
        return np.ones(4) / 4

    monkeypatch.setattr(regimes_module, "fit_hmm_regime_inputs", fake_fit_hmm_regime_inputs)
    monkeypatch.setattr(regimes_module, "solve_mpc", fake_solve_mpc)

    target_fn, records = make_hmm_mpc_target(
        returns=returns,
        features=features,
        initial_train_days=10,
    )

    target, cost_vector = target_fn(dates[-1], np.ones(4) / 4)

    assert np.allclose(target, np.ones(4) / 4)
    assert cost_vector.shape == (4,)
    assert len(records) == 1
    assert records[0]["solver_status"] == "hmm_mpc"