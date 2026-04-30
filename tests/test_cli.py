import pandas as pd

from regime_mpc.cli import make_synthetic_returns, run_quick_demo


def test_make_synthetic_returns_is_reproducible():
    returns_a = make_synthetic_returns(n_days=10, n_assets=3, seed=123)
    returns_b = make_synthetic_returns(n_days=10, n_assets=3, seed=123)

    pd.testing.assert_frame_equal(returns_a, returns_b)
    assert returns_a.shape == (10, 3)


def test_run_quick_demo_writes_outputs(tmp_path):
    table = run_quick_demo(output_dir=tmp_path)

    assert len(table) == 2
    assert (tmp_path / "performance.csv").exists()
    assert (tmp_path / "strategy_returns.csv").exists()
    assert (tmp_path / "turnover.csv").exists()
    assert (tmp_path / "costs.csv").exists()
    assert (tmp_path / "equal_weight_weights.csv").exists()
    assert (tmp_path / "markowitz_weights.csv").exists()