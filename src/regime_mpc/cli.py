"""Command-line interface for the regime MPC portfolio project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regime_mpc.backtest import equal_weight_target, run_backtest
from regime_mpc.metrics import performance_table
from regime_mpc.optimization import solve_markowitz


def make_synthetic_returns(
    n_days: int = 300,
    n_assets: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Create synthetic asset returns for a reproducible quick demo."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    market = rng.normal(0.0004, 0.008, size=(n_days, 1))
    idiosyncratic = rng.normal(0.0, 0.006, size=(n_days, n_assets))
    returns = market + idiosyncratic

    columns = [f"Asset_{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(returns, index=dates, columns=columns)


def make_markowitz_target(
    returns: pd.DataFrame,
    initial_train_days: int,
):
    """Create a simple rolling Markowitz target function."""

    def target(date: pd.Timestamp, _current_weights: np.ndarray) -> np.ndarray:
        train = returns.loc[:date].tail(initial_train_days)
        return solve_markowitz(train, weight_cap=0.7)

    return target


def run_quick_demo(output_dir: str | Path = "outputs/quick_demo") -> pd.DataFrame:
    """Run a small reproducible backtest on synthetic data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = make_synthetic_returns()
    initial_train_days = 60

    equal_returns, equal_turnover, equal_costs, equal_weights = run_backtest(
        returns=returns,
        strategy_name="EqualWeight",
        target_weight_fn=equal_weight_target,
        initial_train_days=initial_train_days,
        rebalance_freq="W-FRI",
        show_progress=False,
    )

    markowitz_returns, markowitz_turnover, markowitz_costs, markowitz_weights = run_backtest(
        returns=returns,
        strategy_name="Markowitz",
        target_weight_fn=make_markowitz_target(returns, initial_train_days),
        initial_train_days=initial_train_days,
        rebalance_freq="W-FRI",
        show_progress=False,
    )

    strategy_returns = pd.concat([equal_returns, markowitz_returns], axis=1)
    turnover = pd.concat([equal_turnover, markowitz_turnover], axis=1)
    costs = pd.concat([equal_costs, markowitz_costs], axis=1)

    table = performance_table(
        strategy_returns=strategy_returns,
        turnover=turnover,
        costs=costs,
        output_path=output_dir / "performance.csv",
    )

    strategy_returns.to_csv(output_dir / "strategy_returns.csv")
    turnover.to_csv(output_dir / "turnover.csv")
    costs.to_csv(output_dir / "costs.csv")
    equal_weights.to_csv(output_dir / "equal_weight_weights.csv")
    markowitz_weights.to_csv(output_dir / "markowitz_weights.csv")

    return table


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Regime-aware MPC portfolio optimization project CLI."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    quick_demo = subparsers.add_parser(
        "quick-demo",
        help="Run a small reproducible demo using synthetic data.",
    )
    quick_demo.add_argument(
        "--output-dir",
        default="outputs/quick_demo",
        help="Directory where demo outputs are written.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "quick-demo":
        table = run_quick_demo(output_dir=args.output_dir)
        print("Quick demo completed.")
        print(table.to_string(index=False))
        print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()