from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
TABLE_DIR = ROOT / "outputs" / "tables"

ASSETS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
START_DATE = pd.Timestamp("2013-01-08")
LOOKBACK_MU = 63
LOOKBACK_COV = 252
BASE_TRADE_COST = 0.0005
GAMMA = 4.0
ALPHA_GRID = [0.0, 0.05, 0.10, 0.25, 0.50]


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) <= 1:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(252.0))


def sharpe_ratio(r: pd.Series) -> float:
    vol = ann_vol(r)
    if not np.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(ann_return(r) / vol)


def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True).sort_index()
    regime_features = pd.read_csv(DATA_DIR / "regime_features.csv", index_col=0, parse_dates=True).sort_index()
    regime_records = pd.read_csv(TABLE_DIR / "hmm_regime_records.csv", parse_dates=["date"]).sort_values("date")
    return returns, regime_features, regime_records


def build_regime_lookup(regime_records: pd.DataFrame) -> pd.DataFrame:
    lookup = regime_records.set_index("date").copy()
    lookup = lookup[["prob_calm", "prob_transition", "prob_stress", "next_cost_multiplier", "dominant_regime"]]
    return lookup


def solve_weights(mu: np.ndarray, sigma: np.ndarray, prev_weights: np.ndarray, trade_cost: float, gamma: float) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    sigma = 0.5 * (sigma + sigma.T)
    sigma = sigma + 1e-8 * np.eye(n)
    objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, sigma) - trade_cost * cp.norm1(w - prev_weights))
    problem = cp.Problem(objective, [cp.sum(w) == 1.0, w >= 0.0])

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        problem.solve(solver=cp.SCS, warm_start=True, verbose=False)

    if w.value is None:
        return prev_weights.copy()

    weights = np.asarray(w.value).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return prev_weights.copy()
    return weights / total


def compute_sample_cov(history: pd.DataFrame, stress_prob: float, transition_prob: float) -> np.ndarray:
    lw = LedoitWolf().fit(history.values)
    sigma = lw.covariance_
    risk_scale = 1.0 + 0.75 * float(stress_prob) + 0.25 * float(transition_prob)
    return sigma * risk_scale


def backtest_alpha_setting(
    returns: pd.DataFrame,
    regime_lookup: pd.DataFrame,
    alpha_shrink: float,
    gamma: float,
    lookback_mu: int,
    lookback_cov: int,
    base_trade_cost: float,
) -> Dict[str, object]:
    dates = returns.index
    weights_rows: List[pd.Series] = []
    daily_returns: List[float] = []
    daily_turnover: List[float] = []
    daily_costs: List[float] = []

    current_weights = np.ones(len(ASSETS), dtype=float) / len(ASSETS)
    rebalance_count = 0
    solved_rebalances = 0

    rebalance_dates = set(regime_lookup.index.intersection(dates))

    for date in dates:
        if date in rebalance_dates:
            rebalance_count += 1
            prev_dates = dates[dates < date]
            if len(prev_dates) >= lookback_mu:
                hist_end = prev_dates[-1]
                hist_start = prev_dates[max(0, len(prev_dates) - lookback_cov)]
                history = returns.loc[hist_start:hist_end, ASSETS].dropna()
                if len(history) >= max(lookback_mu, 30):
                    mu_raw = history.tail(lookback_mu).mean()
                    mu = alpha_shrink * (mu_raw - mu_raw.mean())
                    regime_row = regime_lookup.loc[date]
                    sigma = compute_sample_cov(history, regime_row["prob_stress"], regime_row["prob_transition"])
                    trade_cost = base_trade_cost * float(regime_row["next_cost_multiplier"])
                    target_weights = solve_weights(mu.values, sigma, current_weights, trade_cost, gamma)
                    turnover = 0.5 * float(np.abs(target_weights - current_weights).sum())
                    cost = trade_cost * turnover
                    current_weights = target_weights
                    solved_rebalances += 1
                else:
                    turnover = 0.0
                    cost = 0.0
            else:
                turnover = 0.0
                cost = 0.0
            daily_turnover.append(turnover)
            daily_costs.append(cost)
        else:
            daily_turnover.append(0.0)
            daily_costs.append(0.0)

        daily_returns.append(float(np.dot(current_weights, returns.loc[date, ASSETS].values)))
        weights_rows.append(pd.Series(current_weights, index=ASSETS, name=date))

    weights = pd.DataFrame(weights_rows)
    weights.index.name = "date"
    strategy_returns = pd.Series(daily_returns, index=dates, name="weak_alpha_return")
    turnover_series = pd.Series(daily_turnover, index=dates, name="weak_alpha_turnover")
    cost_series = pd.Series(daily_costs, index=dates, name="weak_alpha_cost")

    performance = {
        "alpha_shrink": alpha_shrink,
        "gamma": gamma,
        "lookback_mu": lookback_mu,
        "lookback_cov": lookback_cov,
        "observations": int(strategy_returns.notna().sum()),
        "rebalance_count": int(rebalance_count),
        "solved_rebalances": int(solved_rebalances),
        "annualized_return": ann_return(strategy_returns),
        "annualized_volatility": ann_vol(strategy_returns),
        "sharpe": sharpe_ratio(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "final_wealth": float((1.0 + strategy_returns.dropna()).prod()),
        "avg_daily_turnover": float(turnover_series.mean()),
        "avg_rebalance_turnover": float(turnover_series[turnover_series > 0].mean()) if (turnover_series > 0).any() else 0.0,
        "total_cost": float(cost_series.sum()),
    }

    return {
        "performance": performance,
        "weights": weights,
        "strategy_returns": strategy_returns,
        "turnover": turnover_series,
        "costs": cost_series,
    }


def load_existing_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy_perf = pd.read_csv(TABLE_DIR / "strategy_performance.csv")
    strategy_returns = pd.read_csv(TABLE_DIR / "strategy_daily_returns.csv", index_col=0, parse_dates=True)
    hmm_regimes = pd.read_csv(TABLE_DIR / "hmm_regime_records.csv", parse_dates=["date"])
    return strategy_perf, strategy_returns, hmm_regimes


def summarize_against_baselines(selected_perf: pd.Series, baseline_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_perf = baseline_perf.set_index("strategy")
    for name in ["EqualWeight_weekly", "Markowitz_CVXPY_weekly", "HMM_MPC_CVXPY_weekly"]:
        b = baseline_perf.loc[name]
        rows.append(
            {
                "comparison": f"WeakAlpha_selected_minus_{name}",
                "annualized_return_diff": float(selected_perf["annualized_return"] - b["annualized_return"]),
                "annualized_volatility_diff": float(selected_perf["annualized_volatility"] - b["annualized_volatility"]),
                "sharpe_diff": float(selected_perf["sharpe"] - b["sharpe"]),
                "max_drawdown_diff": float(selected_perf["max_drawdown"] - b["max_drawdown"]),
                "final_wealth_ratio": float(selected_perf["final_wealth"] / b["final_wealth"]),
                "avg_rebalance_turnover_diff": float(selected_perf["avg_rebalance_turnover"] - b["avg_rebalance_turnover"]),
                "total_cost_diff": float(selected_perf["total_cost"] - b["total_cost"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns, regime_features, regime_records = load_inputs()
    baseline_perf, baseline_returns, _ = load_existing_results()
    regime_lookup = build_regime_lookup(regime_records)

    common_index = returns.index.intersection(regime_features.index)
    returns = returns.loc[common_index].copy()
    returns = returns.loc[returns.index >= START_DATE].copy()
    regime_lookup = regime_lookup.loc[regime_lookup.index.intersection(returns.index)].copy()

    candidate_rows = []
    candidate_outputs = {}
    for alpha in ALPHA_GRID:
        result = backtest_alpha_setting(
            returns=returns,
            regime_lookup=regime_lookup,
            alpha_shrink=alpha,
            gamma=GAMMA,
            lookback_mu=LOOKBACK_MU,
            lookback_cov=LOOKBACK_COV,
            base_trade_cost=BASE_TRADE_COST,
        )
        candidate_outputs[alpha] = result
        row = dict(result["performance"])
        row["strategy"] = f"WeakAlpha_alpha_{alpha:.2f}"
        candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows).sort_values(["sharpe", "annualized_return"], ascending=False)
    selected_alpha = float(candidates.iloc[0]["alpha_shrink"])
    selected = candidate_outputs[selected_alpha]
    selected_perf = pd.Series(selected["performance"])

    # Selected strategy outputs.
    selected["weights"].to_csv(OUTPUT_DIR / "weak_alpha_selected_daily_weights.csv")
    selected["strategy_returns"].to_frame().to_csv(OUTPUT_DIR / "weak_alpha_selected_daily_returns.csv")
    selected["turnover"].to_frame().to_csv(OUTPUT_DIR / "weak_alpha_selected_daily_turnover.csv")
    selected["costs"].to_frame().to_csv(OUTPUT_DIR / "weak_alpha_selected_daily_costs.csv")

    # Candidate table and comparison table.
    candidates.to_csv(OUTPUT_DIR / "weak_alpha_candidate_performance.csv", index=False)
    comparison = summarize_against_baselines(selected_perf, baseline_perf)
    comparison.to_csv(OUTPUT_DIR / "weak_alpha_excess_vs_baselines.csv", index=False)

    # Helpful benchmark snapshot for this folder.
    benchmark_slice = baseline_perf.set_index("strategy").loc[
        ["EqualWeight_weekly", "Markowitz_CVXPY_weekly", "HMM_MPC_CVXPY_weekly"]
    ].reset_index()
    benchmark_slice.to_csv(OUTPUT_DIR / "baseline_snapshot.csv", index=False)

    summary = {
        "selected_alpha_shrink": selected_alpha,
        "gamma": GAMMA,
        "lookback_mu": LOOKBACK_MU,
        "lookback_cov": LOOKBACK_COV,
        "base_trade_cost": BASE_TRADE_COST,
        "selected_sharpe": selected_perf["sharpe"],
        "selected_annualized_return": selected_perf["annualized_return"],
        "selected_max_drawdown": selected_perf["max_drawdown"],
        "selected_total_cost": selected_perf["total_cost"],
    }
    pd.Series(summary).to_csv(OUTPUT_DIR / "weak_alpha_selected_summary.csv", header=False)

    print("Weak-alpha experiment finished.")
    print("Selected alpha shrink:", selected_alpha)
    print(candidates.head(5).to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
