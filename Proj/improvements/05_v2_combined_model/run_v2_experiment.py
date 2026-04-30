from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cvxpy as cp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

ASSETS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
REFERENCE_STRATEGIES = [
    "EqualWeight_weekly",
    "Markowitz_CVXPY_weekly",
    "HMM_MPC_CVXPY_weekly",
]

START_DATE = pd.Timestamp("2013-01-08")
VALIDATION_START = pd.Timestamp("2013-01-08")
VALIDATION_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

LOOKBACK = 252
MOMENTUM_SKIP_DAYS = 21
COV_SHRINKAGE = 0.15
BASE_TRADE_COST = 0.0005
MPC_HORIZON = 5
MAX_WEIGHT = 0.45
TRADE_QUAD_PENALTY = 1e-3
COV_RIDGE = 1e-8


@dataclass(frozen=True)
class V2Params:
    gamma: float
    alpha_shrink: float
    turnover_cap: float
    stress_multiplier: float


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    daily_weights: pd.DataFrame
    rebalance_records: pd.DataFrame
    forecast_records: pd.DataFrame


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_returns() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "returns.csv"
    returns = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"})
    returns = returns.set_index("date").sort_index()
    missing = [asset for asset in ASSETS if asset not in returns.columns]
    if missing:
        raise ValueError(f"returns.csv missing required asset columns: {missing}")
    returns = returns[ASSETS].astype(float)
    # Keep pre-2013 observations for rolling forecasts/covariances. Backtest
    # outputs are still sliced by start_date/end_date inside run_backtest.
    return returns.loc[returns.index <= TEST_END]


def load_regime_records() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "tables" / "hmm_regime_records.csv"
    regimes = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    numeric_cols = [
        "prob_calm",
        "prob_transition",
        "prob_stress",
        "next_prob_calm",
        "next_prob_transition",
        "next_prob_stress",
        "next_cost_multiplier",
    ]
    for col in numeric_cols:
        if col in regimes.columns:
            regimes[col] = pd.to_numeric(regimes[col], errors="coerce")
    return regimes


def load_reference_series() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    returns = pd.read_csv(table_dir / "strategy_daily_returns.csv", index_col=0, parse_dates=True).sort_index()
    costs = pd.read_csv(table_dir / "strategy_daily_costs.csv", index_col=0, parse_dates=True).sort_index()
    turnover = pd.read_csv(table_dir / "strategy_daily_turnover.csv", index_col=0, parse_dates=True).sort_index()
    return returns, costs, turnover


def winsorize_series(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    if series.dropna().empty:
        return series.fillna(0.0)
    return series.clip(lower=series.quantile(lower_q), upper=series.quantile(upper_q))


def rescale_to_reference(signal: pd.Series, reference: pd.Series) -> pd.Series:
    signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    reference = reference.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal = winsorize_series(signal)

    ref_std = float(reference.std(ddof=0))
    sig_std = float(signal.std(ddof=0))
    if not np.isfinite(ref_std) or ref_std <= 1e-12 or not np.isfinite(sig_std) or sig_std <= 1e-12:
        return reference.copy()

    centered = signal - float(signal.mean())
    scaled = centered / sig_std * ref_std
    out = scaled + float(reference.mean())
    clip_level = 3.0 * ref_std
    return out.clip(lower=float(reference.mean()) - clip_level, upper=float(reference.mean()) + clip_level)


def robust_momentum_forecast(history: pd.DataFrame, alpha_shrink: float) -> pd.Series:
    recent = history.tail(LOOKBACK)
    reference = recent.mean(axis=0)
    if len(recent) > MOMENTUM_SKIP_DAYS:
        signal_window = recent.iloc[:-MOMENTUM_SKIP_DAYS]
    else:
        signal_window = recent

    momentum_signal = np.log1p(signal_window.clip(lower=-0.999)).mean(axis=0)
    scaled = rescale_to_reference(momentum_signal, reference)
    centered = scaled - float(scaled.mean())
    return alpha_shrink * centered


def nearest_psd(matrix: np.ndarray, min_eig: float = COV_RIDGE) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, min_eig)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (psd + psd.T)


def regime_probabilities(date: pd.Timestamp, regime_records: pd.DataFrame) -> np.ndarray:
    if date in regime_records.index:
        row = regime_records.loc[date]
        cols = ["next_prob_calm", "next_prob_transition", "next_prob_stress"]
        if all(col in row.index for col in cols) and row[cols].notna().all():
            q = row[cols].to_numpy(dtype=float)
        else:
            q = row[["prob_calm", "prob_transition", "prob_stress"]].to_numpy(dtype=float)
    else:
        q = np.array([1.0, 0.0, 0.0], dtype=float)

    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.maximum(q, 0.0)
    total = float(q.sum())
    if total <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return q / total


def regime_cost_multiplier(q: np.ndarray, stress_multiplier: float) -> float:
    return float(1.0 + 0.5 * q[1] + stress_multiplier * q[2])


def compute_covariance(history: pd.DataFrame, q: np.ndarray) -> np.ndarray:
    cov = history.tail(LOOKBACK).cov().to_numpy(dtype=float)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - COV_SHRINKAGE) * cov + COV_SHRINKAGE * diag
    risk_scale = 1.0 + 0.25 * q[1] + 0.75 * q[2]
    return nearest_psd(shrunk * risk_scale + COV_RIDGE * np.eye(len(ASSETS)))


def solve_v2_mpc(
    mu: pd.Series,
    cov: np.ndarray,
    current_weights: np.ndarray,
    q: np.ndarray,
    params: V2Params,
) -> Tuple[np.ndarray, str]:
    n_assets = len(ASSETS)
    x = cp.Variable((MPC_HORIZON, n_assets))
    buys = cp.Variable((MPC_HORIZON, n_assets), nonneg=True)
    sells = cp.Variable((MPC_HORIZON, n_assets), nonneg=True)

    objective_terms = []
    constraints = []
    previous = current_weights
    cost_per_dollar = BASE_TRADE_COST * regime_cost_multiplier(q, params.stress_multiplier)
    cov_psd = cp.psd_wrap(cov)
    mu_values = mu.to_numpy(dtype=float)

    for h in range(MPC_HORIZON):
        trade = buys[h] - sells[h]
        turnover = 0.5 * cp.sum(buys[h] + sells[h])
        constraints += [
            x[h] - previous == trade,
            cp.sum(x[h]) == 1.0,
            x[h] >= 0.0,
            x[h] <= MAX_WEIGHT,
            turnover <= params.turnover_cap,
        ]
        objective_terms.append(
            mu_values @ x[h]
            - 0.5 * params.gamma * cp.quad_form(x[h], cov_psd)
            - cost_per_dollar * cp.sum(buys[h] + sells[h])
            - 0.5 * TRADE_QUAD_PENALTY * cp.sum_squares(trade)
        )
        previous = x[h]

    problem = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints)
    status = "not_solved"
    for solver in ["OSQP", "CLARABEL", "ECOS", "SCS"]:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        status = problem.status or "unknown"
        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
            weights = np.asarray(x.value[0], dtype=float).reshape(-1)
            weights = np.clip(weights, 0.0, None)
            total = float(weights.sum())
            if np.isfinite(total) and total > 1e-12:
                weights = weights / total
                return weights, status

    return current_weights.copy(), f"fallback:{status}"


def run_backtest(
    returns: pd.DataFrame,
    regime_records: pd.DataFrame,
    params: V2Params,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    name: str = "V2_HMM_MPC",
) -> BacktestResult:
    dates = returns.index
    eval_dates = dates[(dates >= start_date) & (dates <= end_date)]
    rebalance_set = set(regime_records.index.intersection(dates))

    current_weights = np.ones(len(ASSETS), dtype=float) / len(ASSETS)
    return_rows: List[Tuple[pd.Timestamp, float]] = []
    turnover_rows: List[Tuple[pd.Timestamp, float]] = []
    cost_rows: List[Tuple[pd.Timestamp, float]] = []
    daily_weight_rows: List[pd.Series] = []
    rebalance_rows: List[pd.Series] = []
    forecast_rows: List[pd.Series] = []

    for date in eval_dates:
        loc = dates.get_loc(date)
        prev_date = dates[loc - 1] if loc > 0 else None
        cost = 0.0
        turnover = 0.0

        if prev_date is not None and prev_date in rebalance_set and prev_date >= start_date:
            history = returns.loc[:prev_date].tail(LOOKBACK)
            if len(history) >= LOOKBACK:
                q = regime_probabilities(prev_date, regime_records)
                mu = robust_momentum_forecast(history, params.alpha_shrink)
                cov = compute_covariance(history, q)
                target_weights, status = solve_v2_mpc(mu, cov, current_weights, q, params)
                traded = np.abs(target_weights - current_weights)
                turnover = 0.5 * float(traded.sum())
                cost_per_dollar = BASE_TRADE_COST * regime_cost_multiplier(q, params.stress_multiplier)
                cost = cost_per_dollar * float(traded.sum())
                current_weights = target_weights

                rebalance_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "effective_date": date,
                            "solver_status": status,
                            "turnover": turnover,
                            "transaction_cost": cost,
                            "cost_multiplier": regime_cost_multiplier(q, params.stress_multiplier),
                            "prob_calm": q[0],
                            "prob_transition": q[1],
                            "prob_stress": q[2],
                            **{asset: current_weights[i] for i, asset in enumerate(ASSETS)},
                        }
                    )
                )
                forecast_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "effective_date": date,
                            "alpha_shrink": params.alpha_shrink,
                            **{asset: mu.loc[asset] for asset in ASSETS},
                        }
                    )
                )
            else:
                rebalance_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "effective_date": date,
                            "solver_status": "insufficient_history",
                            "turnover": 0.0,
                            "transaction_cost": 0.0,
                            "cost_multiplier": np.nan,
                            "prob_calm": np.nan,
                            "prob_transition": np.nan,
                            "prob_stress": np.nan,
                            **{asset: current_weights[i] for i, asset in enumerate(ASSETS)},
                        }
                    )
                )

        asset_returns = returns.loc[date, ASSETS].to_numpy(dtype=float)
        gross_return = float(current_weights @ asset_returns)
        net_return = (1.0 - cost) * (1.0 + gross_return) - 1.0

        denom = 1.0 + gross_return
        if denom > 0:
            current_weights = current_weights * (1.0 + asset_returns) / denom
            current_weights = current_weights / current_weights.sum()

        return_rows.append((date, net_return))
        turnover_rows.append((date, turnover))
        cost_rows.append((date, cost))
        daily_weight_rows.append(pd.Series(current_weights, index=ASSETS, name=date))

    ret = pd.Series(dict(return_rows), name=name).sort_index()
    turnover_series = pd.Series(dict(turnover_rows), name=name).sort_index()
    cost_series = pd.Series(dict(cost_rows), name=name).sort_index()
    daily_weights = pd.DataFrame(daily_weight_rows)
    daily_weights.index.name = "date"
    rebalance_records = pd.DataFrame(rebalance_rows)
    forecast_records = pd.DataFrame(forecast_rows)
    return BacktestResult(
        name=name,
        returns=ret,
        turnover=turnover_series,
        costs=cost_series,
        daily_weights=daily_weights,
        rebalance_records=rebalance_records,
        forecast_records=forecast_records,
    )


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) <= 1:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(252.0))


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series, turnover: pd.Series, costs: pd.Series) -> Dict[str, float]:
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    return {
        "start": returns.dropna().index.min().date().isoformat() if not returns.dropna().empty else "",
        "end": returns.dropna().index.max().date().isoformat() if not returns.dropna().empty else "",
        "observations": int(returns.dropna().shape[0]),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": float(ann_ret / ann_vol) if np.isfinite(ann_vol) and ann_vol > 0 else np.nan,
        "max_drawdown": max_drawdown(returns),
        "final_wealth": float((1.0 + returns.dropna()).prod()) if not returns.dropna().empty else np.nan,
        "avg_daily_turnover": float(turnover.reindex(returns.index).fillna(0.0).mean()) if not returns.empty else np.nan,
        "avg_rebalance_turnover": float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0,
        "total_cost": float(costs.reindex(returns.index).fillna(0.0).sum()) if not returns.empty else np.nan,
    }


def metrics_table(
    strategy_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    strategies: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        if strategy not in strategy_returns.columns:
            continue
        ret = strategy_returns[strategy].dropna()
        t = turnover[strategy].reindex(ret.index).fillna(0.0) if strategy in turnover.columns else pd.Series(0.0, index=ret.index)
        c = costs[strategy].reindex(ret.index).fillna(0.0) if strategy in costs.columns else pd.Series(0.0, index=ret.index)
        row = performance_metrics(ret, t, c)
        row["strategy"] = strategy
        rows.append(row)
    cols = [
        "strategy",
        "start",
        "end",
        "observations",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "final_wealth",
        "avg_daily_turnover",
        "avg_rebalance_turnover",
        "total_cost",
    ]
    return pd.DataFrame(rows)[cols]


def summarize_excess(performance: pd.DataFrame, returns: pd.DataFrame, v2_name: str) -> pd.DataFrame:
    perf = performance.set_index("strategy")
    rows = []
    for baseline in REFERENCE_STRATEGIES:
        if baseline not in perf.index or v2_name not in perf.index or baseline not in returns.columns:
            continue
        active = (returns[v2_name] - returns[baseline]).dropna()
        active_return = float(active.mean() * 252.0)
        tracking_error = float(active.std(ddof=1) * math.sqrt(252.0)) if len(active) > 1 else np.nan
        rows.append(
            {
                "comparison": f"{v2_name} minus {baseline}",
                "annualized_active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": active_return / tracking_error if tracking_error and tracking_error > 0 else np.nan,
                "annualized_return_difference": perf.loc[v2_name, "annualized_return"] - perf.loc[baseline, "annualized_return"],
                "sharpe_difference": perf.loc[v2_name, "sharpe"] - perf.loc[baseline, "sharpe"],
                "max_drawdown_improvement": perf.loc[v2_name, "max_drawdown"] - perf.loc[baseline, "max_drawdown"],
                "final_wealth_ratio": perf.loc[v2_name, "final_wealth"] / perf.loc[baseline, "final_wealth"],
                "avg_rebalance_turnover_difference": (
                    perf.loc[v2_name, "avg_rebalance_turnover"] - perf.loc[baseline, "avg_rebalance_turnover"]
                ),
                "total_cost_difference": perf.loc[v2_name, "total_cost"] - perf.loc[baseline, "total_cost"],
            }
        )
    return pd.DataFrame(rows)


def make_grid() -> List[V2Params]:
    gammas = [3.0, 5.0, 8.0]
    alpha_shrinks = [0.25, 0.50, 0.75]
    turnover_caps = [0.05, 0.10]
    stress_multipliers = [1.0, 3.0]
    return [
        V2Params(gamma=gamma, alpha_shrink=alpha, turnover_cap=cap, stress_multiplier=stress)
        for gamma, alpha, cap, stress in product(gammas, alpha_shrinks, turnover_caps, stress_multipliers)
    ]


def formatted_table(df: pd.DataFrame, decimals: int = 4) -> str:
    if df.empty:
        return "No rows."
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")
    try:
        return out.to_markdown(index=False)
    except Exception:
        return out.to_string(index=False)


def wealth_frame(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).cumprod()


def drawdown_frame(returns: pd.DataFrame) -> pd.DataFrame:
    wealth = wealth_frame(returns)
    return wealth / wealth.cummax() - 1.0


def plot_cumulative_wealth(returns: pd.DataFrame, path: Path, title: str) -> None:
    wealth = wealth_frame(returns)
    plt.figure(figsize=(12, 6))
    for col in wealth.columns:
        linewidth = 2.2 if col == "V2_HMM_MPC" else 1.4
        plt.plot(wealth.index, wealth[col], linewidth=linewidth, label=col)
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_drawdowns(returns: pd.DataFrame, path: Path, title: str) -> None:
    drawdowns = drawdown_frame(returns)
    plt.figure(figsize=(12, 5))
    for col in drawdowns.columns:
        linewidth = 2.0 if col == "V2_HMM_MPC" else 1.2
        plt.plot(drawdowns.index, drawdowns[col], linewidth=linewidth, label=col)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_excess_wealth(returns: pd.DataFrame, path: Path, title: str) -> pd.DataFrame:
    wealth = wealth_frame(returns)
    excess = pd.DataFrame(index=wealth.index)
    for baseline in REFERENCE_STRATEGIES:
        if baseline in wealth.columns and "V2_HMM_MPC" in wealth.columns:
            excess[f"vs_{baseline}"] = wealth["V2_HMM_MPC"] / wealth[baseline] - 1.0

    plt.figure(figsize=(12, 5))
    for col in excess.columns:
        plt.plot(excess.index, excess[col], linewidth=1.8, label=col.replace("vs_", "V2 / "))
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.title(title)
    plt.ylabel("Relative wealth minus 1")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
    return excess


def plot_rolling_active_return(returns: pd.DataFrame, path: Path) -> None:
    active = pd.DataFrame(index=returns.index)
    for baseline in REFERENCE_STRATEGIES:
        if baseline in returns.columns:
            active[f"V2 minus {baseline}"] = returns["V2_HMM_MPC"] - returns[baseline]
    rolling = active.rolling(252).mean() * 252.0
    plt.figure(figsize=(12, 5))
    for col in rolling.columns:
        plt.plot(rolling.index, rolling[col], linewidth=1.6, label=col)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Rolling 1Y Active Return, Test Sample")
    plt.ylabel("Annualized active return")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_weights(weights: pd.DataFrame, path: Path) -> None:
    if weights.empty:
        return
    monthly = weights[ASSETS].resample("M").last().dropna(how="all")
    plt.figure(figsize=(12, 6))
    plt.stackplot(monthly.index, monthly[ASSETS].T, labels=ASSETS, alpha=0.92)
    plt.title("V2 Monthly Portfolio Weights")
    plt.ylabel("Weight")
    plt.ylim(0, 1)
    plt.grid(alpha=0.20)
    plt.legend(ncol=3, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_turnover_cost(daily: pd.DataFrame, path: Path) -> None:
    monthly = daily[["turnover", "cost"]].resample("M").sum()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(monthly.index, monthly["turnover"], width=20, alpha=0.55, label="Monthly turnover")
    ax1.set_ylabel("Monthly turnover")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["cost"], color="tab:red", linewidth=1.5, label="Monthly cost")
    ax2.set_ylabel("Monthly transaction cost")
    ax1.set_title("V2 Turnover and Transaction Costs")
    ax1.grid(alpha=0.20)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_validation_grid(grid: pd.DataFrame, path: Path) -> None:
    top = grid.sort_values(["sharpe", "annualized_return"], ascending=False).head(12).copy()
    labels = [
        f"g={row.gamma:g}, a={row.alpha_shrink:g}, cap={row.turnover_cap:g}, s={row.stress_multiplier:g}"
        for row in top.itertuples(index=False)
    ]
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top)), top["sharpe"], color="#4c78a8", alpha=0.85)
    plt.yticks(range(len(top)), labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Validation Sharpe")
    plt.title("Top V2 Validation Grid Candidates")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def combine_with_references(
    v2: BacktestResult,
    ref_returns: pd.DataFrame,
    ref_costs: pd.DataFrame,
    ref_turnover: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [col for col in REFERENCE_STRATEGIES if col in ref_returns.columns]
    idx = ref_returns.loc[(ref_returns.index >= start_date) & (ref_returns.index <= end_date)].index
    idx = idx.intersection(v2.returns.index)
    combined_returns = ref_returns.loc[idx, cols].copy()
    combined_returns["V2_HMM_MPC"] = v2.returns.reindex(idx)

    combined_costs = ref_costs.loc[idx, cols].copy()
    combined_costs["V2_HMM_MPC"] = v2.costs.reindex(idx).fillna(0.0)

    combined_turnover = ref_turnover.loc[idx, cols].copy()
    combined_turnover["V2_HMM_MPC"] = v2.turnover.reindex(idx).fillna(0.0)
    return combined_returns, combined_turnover, combined_costs


def write_readme(
    selected: V2Params,
    validation_metrics: pd.DataFrame,
    test_performance: pd.DataFrame,
    test_excess: pd.DataFrame,
    full_performance: pd.DataFrame,
    full_excess: pd.DataFrame,
    grid: pd.DataFrame,
) -> None:
    selected_row = {
        "gamma": selected.gamma,
        "alpha_shrink": selected.alpha_shrink,
        "turnover_cap": selected.turnover_cap,
        "stress_multiplier": selected.stress_multiplier,
    }
    top_grid = grid.sort_values(["sharpe", "annualized_return"], ascending=False).head(8)
    v2_test = test_performance.set_index("strategy").loc["V2_HMM_MPC"]

    readme = f"""# V2 Combined HMM-MPC Experiment

## 改进目标

V2 把前三个可行方向合并为一个新的实证版本：方向 1 的稳健 12-1 截面动量预期收益、方向 2 的 weak-alpha 缩放、方向 3 的验证集调参。该文件夹是独立实验，不修改项目原始 baseline 或原始 HMM-MPC 代码。

## 模型设定

- 资产池：`{", ".join(ASSETS)}`。
- 预测信号：每次调仓在历史 `{LOOKBACK}` 个交易日内计算 12-1 动量，即跳过最近 `{MOMENTUM_SKIP_DAYS}` 个交易日，只使用更早窗口的收益信息。
- weak-alpha：对截面动量信号做去均值和缩放后乘以 `alpha_shrink`，避免把噪声均值当成强 alpha。
- 风险估计：滚动 `{LOOKBACK}` 日协方差，使用 `{COV_SHRINKAGE:.0%}` 对角收缩，并根据 HMM 下一期 stress/transition 概率做温和风险放大。
- 优化器：CVXPY 多期凸优化，MPC horizon 为 `{MPC_HORIZON}` 个交易日，约束为 long-only、full investment、单资产权重上限 `{MAX_WEIGHT:.0%}`、每次调仓 turnover cap。
- 交易成本：基础成本为 `{BASE_TRADE_COST:.2%}` 每美元交易额，并随 HMM stress 概率由 `stress_multiplier` 放大。

## 防止未来数据泄漏

- 验证集：`{VALIDATION_START.date()}` 至 `{VALIDATION_END.date()}`。
- 测试集：`{TEST_START.date()}` 至 `{TEST_END.date()}`。
- 参数只按照验证集 Sharpe 选择；测试集结果使用冻结参数。
- 每个测试日的组合权重只由前一交易日及以前的数据决定。调仓决策日为 `decision_date`，实际生效日为下一交易日 `effective_date`。
- HMM regime records 使用项目原始 walk-forward 输出；V2 只读取相应调仓日已经生成的下一期概率，不重新用测试期未来信息拟合参数。

## 选中的 V2 参数

{formatted_table(pd.DataFrame([selected_row]))}

验证集选中参数的表现：

{formatted_table(validation_metrics)}

## 测试集表现

{formatted_table(test_performance)}

测试集相对 baseline 和原始模型的超额表现：

{formatted_table(test_excess)}

## 全样本描述性结果

全样本结果用于和前面实验保持口径一致，但不用于选参。

{formatted_table(full_performance)}

全样本超额表现：

{formatted_table(full_excess)}

## 结果解读

V2 测试集年化收益为 `{v2_test['annualized_return']:.2%}`，Sharpe 为 `{v2_test['sharpe']:.4f}`，最大回撤为 `{v2_test['max_drawdown']:.2%}`。与原始 HMM-MPC 相比，V2 的核心变化不是提高模型复杂度，而是降低均值估计噪声：使用更稳的 12-1 截面动量替代滚动均值，再通过 weak-alpha 缩放控制信号强度，并让风险厌恶、换手约束、stress 成本放大都由验证集决定。

如果 V2 相对等权仍有阶段性落后，主要原因通常来自两点：第一，行业 ETF 的截面机会本身在部分年份较弱；第二，long-only sector rotation 在市场由少数高 beta 行业主导时容易错失极端上涨。但相对原始 HMM-MPC，V2 的收益来源更清晰，调仓强度更可控，也更适合作为后续论文实证中的改进模型。

## 主要输出

- `results/tables/validation_grid_results.csv`
- `results/tables/selected_parameters.csv`
- `results/tables/test_performance.csv`
- `results/tables/test_excess_vs_references.csv`
- `results/tables/full_sample_performance.csv`
- `results/tables/full_sample_excess_vs_references.csv`
- `results/tables/v2_test_daily_series.csv`
- `results/tables/v2_full_daily_series.csv`
- `results/tables/v2_daily_weights_full.csv`
- `results/figures/test_cumulative_wealth.png`
- `results/figures/test_drawdowns.png`
- `results/figures/test_excess_wealth_curves.png`
- `results/figures/rolling_1y_active_return_test.png`
- `results/figures/full_cumulative_wealth.png`
- `results/figures/v2_monthly_weights.png`
- `results/figures/v2_turnover_and_costs.png`
- `results/figures/validation_top_grid.png`

## 验证集排名前列参数

{formatted_table(top_grid)}
"""
    (HERE / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    returns = load_returns()
    regime_records = load_regime_records()
    ref_returns, ref_costs, ref_turnover = load_reference_series()

    grid_rows = []
    grid = make_grid()
    for i, params in enumerate(grid, start=1):
        result = run_backtest(
            returns=returns,
            regime_records=regime_records,
            params=params,
            start_date=VALIDATION_START,
            end_date=VALIDATION_END,
        )
        metrics = performance_metrics(result.returns, result.turnover, result.costs)
        metrics.update(asdict(params))
        metrics["strategy"] = "V2_HMM_MPC"
        grid_rows.append(metrics)
        print(f"Validation grid {i:02d}/{len(grid)}: {asdict(params)} Sharpe={metrics['sharpe']:.4f}")

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["sharpe", "annualized_return", "max_drawdown"],
        ascending=[False, False, False],
    )
    grid_df.to_csv(TABLES_DIR / "validation_grid_results.csv", index=False)

    selected = V2Params(
        gamma=float(grid_df.iloc[0]["gamma"]),
        alpha_shrink=float(grid_df.iloc[0]["alpha_shrink"]),
        turnover_cap=float(grid_df.iloc[0]["turnover_cap"]),
        stress_multiplier=float(grid_df.iloc[0]["stress_multiplier"]),
    )
    pd.DataFrame([asdict(selected)]).to_csv(TABLES_DIR / "selected_parameters.csv", index=False)

    validation_v2 = run_backtest(returns, regime_records, selected, VALIDATION_START, VALIDATION_END)
    test_v2 = run_backtest(returns, regime_records, selected, TEST_START, TEST_END)
    full_v2 = run_backtest(returns, regime_records, selected, START_DATE, TEST_END)

    validation_metrics = pd.DataFrame(
        [
            {
                "strategy": "V2_HMM_MPC",
                **performance_metrics(validation_v2.returns, validation_v2.turnover, validation_v2.costs),
                **asdict(selected),
            }
        ]
    )
    validation_metrics.to_csv(TABLES_DIR / "validation_selected_metrics.csv", index=False)

    test_returns, test_turnover, test_costs = combine_with_references(
        test_v2, ref_returns, ref_costs, ref_turnover, TEST_START, TEST_END
    )
    full_returns, full_turnover, full_costs = combine_with_references(
        full_v2, ref_returns, ref_costs, ref_turnover, START_DATE, TEST_END
    )

    strategy_order = [*REFERENCE_STRATEGIES, "V2_HMM_MPC"]
    test_performance = metrics_table(test_returns, test_turnover, test_costs, strategy_order)
    full_performance = metrics_table(full_returns, full_turnover, full_costs, strategy_order)
    test_excess = summarize_excess(test_performance, test_returns, "V2_HMM_MPC")
    full_excess = summarize_excess(full_performance, full_returns, "V2_HMM_MPC")

    test_performance.to_csv(TABLES_DIR / "test_performance.csv", index=False)
    full_performance.to_csv(TABLES_DIR / "full_sample_performance.csv", index=False)
    test_excess.to_csv(TABLES_DIR / "test_excess_vs_references.csv", index=False)
    full_excess.to_csv(TABLES_DIR / "full_sample_excess_vs_references.csv", index=False)
    test_returns.to_csv(TABLES_DIR / "test_strategy_daily_returns.csv")
    full_returns.to_csv(TABLES_DIR / "full_strategy_daily_returns.csv")

    pd.DataFrame(
        {
            "net_return": test_v2.returns,
            "turnover": test_v2.turnover,
            "cost": test_v2.costs,
            "wealth": (1.0 + test_v2.returns).cumprod(),
        }
    ).to_csv(TABLES_DIR / "v2_test_daily_series.csv")
    pd.DataFrame(
        {
            "net_return": full_v2.returns,
            "turnover": full_v2.turnover,
            "cost": full_v2.costs,
            "wealth": (1.0 + full_v2.returns).cumprod(),
        }
    ).to_csv(TABLES_DIR / "v2_full_daily_series.csv")
    full_v2.daily_weights.to_csv(TABLES_DIR / "v2_daily_weights_full.csv")
    test_v2.daily_weights.to_csv(TABLES_DIR / "v2_daily_weights_test.csv")
    full_v2.rebalance_records.to_csv(TABLES_DIR / "v2_rebalance_records_full.csv", index=False)
    test_v2.rebalance_records.to_csv(TABLES_DIR / "v2_rebalance_records_test.csv", index=False)
    full_v2.forecast_records.to_csv(TABLES_DIR / "v2_forecast_records_full.csv", index=False)

    test_excess_curves = plot_excess_wealth(
        test_returns,
        FIGURES_DIR / "test_excess_wealth_curves.png",
        "V2 Relative Wealth vs References, Test Sample",
    )
    test_excess_curves.to_csv(TABLES_DIR / "test_excess_wealth_curves.csv")
    plot_cumulative_wealth(test_returns, FIGURES_DIR / "test_cumulative_wealth.png", "Strategy Cumulative Wealth, Test Sample")
    plot_drawdowns(test_returns, FIGURES_DIR / "test_drawdowns.png", "Strategy Drawdowns, Test Sample")
    plot_rolling_active_return(test_returns, FIGURES_DIR / "rolling_1y_active_return_test.png")
    plot_cumulative_wealth(full_returns, FIGURES_DIR / "full_cumulative_wealth.png", "Strategy Cumulative Wealth, Full Sample")
    plot_drawdowns(full_returns, FIGURES_DIR / "full_drawdowns.png", "Strategy Drawdowns, Full Sample")
    plot_weights(full_v2.daily_weights, FIGURES_DIR / "v2_monthly_weights.png")
    plot_turnover_cost(
        pd.DataFrame({"turnover": full_v2.turnover, "cost": full_v2.costs}),
        FIGURES_DIR / "v2_turnover_and_costs.png",
    )
    plot_validation_grid(grid_df, FIGURES_DIR / "validation_top_grid.png")

    manifest = {
        "selected_parameters": asdict(selected),
        "validation_period": [str(VALIDATION_START.date()), str(VALIDATION_END.date())],
        "test_period": [str(TEST_START.date()), str(TEST_END.date())],
        "no_future_leakage_controls": [
            "parameters selected only on validation grid",
            "rebalance decisions use returns through the prior trading date",
            "test set evaluated after freezing selected parameters",
        ],
        "outputs": {
            "tables": sorted(str(p.relative_to(HERE)) for p in TABLES_DIR.glob("*.csv")),
            "figures": sorted(str(p.relative_to(HERE)) for p in FIGURES_DIR.glob("*.png")),
        },
    }
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_readme(
        selected=selected,
        validation_metrics=validation_metrics,
        test_performance=test_performance,
        test_excess=test_excess,
        full_performance=full_performance,
        full_excess=full_excess,
        grid=grid_df,
    )

    print("\nSelected V2 parameters:")
    print(pd.DataFrame([asdict(selected)]).to_string(index=False))
    print("\nTest performance:")
    print(test_performance.to_string(index=False))
    print("\nTest excess:")
    print(test_excess.to_string(index=False))
    print(f"\nWrote V2 outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
