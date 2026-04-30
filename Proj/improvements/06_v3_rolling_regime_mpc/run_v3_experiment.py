from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
TEST_START = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2024-12-31")

LOOKBACK = 252
MOMENTUM_SKIP_DAYS = 21
SELECTION_WINDOW = 756
MIN_SELECTION_OBS = 252
MPC_HORIZON = 3
MAX_WEIGHT = 0.60
COV_SHRINKAGE = 0.15
BASE_TRADE_COST = 0.0005
TRADE_QUAD_PENALTY = 2e-4
COV_RIDGE = 1e-8
STRESS_MULTIPLIER = 1.0


@dataclass(frozen=True)
class CandidateParams:
    gamma: float
    base_alpha: float
    turnover_cap: float
    stress_multiplier: float = STRESS_MULTIPLIER

    @property
    def name(self) -> str:
        return (
            f"g{self.gamma:g}_a{int(round(self.base_alpha * 100)):03d}"
            f"_cap{int(round(self.turnover_cap * 100)):02d}_s{self.stress_multiplier:g}"
        )


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
        raise ValueError(f"returns.csv missing required columns: {missing}")
    return returns.loc[returns.index <= END_DATE, ASSETS].astype(float)


def load_regime_records() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "tables" / "hmm_regime_records.csv"
    records = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    for col in [
        "prob_calm",
        "prob_transition",
        "prob_stress",
        "next_prob_calm",
        "next_prob_transition",
        "next_prob_stress",
        "next_cost_multiplier",
    ]:
        if col in records.columns:
            records[col] = pd.to_numeric(records[col], errors="coerce")
    return records


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


def effective_alpha(base_alpha: float, q: np.ndarray) -> float:
    regime_discount = 1.0 - 0.20 * float(q[1]) - 0.65 * float(q[2])
    return float(np.clip(base_alpha * regime_discount, 0.10, base_alpha))


def robust_momentum_forecast(history: pd.DataFrame, params: CandidateParams, q: np.ndarray) -> Tuple[pd.Series, float]:
    recent = history.tail(LOOKBACK)
    reference = recent.mean(axis=0)
    signal_window = recent.iloc[:-MOMENTUM_SKIP_DAYS] if len(recent) > MOMENTUM_SKIP_DAYS else recent
    momentum_signal = np.log1p(signal_window.clip(lower=-0.999)).mean(axis=0)
    scaled = rescale_to_reference(momentum_signal, reference)
    centered = scaled - float(scaled.mean())
    alpha = effective_alpha(params.base_alpha, q)
    return alpha * centered, alpha


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


def dominant_regime(q: np.ndarray) -> str:
    return ["calm", "transition", "stress"][int(np.argmax(q))]


def regime_cost_multiplier(q: np.ndarray, stress_multiplier: float) -> float:
    return float(1.0 + 0.5 * q[1] + stress_multiplier * q[2])


def compute_covariance(history: pd.DataFrame, q: np.ndarray) -> np.ndarray:
    cov = history.tail(LOOKBACK).cov().to_numpy(dtype=float)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - COV_SHRINKAGE) * cov + COV_SHRINKAGE * diag
    risk_scale = 1.0 + 0.25 * q[1] + 0.75 * q[2]
    return nearest_psd(shrunk * risk_scale + COV_RIDGE * np.eye(len(ASSETS)))


def solve_mpc(
    mu: pd.Series,
    cov: np.ndarray,
    current_weights: np.ndarray,
    q: np.ndarray,
    params: CandidateParams,
) -> Tuple[np.ndarray, str]:
    n_assets = len(ASSETS)
    x = cp.Variable((MPC_HORIZON, n_assets))
    buys = cp.Variable((MPC_HORIZON, n_assets), nonneg=True)
    sells = cp.Variable((MPC_HORIZON, n_assets), nonneg=True)

    constraints = []
    objective_terms = []
    previous = current_weights
    mu_values = mu.to_numpy(dtype=float)
    cov_psd = cp.psd_wrap(cov)
    cost_per_dollar = BASE_TRADE_COST * regime_cost_multiplier(q, params.stress_multiplier)

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
                return weights / total, status
    return current_weights.copy(), f"fallback:{status}"


def run_fixed_param_backtest(
    returns: pd.DataFrame,
    regime_records: pd.DataFrame,
    params: CandidateParams,
    name: str,
) -> BacktestResult:
    dates = returns.index
    eval_dates = dates[(dates >= START_DATE) & (dates <= END_DATE)]
    rebalance_set = set(regime_records.index.intersection(dates))

    current_weights = np.ones(len(ASSETS), dtype=float) / len(ASSETS)
    ret_rows: List[Tuple[pd.Timestamp, float]] = []
    turnover_rows: List[Tuple[pd.Timestamp, float]] = []
    cost_rows: List[Tuple[pd.Timestamp, float]] = []
    weight_rows: List[pd.Series] = []
    rebalance_rows: List[pd.Series] = []
    forecast_rows: List[pd.Series] = []

    for date in eval_dates:
        loc = dates.get_loc(date)
        prev_date = dates[loc - 1] if loc > 0 else None
        cost = 0.0
        turnover = 0.0

        if prev_date is not None and prev_date in rebalance_set and prev_date >= START_DATE:
            history = returns.loc[:prev_date].tail(LOOKBACK)
            if len(history) >= LOOKBACK:
                q = regime_probabilities(prev_date, regime_records)
                mu, alpha = robust_momentum_forecast(history, params, q)
                cov = compute_covariance(history, q)
                target, status = solve_mpc(mu, cov, current_weights, q, params)
                traded = np.abs(target - current_weights)
                turnover = 0.5 * float(traded.sum())
                cost_per_dollar = BASE_TRADE_COST * regime_cost_multiplier(q, params.stress_multiplier)
                cost = cost_per_dollar * float(traded.sum())
                current_weights = target
                rebalance_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "effective_date": date,
                            "candidate": name,
                            "solver_status": status,
                            "turnover": turnover,
                            "transaction_cost": cost,
                            "effective_alpha": alpha,
                            "regime": dominant_regime(q),
                            "prob_calm": q[0],
                            "prob_transition": q[1],
                            "prob_stress": q[2],
                            **asdict(params),
                            **{asset: current_weights[i] for i, asset in enumerate(ASSETS)},
                        }
                    )
                )
                forecast_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "candidate": name,
                            "effective_alpha": alpha,
                            **{asset: mu.loc[asset] for asset in ASSETS},
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

        ret_rows.append((date, net_return))
        turnover_rows.append((date, turnover))
        cost_rows.append((date, cost))
        weight_rows.append(pd.Series(current_weights, index=ASSETS, name=date))

    daily_weights = pd.DataFrame(weight_rows)
    daily_weights.index.name = "date"
    return BacktestResult(
        name=name,
        returns=pd.Series(dict(ret_rows), name=name).sort_index(),
        turnover=pd.Series(dict(turnover_rows), name=name).sort_index(),
        costs=pd.Series(dict(cost_rows), name=name).sort_index(),
        daily_weights=daily_weights,
        rebalance_records=pd.DataFrame(rebalance_rows),
        forecast_records=pd.DataFrame(forecast_rows),
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
    return float((wealth / wealth.cummax() - 1.0).min())


def performance_metrics(returns: pd.Series, turnover: pd.Series, costs: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    turnover = turnover.reindex(returns.index).fillna(0.0)
    costs = costs.reindex(returns.index).fillna(0.0)
    return {
        "start": returns.index.min().date().isoformat() if not returns.empty else "",
        "end": returns.index.max().date().isoformat() if not returns.empty else "",
        "observations": int(len(returns)),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else np.nan,
        "max_drawdown": max_drawdown(returns),
        "final_wealth": float((1.0 + returns).prod()) if not returns.empty else np.nan,
        "avg_daily_turnover": float(turnover.mean()) if not returns.empty else np.nan,
        "avg_rebalance_turnover": float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0,
        "total_cost": float(costs.sum()) if not returns.empty else np.nan,
    }


def make_candidates() -> List[CandidateParams]:
    gammas = [5.0, 8.0]
    base_alphas = [0.25, 0.75, 1.00]
    turnover_caps = [0.10, 0.25]
    return [
        CandidateParams(gamma=gamma, base_alpha=alpha, turnover_cap=cap, stress_multiplier=STRESS_MULTIPLIER)
        for gamma, alpha, cap in product(gammas, base_alphas, turnover_caps)
    ]


def default_candidate(candidates: Sequence[CandidateParams]) -> CandidateParams:
    for params in candidates:
        if params.gamma == 5.0 and params.base_alpha == 1.0 and params.turnover_cap == 0.25:
            return params
    return candidates[-1]


def score_candidate(history: pd.Series, equal_weight: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([history, equal_weight], axis=1).dropna()
    aligned.columns = ["candidate", "equal_weight"]
    if len(aligned) < MIN_SELECTION_OBS:
        return {"score": -np.inf}

    r = aligned["candidate"]
    ew = aligned["equal_weight"]
    active = r - ew
    ann_ret = annualized_return(r)
    ann_vol = annualized_volatility(r)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    active_return = float(active.mean() * 252.0)
    tracking_error = float(active.std(ddof=1) * math.sqrt(252.0))
    information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0
    drawdown = max_drawdown(r)
    ew_drawdown = max_drawdown(ew)
    drawdown_improvement = drawdown - ew_drawdown

    score = float(sharpe + 0.40 * information_ratio + 2.00 * active_return + 0.15 * drawdown_improvement)
    return {
        "score": score,
        "selection_ann_return": ann_ret,
        "selection_sharpe": sharpe,
        "selection_active_return": active_return,
        "selection_information_ratio": information_ratio,
        "selection_drawdown_improvement": drawdown_improvement,
    }


def select_params(
    candidate_returns: pd.DataFrame,
    equal_weight_returns: pd.Series,
    candidates: Sequence[CandidateParams],
    prev_date: pd.Timestamp,
) -> Tuple[CandidateParams, Dict[str, float]]:
    end_loc = candidate_returns.index.searchsorted(prev_date, side="right")
    start_loc = max(0, end_loc - SELECTION_WINDOW)
    hist = candidate_returns.iloc[start_loc:end_loc]
    ew_hist = equal_weight_returns.reindex(hist.index)

    if len(hist) < MIN_SELECTION_OBS:
        params = default_candidate(candidates)
        return params, {
            "score": np.nan,
            "selector_reason": "default_until_min_history",
            "selection_observations": int(len(hist)),
        }

    rows = []
    by_name = {p.name: p for p in candidates}
    for name in hist.columns:
        score_data = score_candidate(hist[name], ew_hist)
        score_data["candidate"] = name
        rows.append(score_data)
    score_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    score_df = score_df.sort_values(["score", "selection_ann_return"], ascending=False)
    best_name = str(score_df.iloc[0]["candidate"])
    params = by_name[best_name]
    score_payload = score_df.iloc[0].to_dict()
    score_payload["selector_reason"] = "rolling_score"
    score_payload["selection_observations"] = int(len(hist))
    return params, score_payload


def run_v3_meta_strategy(
    returns: pd.DataFrame,
    regime_records: pd.DataFrame,
    candidates: Sequence[CandidateParams],
    candidate_returns: pd.DataFrame,
    equal_weight_returns: pd.Series,
) -> BacktestResult:
    dates = returns.index
    eval_dates = dates[(dates >= START_DATE) & (dates <= END_DATE)]
    rebalance_set = set(regime_records.index.intersection(dates))

    current_weights = np.ones(len(ASSETS), dtype=float) / len(ASSETS)
    ret_rows: List[Tuple[pd.Timestamp, float]] = []
    turnover_rows: List[Tuple[pd.Timestamp, float]] = []
    cost_rows: List[Tuple[pd.Timestamp, float]] = []
    weight_rows: List[pd.Series] = []
    rebalance_rows: List[pd.Series] = []
    forecast_rows: List[pd.Series] = []

    for date in eval_dates:
        loc = dates.get_loc(date)
        prev_date = dates[loc - 1] if loc > 0 else None
        cost = 0.0
        turnover = 0.0

        if prev_date is not None and prev_date in rebalance_set and prev_date >= START_DATE:
            history = returns.loc[:prev_date].tail(LOOKBACK)
            if len(history) >= LOOKBACK:
                params, score_payload = select_params(candidate_returns, equal_weight_returns, candidates, prev_date)
                q = regime_probabilities(prev_date, regime_records)
                mu, alpha = robust_momentum_forecast(history, params, q)
                cov = compute_covariance(history, q)
                target, status = solve_mpc(mu, cov, current_weights, q, params)
                traded = np.abs(target - current_weights)
                turnover = 0.5 * float(traded.sum())
                cost_per_dollar = BASE_TRADE_COST * regime_cost_multiplier(q, params.stress_multiplier)
                cost = cost_per_dollar * float(traded.sum())
                current_weights = target

                rebalance_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "effective_date": date,
                            "selected_candidate": params.name,
                            "solver_status": status,
                            "turnover": turnover,
                            "transaction_cost": cost,
                            "effective_alpha": alpha,
                            "regime": dominant_regime(q),
                            "prob_calm": q[0],
                            "prob_transition": q[1],
                            "prob_stress": q[2],
                            **asdict(params),
                            **score_payload,
                            **{asset: current_weights[i] for i, asset in enumerate(ASSETS)},
                        }
                    )
                )
                forecast_rows.append(
                    pd.Series(
                        {
                            "decision_date": prev_date,
                            "selected_candidate": params.name,
                            "effective_alpha": alpha,
                            **{asset: mu.loc[asset] for asset in ASSETS},
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

        ret_rows.append((date, net_return))
        turnover_rows.append((date, turnover))
        cost_rows.append((date, cost))
        weight_rows.append(pd.Series(current_weights, index=ASSETS, name=date))

    daily_weights = pd.DataFrame(weight_rows)
    daily_weights.index.name = "date"
    return BacktestResult(
        name="V3_Rolling_Regime_MPC",
        returns=pd.Series(dict(ret_rows), name="V3_Rolling_Regime_MPC").sort_index(),
        turnover=pd.Series(dict(turnover_rows), name="V3_Rolling_Regime_MPC").sort_index(),
        costs=pd.Series(dict(cost_rows), name="V3_Rolling_Regime_MPC").sort_index(),
        daily_weights=daily_weights,
        rebalance_records=pd.DataFrame(rebalance_rows),
        forecast_records=pd.DataFrame(forecast_rows),
    )


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


def summarize_excess(performance: pd.DataFrame, returns: pd.DataFrame, v3_name: str) -> pd.DataFrame:
    perf = performance.set_index("strategy")
    rows = []
    for baseline in REFERENCE_STRATEGIES:
        if baseline not in perf.index or v3_name not in perf.index or baseline not in returns.columns:
            continue
        active = (returns[v3_name] - returns[baseline]).dropna()
        active_return = float(active.mean() * 252.0)
        tracking_error = float(active.std(ddof=1) * math.sqrt(252.0)) if len(active) > 1 else np.nan
        rows.append(
            {
                "comparison": f"{v3_name} minus {baseline}",
                "annualized_active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": active_return / tracking_error if tracking_error and tracking_error > 0 else np.nan,
                "annualized_return_difference": perf.loc[v3_name, "annualized_return"] - perf.loc[baseline, "annualized_return"],
                "sharpe_difference": perf.loc[v3_name, "sharpe"] - perf.loc[baseline, "sharpe"],
                "max_drawdown_improvement": perf.loc[v3_name, "max_drawdown"] - perf.loc[baseline, "max_drawdown"],
                "final_wealth_ratio": perf.loc[v3_name, "final_wealth"] / perf.loc[baseline, "final_wealth"],
                "avg_rebalance_turnover_difference": (
                    perf.loc[v3_name, "avg_rebalance_turnover"] - perf.loc[baseline, "avg_rebalance_turnover"]
                ),
                "total_cost_difference": perf.loc[v3_name, "total_cost"] - perf.loc[baseline, "total_cost"],
            }
        )
    return pd.DataFrame(rows)


def combine_with_references(
    strategy: BacktestResult,
    ref_returns: pd.DataFrame,
    ref_costs: pd.DataFrame,
    ref_turnover: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref_cols = [col for col in REFERENCE_STRATEGIES if col in ref_returns.columns]
    idx = ref_returns.loc[(ref_returns.index >= start_date) & (ref_returns.index <= end_date)].index
    idx = idx.intersection(strategy.returns.index)
    combined_returns = ref_returns.loc[idx, ref_cols].copy()
    combined_returns[strategy.name] = strategy.returns.reindex(idx)
    combined_costs = ref_costs.loc[idx, ref_cols].copy()
    combined_costs[strategy.name] = strategy.costs.reindex(idx).fillna(0.0)
    combined_turnover = ref_turnover.loc[idx, ref_cols].copy()
    combined_turnover[strategy.name] = strategy.turnover.reindex(idx).fillna(0.0)
    return combined_returns, combined_turnover, combined_costs


def wealth_frame(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).cumprod()


def drawdown_frame(returns: pd.DataFrame) -> pd.DataFrame:
    wealth = wealth_frame(returns)
    return wealth / wealth.cummax() - 1.0


def plot_cumulative_wealth(returns: pd.DataFrame, path: Path, title: str, highlight: str) -> None:
    wealth = wealth_frame(returns)
    plt.figure(figsize=(12, 6))
    for col in wealth.columns:
        plt.plot(wealth.index, wealth[col], linewidth=2.2 if col == highlight else 1.3, label=col)
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_drawdowns(returns: pd.DataFrame, path: Path, title: str, highlight: str) -> None:
    dd = drawdown_frame(returns)
    plt.figure(figsize=(12, 5))
    for col in dd.columns:
        plt.plot(dd.index, dd[col], linewidth=2.0 if col == highlight else 1.2, label=col)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_excess_wealth(returns: pd.DataFrame, path: Path, title: str, strategy_name: str) -> pd.DataFrame:
    wealth = wealth_frame(returns)
    excess = pd.DataFrame(index=wealth.index)
    for baseline in REFERENCE_STRATEGIES:
        if baseline in wealth.columns:
            excess[f"vs_{baseline}"] = wealth[strategy_name] / wealth[baseline] - 1.0
    plt.figure(figsize=(12, 5))
    for col in excess.columns:
        plt.plot(excess.index, excess[col], linewidth=1.7, label=col.replace("vs_", "V3 / "))
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.title(title)
    plt.ylabel("Relative wealth minus 1")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
    return excess


def plot_rolling_active_return(returns: pd.DataFrame, path: Path, strategy_name: str) -> None:
    active = pd.DataFrame(index=returns.index)
    for baseline in REFERENCE_STRATEGIES:
        if baseline in returns.columns:
            active[f"minus {baseline}"] = returns[strategy_name] - returns[baseline]
    rolling = active.rolling(252).mean() * 252.0
    plt.figure(figsize=(12, 5))
    for col in rolling.columns:
        plt.plot(rolling.index, rolling[col], linewidth=1.5, label=col)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.title("V3 Rolling 1Y Active Return")
    plt.ylabel("Annualized active return")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_weights(weights: pd.DataFrame, path: Path) -> None:
    monthly = weights[ASSETS].resample("M").last().dropna(how="all")
    plt.figure(figsize=(12, 6))
    plt.stackplot(monthly.index, monthly[ASSETS].T, labels=ASSETS, alpha=0.92)
    plt.title("V3 Monthly Portfolio Weights")
    plt.ylabel("Weight")
    plt.ylim(0, 1)
    plt.grid(alpha=0.20)
    plt.legend(ncol=3, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_selection_counts(selection: pd.DataFrame, path: Path) -> None:
    counts = selection["selected_candidate"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(12, 6))
    plt.barh(counts.index, counts.values, color="#4c78a8", alpha=0.85)
    plt.title("V3 Selected Candidate Counts")
    plt.xlabel("Rebalance count")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_selected_alpha(selection: pd.DataFrame, path: Path) -> None:
    frame = selection.copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    frame = frame.set_index("effective_date")
    plt.figure(figsize=(12, 5))
    plt.plot(frame.index, frame["base_alpha"], label="base_alpha", linewidth=1.5)
    plt.plot(frame.index, frame["effective_alpha"], label="effective_alpha", linewidth=1.5)
    plt.title("V3 Rolling Alpha Strength")
    plt.ylabel("Alpha shrink")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_yearly_active(yearly: pd.DataFrame, path: Path) -> None:
    columns = [c for c in yearly.columns if c.startswith("active_vs_")]
    plt.figure(figsize=(12, 5))
    x = np.arange(len(yearly))
    width = 0.25
    for i, col in enumerate(columns):
        plt.bar(x + (i - 1) * width, yearly[col], width=width, label=col.replace("active_vs_", "vs "))
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.xticks(x, yearly["year"].astype(str))
    plt.title("V3 Yearly Active Return")
    plt.ylabel("Annual active return")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def candidate_performance_table(
    candidate_returns: pd.DataFrame,
    candidate_turnover: pd.DataFrame,
    candidate_costs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for col in candidate_returns.columns:
        row = performance_metrics(candidate_returns[col], candidate_turnover[col], candidate_costs[col])
        row["candidate"] = col
        rows.append(row)
    cols = ["candidate", "start", "end", "observations", "annualized_return", "annualized_volatility", "sharpe", "max_drawdown", "final_wealth", "avg_rebalance_turnover", "total_cost"]
    return pd.DataFrame(rows)[cols].sort_values(["sharpe", "annualized_return"], ascending=False)


def yearly_active_returns(returns: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    rows = []
    for year, group in returns.groupby(returns.index.year):
        row = {"year": int(year), "v3_return": annualized_return(group[strategy_name])}
        for baseline in REFERENCE_STRATEGIES:
            if baseline in group.columns:
                row[f"active_vs_{baseline}"] = float((group[strategy_name] - group[baseline]).mean() * 252.0)
        rows.append(row)
    return pd.DataFrame(rows)


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


def write_readme(
    candidates: Sequence[CandidateParams],
    full_perf: pd.DataFrame,
    full_excess: pd.DataFrame,
    test_perf: pd.DataFrame,
    test_excess: pd.DataFrame,
    selection_summary: pd.DataFrame,
    candidate_perf: pd.DataFrame,
) -> None:
    v3_row = full_perf.set_index("strategy").loc["V3_Rolling_Regime_MPC"]
    readme = f"""# V3 Rolling Regime-MPC Experiment

## 改进目标

V3 是对 V2 的直接修正：不再用 `2013-2018` 固定验证集选出一组永久参数，而是从 `2013-01-08` 开始完整滚动运行。每个调仓点只用过去已经实现的候选策略收益来选择当前参数，从而避免把某一段历史的市场风格锁死到未来。

## 核心变化

- 起始时间仍为 `2013-01-08`，与主实证和 baseline 口径一致。
- 参数网格保持小规模，共 `{len(candidates)}` 个候选，避免滚动选择过慢。
- 每个候选都是一条实时 paper-trading 策略，候选本身只使用当时可见的历史收益和 HMM regime probability。
- V3 每次调仓使用过去 `{SELECTION_WINDOW}` 个交易日的候选表现打分；早期不足 `{MIN_SELECTION_OBS}` 个交易日时，使用强 momentum 默认候选。
- 打分函数不仅看 Sharpe，也加入相对等权的 active return 和 information ratio，避免重新选出过度防御的参数。
- alpha 不再是固定全局 shrink，而是 regime-conditioned：calm 时保留更多 12-1 momentum，stress 时自动降低 alpha 强度。

## 候选参数

| 参数 | 候选值 |
|---|---|
| `gamma` | `5.0, 8.0` |
| `base_alpha` | `0.25, 0.75, 1.0` |
| `turnover_cap` | `0.10, 0.25` |
| `stress_multiplier` | `1.0` |
| MPC horizon | `{MPC_HORIZON}` |
| 单资产权重上限 | `{MAX_WEIGHT:.0%}` |

## 无未来数据泄漏控制

- 候选策略收益是按时间顺序逐日生成的 paper-trading 结果。
- 在调仓日 `t`，V3 只使用 `t` 之前已经实现的候选收益打分。
- 调仓权重只使用 `t` 及以前的收益历史和 HMM 下一期概率，并在下一交易日生效。
- 所有策略都从 2013 年开始运行；测试段只是对 2019-2024 收益进行切片评估。

## 全样本结果

{formatted_table(full_perf)}

全样本超额表现：

{formatted_table(full_excess)}

## 2019-2024 测试切片

{formatted_table(test_perf)}

测试切片超额表现：

{formatted_table(test_excess)}

## 参数选择诊断

V3 全样本年化收益为 `{v3_row['annualized_return']:.2%}`，Sharpe 为 `{v3_row['sharpe']:.4f}`，最大回撤为 `{v3_row['max_drawdown']:.2%}`。相较 V2，V3 的关键改进是允许 strong momentum 候选重新进入模型，同时让参数随近期历史表现滚动变化。

选择次数最多的候选：

{formatted_table(selection_summary.head(10))}

候选自身全样本 paper-trading 表现：

{formatted_table(candidate_perf.head(10))}

## 主要输出

- `results/tables/full_sample_performance.csv`
- `results/tables/full_sample_excess_vs_references.csv`
- `results/tables/test_performance.csv`
- `results/tables/test_excess_vs_references.csv`
- `results/tables/v3_daily_series_full.csv`
- `results/tables/v3_daily_weights_full.csv`
- `results/tables/selected_params_over_time.csv`
- `results/tables/regime_to_params_summary.csv`
- `results/tables/average_weights_by_regime.csv`
- `results/tables/yearly_active_returns.csv`
- `results/figures/full_cumulative_wealth.png`
- `results/figures/full_excess_wealth_curves.png`
- `results/figures/test_cumulative_wealth.png`
- `results/figures/test_excess_wealth_curves.png`
- `results/figures/v3_monthly_weights.png`
- `results/figures/v3_selection_counts.png`
- `results/figures/v3_alpha_strength.png`
- `results/figures/v3_yearly_active_returns.png`
"""
    (HERE / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    returns = load_returns()
    regime_records = load_regime_records()
    ref_returns, ref_costs, ref_turnover = load_reference_series()
    candidates = make_candidates()

    candidate_returns = {}
    candidate_turnover = {}
    candidate_costs = {}
    candidate_rebalance_records = []

    for i, params in enumerate(candidates, start=1):
        print(f"Running candidate {i:02d}/{len(candidates)}: {params.name}", flush=True)
        result = run_fixed_param_backtest(returns, regime_records, params, params.name)
        candidate_returns[params.name] = result.returns
        candidate_turnover[params.name] = result.turnover
        candidate_costs[params.name] = result.costs
        if not result.rebalance_records.empty:
            candidate_rebalance_records.append(result.rebalance_records)

    candidate_returns_df = pd.DataFrame(candidate_returns).sort_index()
    candidate_turnover_df = pd.DataFrame(candidate_turnover).sort_index()
    candidate_costs_df = pd.DataFrame(candidate_costs).sort_index()
    candidate_returns_df.to_csv(TABLES_DIR / "candidate_daily_returns.csv")
    candidate_turnover_df.to_csv(TABLES_DIR / "candidate_daily_turnover.csv")
    candidate_costs_df.to_csv(TABLES_DIR / "candidate_daily_costs.csv")
    if candidate_rebalance_records:
        pd.concat(candidate_rebalance_records, ignore_index=True).to_csv(TABLES_DIR / "candidate_rebalance_records.csv", index=False)

    equal_weight = ref_returns["EqualWeight_weekly"].reindex(candidate_returns_df.index)
    print("Running V3 rolling selector", flush=True)
    v3 = run_v3_meta_strategy(
        returns=returns,
        regime_records=regime_records,
        candidates=candidates,
        candidate_returns=candidate_returns_df,
        equal_weight_returns=equal_weight,
    )

    full_returns, full_turnover, full_costs = combine_with_references(
        v3, ref_returns, ref_costs, ref_turnover, START_DATE, END_DATE
    )
    test_returns, test_turnover, test_costs = combine_with_references(
        v3, ref_returns, ref_costs, ref_turnover, TEST_START, END_DATE
    )

    strategy_order = [*REFERENCE_STRATEGIES, v3.name]
    full_perf = metrics_table(full_returns, full_turnover, full_costs, strategy_order)
    test_perf = metrics_table(test_returns, test_turnover, test_costs, strategy_order)
    full_excess = summarize_excess(full_perf, full_returns, v3.name)
    test_excess = summarize_excess(test_perf, test_returns, v3.name)
    candidate_perf = candidate_performance_table(candidate_returns_df, candidate_turnover_df, candidate_costs_df)

    full_perf.to_csv(TABLES_DIR / "full_sample_performance.csv", index=False)
    test_perf.to_csv(TABLES_DIR / "test_performance.csv", index=False)
    full_excess.to_csv(TABLES_DIR / "full_sample_excess_vs_references.csv", index=False)
    test_excess.to_csv(TABLES_DIR / "test_excess_vs_references.csv", index=False)
    candidate_perf.to_csv(TABLES_DIR / "candidate_performance_full.csv", index=False)
    full_returns.to_csv(TABLES_DIR / "full_strategy_daily_returns.csv")
    test_returns.to_csv(TABLES_DIR / "test_strategy_daily_returns.csv")

    v3_series = pd.DataFrame(
        {
            "net_return": v3.returns,
            "turnover": v3.turnover,
            "cost": v3.costs,
            "wealth": (1.0 + v3.returns).cumprod(),
        }
    )
    v3_series.to_csv(TABLES_DIR / "v3_daily_series_full.csv")
    v3.daily_weights.to_csv(TABLES_DIR / "v3_daily_weights_full.csv")
    v3.rebalance_records.to_csv(TABLES_DIR / "selected_params_over_time.csv", index=False)
    v3.forecast_records.to_csv(TABLES_DIR / "v3_forecast_records.csv", index=False)

    selection = v3.rebalance_records.copy()
    selection_summary = (
        selection.groupby(["selected_candidate", "gamma", "base_alpha", "turnover_cap"], dropna=False)
        .size()
        .reset_index(name="selection_count")
        .sort_values("selection_count", ascending=False)
    )
    selection_summary.to_csv(TABLES_DIR / "selection_summary.csv", index=False)

    regime_to_params = (
        selection.groupby(["regime", "selected_candidate"], dropna=False)
        .size()
        .reset_index(name="selection_count")
        .sort_values(["regime", "selection_count"], ascending=[True, False])
    )
    regime_to_params.to_csv(TABLES_DIR / "regime_to_params_summary.csv", index=False)

    average_weights_by_regime = selection.groupby("regime")[ASSETS].mean().reset_index()
    average_weights_by_regime.to_csv(TABLES_DIR / "average_weights_by_regime.csv", index=False)

    yearly = yearly_active_returns(full_returns, v3.name)
    yearly.to_csv(TABLES_DIR / "yearly_active_returns.csv", index=False)

    full_excess_curves = plot_excess_wealth(
        full_returns, FIGURES_DIR / "full_excess_wealth_curves.png", "V3 Relative Wealth vs References, Full Sample", v3.name
    )
    test_excess_curves = plot_excess_wealth(
        test_returns, FIGURES_DIR / "test_excess_wealth_curves.png", "V3 Relative Wealth vs References, 2019-2024", v3.name
    )
    full_excess_curves.to_csv(TABLES_DIR / "full_excess_wealth_curves.csv")
    test_excess_curves.to_csv(TABLES_DIR / "test_excess_wealth_curves.csv")

    plot_cumulative_wealth(full_returns, FIGURES_DIR / "full_cumulative_wealth.png", "Strategy Cumulative Wealth, Full Sample", v3.name)
    plot_cumulative_wealth(test_returns, FIGURES_DIR / "test_cumulative_wealth.png", "Strategy Cumulative Wealth, 2019-2024", v3.name)
    plot_drawdowns(full_returns, FIGURES_DIR / "full_drawdowns.png", "Strategy Drawdowns, Full Sample", v3.name)
    plot_drawdowns(test_returns, FIGURES_DIR / "test_drawdowns.png", "Strategy Drawdowns, 2019-2024", v3.name)
    plot_rolling_active_return(full_returns, FIGURES_DIR / "rolling_1y_active_return_full.png", v3.name)
    plot_weights(v3.daily_weights, FIGURES_DIR / "v3_monthly_weights.png")
    plot_selection_counts(selection, FIGURES_DIR / "v3_selection_counts.png")
    plot_selected_alpha(selection, FIGURES_DIR / "v3_alpha_strength.png")
    plot_yearly_active(yearly, FIGURES_DIR / "v3_yearly_active_returns.png")

    manifest = {
        "model": "V3_Rolling_Regime_MPC",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "test_slice": [str(TEST_START.date()), str(END_DATE.date())],
        "candidate_count": len(candidates),
        "selection_window": SELECTION_WINDOW,
        "min_selection_observations": MIN_SELECTION_OBS,
        "candidates": [asdict(p) for p in candidates],
        "no_future_leakage_controls": [
            "candidate returns are generated sequentially",
            "selection at each rebalance uses candidate returns through the previous trading day",
            "portfolio optimization uses returns through the decision date and trades on the next trading day",
        ],
        "outputs": {
            "tables": sorted(str(p.relative_to(HERE)) for p in TABLES_DIR.glob("*.csv")),
            "figures": sorted(str(p.relative_to(HERE)) for p in FIGURES_DIR.glob("*.png")),
        },
    }
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_readme(
        candidates=candidates,
        full_perf=full_perf,
        full_excess=full_excess,
        test_perf=test_perf,
        test_excess=test_excess,
        selection_summary=selection_summary,
        candidate_perf=candidate_perf,
    )

    print("\nFull-sample performance:")
    print(full_perf.to_string(index=False))
    print("\nFull-sample excess:")
    print(full_excess.to_string(index=False))
    print("\nTest-slice performance:")
    print(test_perf.to_string(index=False))
    print("\nSelection summary:")
    print(selection_summary.head(10).to_string(index=False))
    print(f"\nWrote V3 outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
