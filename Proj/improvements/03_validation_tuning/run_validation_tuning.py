from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPROVEMENT_DIR = PROJECT_ROOT / "improvements" / "03_validation_tuning"
RESULTS_DIR = IMPROVEMENT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"


ASSET_ORDER = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


@dataclass(frozen=True)
class GridParams:
    gamma: float
    turnover_cap: float
    stress_multiplier: float


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_returns() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "returns.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"})
    df = df.set_index("date").sort_index()
    missing_cols = [c for c in ASSET_ORDER if c not in df.columns]
    if missing_cols:
        raise ValueError(f"returns.csv missing columns: {missing_cols}")
    return df[ASSET_ORDER].astype(float)


def load_regime_records() -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / "outputs" / "tables" / "hmm_regime_records.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


def load_regime_features() -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / "data" / "processed" / "regime_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"}).set_index("date").sort_index()
    return df


def nearest_psd(matrix: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def compute_rolling_moments(window: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu = window.mean(axis=0).to_numpy(dtype=float)
    cov = window.cov().to_numpy(dtype=float)
    diag = np.diag(np.diag(cov))
    shrunk = 0.85 * cov + 0.15 * diag
    shrunk = nearest_psd(shrunk)
    return mu, shrunk


def extract_stress_probability(
    date: pd.Timestamp,
    regime_records: Optional[pd.DataFrame],
    regime_features: Optional[pd.DataFrame],
) -> float:
    if regime_records is not None and date in regime_records.index:
        row = regime_records.loc[date]
        if "next_prob_stress" in row:
            return float(np.clip(row["next_prob_stress"], 0.0, 1.0))
        if "prob_stress" in row:
            return float(np.clip(row["prob_stress"], 0.0, 1.0))

    if regime_features is None or date not in regime_features.index:
        return 0.0

    row = regime_features.loc[date]
    proxy = 0.55 * float(row.get("vix_log", 0.0))
    proxy += 0.35 * float(row.get("hy_spread", 0.0)) / 10.0
    proxy += 0.10 * float(row.get("drawdown_63", 0.0)) * -10.0
    return float(1.0 / (1.0 + np.exp(-0.7 * proxy)))


def build_rebalance_dates(
    returns: pd.DataFrame,
    regime_records: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[pd.Timestamp]:
    if regime_records is not None:
        dates = [d for d in regime_records.index if start <= d <= end and d in returns.index]
        if dates:
            return dates
    all_dates = [d for d in returns.index if start <= d <= end]
    return all_dates[::5]


def solve_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    prev_w: np.ndarray,
    gamma: float,
    turnover_cap: float,
    stress_prob: float,
    stress_multiplier: float,
    base_cost_bps: float,
) -> Tuple[np.ndarray, float, str]:
    n = len(mu)
    w = cp.Variable(n)
    turnover = 0.5 * cp.norm1(w - prev_w)
    cost_coeff = (base_cost_bps / 10000.0) * (1.0 + stress_multiplier * stress_prob)

    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cov) - cost_coeff * turnover)
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        turnover <= turnover_cap,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)

    status = problem.status or "unknown"
    if w.value is None or status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return prev_w.copy(), 0.0, f"fallback:{status}"

    weights = np.asarray(w.value, dtype=float).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    realized_turnover = 0.5 * float(np.abs(weights - prev_w).sum())
    return weights, realized_turnover, status


def compute_segment_returns(
    returns: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, np.ndarray],
    turnover_by_date: Dict[pd.Timestamp, float],
    stress_prob_by_date: Dict[pd.Timestamp, float],
    gamma: float,
    turnover_cap: float,
    stress_multiplier: float,
    base_cost_bps: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    net_ret: List[float] = []
    turnover_series: List[float] = []
    cost_series: List[float] = []
    dates: List[pd.Timestamp] = []
    rebalance_dates = sorted(weights_by_date.keys())

    for i, reb_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1] + pd.Timedelta(days=1)
        seg = returns.loc[(returns.index >= reb_date) & (returns.index < next_date)]
        w = weights_by_date[reb_date]
        turnover = turnover_by_date[reb_date]
        stress_prob = stress_prob_by_date[reb_date]
        cost_coeff = (base_cost_bps / 10000.0) * (1.0 + stress_multiplier * stress_prob)
        cost = cost_coeff * turnover

        seg_values = seg.to_numpy(dtype=float) @ w
        if len(seg_values) > 0:
            seg_values = seg_values.copy()
            seg_values[0] -= cost
        dates.extend(seg.index.tolist())
        net_ret.extend(seg_values.tolist())
        if len(seg_values) > 0:
            turnover_series.extend([turnover] + [0.0] * max(len(seg_values) - 1, 0))
            cost_series.extend([cost] + [0.0] * max(len(seg_values) - 1, 0))

    ret = pd.Series(net_ret, index=dates, name="net_return").sort_index()
    turnover = pd.Series(turnover_series, index=dates, name="turnover").sort_index()
    cost = pd.Series(cost_series, index=dates, name="cost").sort_index()
    return ret, turnover, cost


def performance_metrics(ret: pd.Series, turnover: pd.Series, cost: pd.Series) -> Dict[str, float]:
    if ret.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "final_wealth": np.nan,
            "avg_turnover": np.nan,
            "total_cost": np.nan,
        }

    wealth = (1.0 + ret).cumprod()
    ann_return = wealth.iloc[-1] ** (252.0 / len(ret)) - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(252.0)
    sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol
    roll_max = wealth.cummax()
    drawdown = wealth / roll_max - 1.0
    max_drawdown = float(drawdown.min())
    avg_turnover = float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0
    total_cost = float(cost.sum())
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "final_wealth": float(wealth.iloc[-1]),
        "avg_turnover": avg_turnover,
        "total_cost": total_cost,
    }


def run_policy(
    returns: pd.DataFrame,
    regime_records: Optional[pd.DataFrame],
    regime_features: Optional[pd.DataFrame],
    rebalance_dates: Sequence[pd.Timestamp],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    params: GridParams,
    lookback: int = 252,
    base_cost_bps: float = 5.0,
) -> Dict[str, object]:
    weights_by_date: Dict[pd.Timestamp, np.ndarray] = {}
    turnover_by_date: Dict[pd.Timestamp, float] = {}
    stress_prob_by_date: Dict[pd.Timestamp, float] = {}
    status_by_date: Dict[pd.Timestamp, str] = {}

    prev_w = np.ones(len(ASSET_ORDER), dtype=float) / len(ASSET_ORDER)
    valid_rebalance_dates = [d for d in rebalance_dates if start_date <= d <= end_date]

    for reb_date in valid_rebalance_dates:
        history = returns.loc[returns.index < reb_date].tail(lookback)
        if len(history) < max(60, min(lookback, 126)):
            continue

        mu, cov = compute_rolling_moments(history)
        stress_prob = extract_stress_probability(reb_date, regime_records, regime_features)
        weights, turnover, status = solve_weights(
            mu=mu,
            cov=cov,
            prev_w=prev_w,
            gamma=params.gamma,
            turnover_cap=params.turnover_cap,
            stress_prob=stress_prob,
            stress_multiplier=params.stress_multiplier,
            base_cost_bps=base_cost_bps,
        )
        weights_by_date[reb_date] = weights
        turnover_by_date[reb_date] = turnover
        stress_prob_by_date[reb_date] = stress_prob
        status_by_date[reb_date] = status
        prev_w = weights

    ret, turnover_series, cost_series = compute_segment_returns(
        returns=returns,
        weights_by_date=weights_by_date,
        turnover_by_date=turnover_by_date,
        stress_prob_by_date=stress_prob_by_date,
        gamma=params.gamma,
        turnover_cap=params.turnover_cap,
        stress_multiplier=params.stress_multiplier,
        base_cost_bps=base_cost_bps,
    )
    metrics = performance_metrics(ret, turnover_series, cost_series)
    metrics.update(
        {
            "gamma": params.gamma,
            "turnover_cap": params.turnover_cap,
            "stress_multiplier": params.stress_multiplier,
            "rebalance_count": len(weights_by_date),
            "failed_rebalance_count": sum(1 for s in status_by_date.values() if str(s).startswith("fallback")),
            "avg_stress_probability": float(np.mean(list(stress_prob_by_date.values()))) if stress_prob_by_date else np.nan,
        }
    )
    return {
        "metrics": metrics,
        "returns": ret,
        "turnover": turnover_series,
        "cost": cost_series,
        "weights_by_date": weights_by_date,
        "turnover_by_date": turnover_by_date,
        "stress_prob_by_date": stress_prob_by_date,
        "status_by_date": status_by_date,
    }


def make_grid() -> List[GridParams]:
    gammas = [1.0, 3.0, 10.0]
    turnover_caps = [0.05, 0.10, 0.20]
    stress_multipliers = [0.0, 1.0, 3.0]
    grid = []
    for gamma in gammas:
        for cap in turnover_caps:
            for stress in stress_multipliers:
                grid.append(GridParams(gamma=gamma, turnover_cap=cap, stress_multiplier=stress))
    return grid


def load_reference_strategy_metrics(
    returns: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "tables" / "strategy_daily_returns.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    cost_path = PROJECT_ROOT / "outputs" / "tables" / "strategy_daily_costs.csv"
    turnover_path = PROJECT_ROOT / "outputs" / "tables" / "strategy_daily_turnover.csv"
    costs = pd.read_csv(cost_path, index_col=0, parse_dates=True).sort_index().loc[df.index] if cost_path.exists() else None
    turnover = pd.read_csv(turnover_path, index_col=0, parse_dates=True).sort_index().loc[df.index] if turnover_path.exists() else None

    rows = []
    for col in df.columns:
        ret = df[col].dropna()
        c = costs[col].reindex(ret.index).fillna(0.0) if costs is not None and col in costs.columns else pd.Series(0.0, index=ret.index)
        t = turnover[col].reindex(ret.index).fillna(0.0) if turnover is not None and col in turnover.columns else pd.Series(0.0, index=ret.index)
        metrics = performance_metrics(ret, t, c)
        metrics["strategy"] = col
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_excess(
    tuned_metrics: Dict[str, float],
    reference_metrics: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    ref = reference_metrics.set_index("strategy").loc[reference_name]
    data = {
        "comparison": [f"tuned minus {reference_name}"],
        "ann_return_diff": [tuned_metrics["ann_return"] - ref["ann_return"]],
        "ann_vol_diff": [tuned_metrics["ann_vol"] - ref["ann_vol"]],
        "sharpe_diff": [tuned_metrics["sharpe"] - ref["sharpe"]],
        "max_drawdown_diff": [tuned_metrics["max_drawdown"] - ref["max_drawdown"]],
        "final_wealth_ratio": [tuned_metrics["final_wealth"] / ref["final_wealth"]],
        "avg_turnover_diff": [tuned_metrics["avg_turnover"] - ref["avg_turnover"]],
        "total_cost_diff": [tuned_metrics["total_cost"] - ref["total_cost"]],
    }
    return pd.DataFrame(data)


def main() -> None:
    ensure_dirs()
    returns = load_returns()
    regime_records = load_regime_records()
    regime_features = load_regime_features()

    validation_start = pd.Timestamp("2013-01-11")
    validation_end = pd.Timestamp("2018-12-31")
    test_start = pd.Timestamp("2019-01-01")
    test_end = pd.Timestamp("2024-12-31")

    rebalance_dates = build_rebalance_dates(returns, regime_records, validation_start, test_end)

    grid = make_grid()
    grid_rows = []
    for params in grid:
        out = run_policy(
            returns=returns,
            regime_records=regime_records,
            regime_features=regime_features,
            rebalance_dates=rebalance_dates,
            start_date=validation_start,
            end_date=validation_end,
            params=params,
        )
        row = dict(out["metrics"])
        grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows).sort_values(
        by=["sharpe", "ann_return", "final_wealth"],
        ascending=[False, False, False],
    )
    grid_path = TABLES_DIR / "validation_grid_results.csv"
    grid_df.to_csv(grid_path, index=False)

    selected = grid_df.iloc[0].to_dict()
    selected_params = GridParams(
        gamma=float(selected["gamma"]),
        turnover_cap=float(selected["turnover_cap"]),
        stress_multiplier=float(selected["stress_multiplier"]),
    )
    selected_path = TABLES_DIR / "selected_parameters.csv"
    pd.DataFrame([selected]).to_csv(selected_path, index=False)

    validation_out = run_policy(
        returns=returns,
        regime_records=regime_records,
        regime_features=regime_features,
        rebalance_dates=rebalance_dates,
        start_date=validation_start,
        end_date=validation_end,
        params=selected_params,
    )
    test_out = run_policy(
        returns=returns,
        regime_records=regime_records,
        regime_features=regime_features,
        rebalance_dates=rebalance_dates,
        start_date=test_start,
        end_date=test_end,
        params=selected_params,
    )

    val_metrics = pd.DataFrame([validation_out["metrics"]])
    test_metrics = pd.DataFrame([test_out["metrics"]])
    val_metrics.to_csv(TABLES_DIR / "validation_selected_metrics.csv", index=False)
    test_metrics.to_csv(TABLES_DIR / "test_selected_metrics.csv", index=False)

    tuned_daily = pd.DataFrame(
        {
            "net_return": test_out["returns"],
            "turnover": test_out["turnover"],
            "cost": test_out["cost"],
        }
    )
    tuned_daily.to_csv(TABLES_DIR / "tuned_daily_series.csv")

    reference_metrics = load_reference_strategy_metrics(returns, test_start, test_end)
    reference_metrics.to_csv(TABLES_DIR / "reference_strategy_metrics_test.csv", index=False)

    excess_rows = []
    if not reference_metrics.empty and "HMM_MPC_CVXPY_weekly" in set(reference_metrics["strategy"]):
        excess_rows.append(summarize_excess(test_out["metrics"], reference_metrics, "HMM_MPC_CVXPY_weekly"))
    if not reference_metrics.empty and "EqualWeight_weekly" in set(reference_metrics["strategy"]):
        excess_rows.append(summarize_excess(test_out["metrics"], reference_metrics, "EqualWeight_weekly"))
    if excess_rows:
        excess = pd.concat(excess_rows, ignore_index=True)
        excess.to_csv(TABLES_DIR / "tuned_excess_vs_references.csv", index=False)
    else:
        excess = pd.DataFrame()

    comparison = {
        "validation_period": f"{validation_start.date()} to {validation_end.date()}",
        "test_period": f"{test_start.date()} to {test_end.date()}",
        "selected_gamma": selected_params.gamma,
        "selected_turnover_cap": selected_params.turnover_cap,
        "selected_stress_multiplier": selected_params.stress_multiplier,
        "validation_sharpe": validation_out["metrics"]["sharpe"],
        "test_sharpe": test_out["metrics"]["sharpe"],
        "test_ann_return": test_out["metrics"]["ann_return"],
        "test_ann_vol": test_out["metrics"]["ann_vol"],
        "test_max_drawdown": test_out["metrics"]["max_drawdown"],
        "test_final_wealth": test_out["metrics"]["final_wealth"],
        "test_avg_turnover": test_out["metrics"]["avg_turnover"],
        "test_total_cost": test_out["metrics"]["total_cost"],
    }
    pd.DataFrame([comparison]).to_csv(TABLES_DIR / "summary_metrics.csv", index=False)

    if not excess.empty:
        excess_summary = excess.copy()
        excess_summary.to_csv(TABLES_DIR / "summary_excess_metrics.csv", index=False)

    report_lines = [
        "# Validation Tuning Experiment",
        "",
        f"- Validation period: {validation_start.date()} to {validation_end.date()}",
        f"- Test period: {test_start.date()} to {test_end.date()}",
        f"- Selected gamma: {selected_params.gamma}",
        f"- Selected turnover cap: {selected_params.turnover_cap}",
        f"- Selected stress multiplier: {selected_params.stress_multiplier}",
        f"- Validation Sharpe: {validation_out['metrics']['sharpe']:.4f}",
        f"- Test Sharpe: {test_out['metrics']['sharpe']:.4f}",
        f"- Test annual return: {test_out['metrics']['ann_return']:.4%}",
        f"- Test max drawdown: {test_out['metrics']['max_drawdown']:.4%}",
        f"- Test average turnover: {test_out['metrics']['avg_turnover']:.4%}",
    ]
    (IMPROVEMENT_DIR / "run_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "selected_parameters": selected_params.__dict__,
        "validation_period": [str(validation_start.date()), str(validation_end.date())],
        "test_period": [str(test_start.date()), str(test_end.date())],
        "outputs": [
            str(grid_path.relative_to(IMPROVEMENT_DIR)),
            str(selected_path.relative_to(IMPROVEMENT_DIR)),
            str((TABLES_DIR / "validation_selected_metrics.csv").relative_to(IMPROVEMENT_DIR)),
            str((TABLES_DIR / "test_selected_metrics.csv").relative_to(IMPROVEMENT_DIR)),
            str((TABLES_DIR / "reference_strategy_metrics_test.csv").relative_to(IMPROVEMENT_DIR)),
            str((TABLES_DIR / "summary_metrics.csv").relative_to(IMPROVEMENT_DIR)),
        ],
    }
    (IMPROVEMENT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
