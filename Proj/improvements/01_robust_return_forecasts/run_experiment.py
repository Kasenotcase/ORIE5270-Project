from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

RETURNS_PATH = ROOT / "data" / "processed" / "returns.csv"
REGIME_RECORDS_PATH = ROOT / "outputs" / "tables" / "hmm_regime_records.csv"
EXTERNAL_BASELINE_PATH = ROOT / "outputs" / "tables" / "strategy_performance.csv"

LOOKBACK = 252
SKIP_DAYS = 21
TRADING_COST = 0.0005
RISK_AVERSION = 5.0
SHRINKAGE = 0.15
SOLVER = "OSQP"


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    weights: pd.DataFrame
    forecast_quality: pd.DataFrame


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    returns = returns.apply(pd.to_numeric, errors="coerce")

    regime = pd.read_csv(REGIME_RECORDS_PATH, parse_dates=["date"]).set_index("date").sort_index()
    regime = regime.apply(pd.to_numeric, errors="ignore")
    return returns, regime


def winsorize_series(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


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


def build_covariance(history: pd.DataFrame, shrinkage: float = SHRINKAGE) -> np.ndarray:
    x = history.to_numpy(dtype=float)
    cov = np.cov(x, rowvar=False, ddof=1)
    cov = np.asarray(cov, dtype=float)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - shrinkage) * cov + shrinkage * diag
    shrunk = 0.5 * (shrunk + shrunk.T)
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    eigvals = np.clip(eigvals, 1e-8, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    psd = 0.5 * (psd + psd.T)
    return psd


def forecast_variants(history: pd.DataFrame) -> Dict[str, pd.Series]:
    reference = history.tail(LOOKBACK).mean()

    short = history.tail(LOOKBACK)
    if len(short) > SKIP_DAYS:
        momentum_window = short.iloc[:-SKIP_DAYS]
    else:
        momentum_window = short

    momentum_121 = np.log1p(momentum_window.clip(lower=-0.999)).mean()
    realized_vol = short.std(ddof=1).replace(0.0, np.nan).fillna(short.std(ddof=0).replace(0.0, 1e-6))
    vol_adj_momentum = momentum_121 / (realized_vol + 1e-8)

    momentum_scaled = rescale_to_reference(momentum_121, reference)
    vol_adj_scaled = rescale_to_reference(vol_adj_momentum, reference)

    blended = 0.60 * reference + 0.40 * vol_adj_scaled
    blended = rescale_to_reference(blended, reference)

    return {
        "raw_mean_252": reference,
        "momentum_12_1_scaled": momentum_scaled,
        "vol_adj_momentum_scaled": vol_adj_scaled,
        "shrink_blend": blended,
    }


def optimize_weights(
    mu: pd.Series,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    cost_mult: float,
    risk_aversion: float = RISK_AVERSION,
    trading_cost: float = TRADING_COST,
) -> Tuple[np.ndarray, str]:
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(
        mu.to_numpy(dtype=float) @ w
        - risk_aversion * cp.quad_form(w, sigma, assume_PSD=True)
        - trading_cost * cost_mult * 0.5 * cp.norm1(w - w_prev)
    )
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000)
    except Exception:
        problem.solve(solver=cp.SCS, warm_start=True, verbose=False, max_iters=10000)

    if w.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return w_prev.copy(), f"fallback_{problem.status}"

    out = np.asarray(w.value).reshape(-1)
    out = np.clip(out, 0.0, None)
    out = out / out.sum()
    return out, problem.status


def forward_average_daily_returns(returns: pd.DataFrame, start: pd.Timestamp, horizon: int = 5) -> pd.DataFrame:
    rows = []
    dates = returns.index
    for idx, date in enumerate(dates):
        if date < start:
            continue
        future = returns.iloc[idx + 1 : idx + 1 + horizon]
        if len(future) < horizon:
            continue
        avg_daily = np.log1p(future.clip(lower=-0.999)).mean()
        rows.append(pd.Series(avg_daily, name=date))
    if not rows:
        return pd.DataFrame(index=pd.Index([], name="date"), columns=returns.columns)
    return pd.DataFrame(rows)


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    return float(dd.min())


def summarize_performance(daily: pd.DataFrame) -> pd.Series:
    net = daily["net_return"].astype(float)
    wealth = daily["wealth"].astype(float)
    annualized_return = float(wealth.iloc[-1] ** (252.0 / max(len(net), 1)) - 1.0)
    annualized_vol = float(net.std(ddof=1) * np.sqrt(252.0))
    sharpe = float((net.mean() / net.std(ddof=1)) * np.sqrt(252.0)) if net.std(ddof=1) > 0 else np.nan
    turnover = daily["turnover"].astype(float)
    cost = daily["transaction_cost"].astype(float)
    return pd.Series(
        {
            "strategy": daily["strategy"].iloc[0],
            "start": str(daily["date"].iloc[0].date()),
            "end": str(daily["date"].iloc[-1].date()),
            "observations": int(len(daily)),
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(wealth),
            "final_wealth": float(wealth.iloc[-1]),
            "avg_daily_turnover": float(turnover.mean()),
            "avg_rebalance_turnover": float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0,
            "total_cost": float(cost.sum()),
        }
    )


def information_coefficient(forecast: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    ic_values = []
    hit_values = []
    top_minus_bottom = []

    for date in forecast.index.intersection(realized.index):
        f = forecast.loc[date]
        r = realized.loc[date]
        pair = pd.concat([f, r], axis=1).dropna()
        if len(pair) < 2:
            continue
        pair.columns = ["forecast", "realized"]
        ic_values.append(pair["forecast"].rank().corr(pair["realized"].rank()))
        hit_values.append(float(np.sign(pair["forecast"]).eq(np.sign(pair["realized"])).mean()))
        top = pair["forecast"].idxmax()
        bottom = pair["forecast"].idxmin()
        top_minus_bottom.append(float(pair.loc[top, "realized"] - pair.loc[bottom, "realized"]))

    return pd.Series(
        {
            "mean_spearman_ic": float(np.nanmean(ic_values)) if ic_values else np.nan,
            "median_spearman_ic": float(np.nanmedian(ic_values)) if ic_values else np.nan,
            "directional_hit_rate": float(np.nanmean(hit_values)) if hit_values else np.nan,
            "annualized_top_minus_bottom": float(np.nanmean(top_minus_bottom) * 252.0 / 5.0)
            if top_minus_bottom
            else np.nan,
            "observations": int(len(ic_values)),
        }
    )


def backtest_strategy(
    name: str,
    returns: pd.DataFrame,
    regime_records: pd.DataFrame,
    rebalance_start: pd.Timestamp,
    strategy_key: str,
) -> BacktestResult:
    assets = list(returns.columns)
    dates = returns.index
    rebalance_dates = [d for d in regime_records.index if d in returns.index and d >= rebalance_start]
    rebalance_dates = pd.Index(rebalance_dates)
    cost_lookup = regime_records["next_cost_multiplier"].to_dict()

    current_w = np.repeat(1.0 / len(assets), len(assets))
    daily_rows = []
    weight_rows = []
    forecast_rows = []
    realized_rows = []

    for date in dates[dates >= rebalance_start]:
        gross_ret = float(np.dot(current_w, returns.loc[date].to_numpy(dtype=float)))
        trade_cost = 0.0
        turnover = 0.0
        status = "hold"

        if date in rebalance_dates:
            history = returns.loc[:date].tail(LOOKBACK)
            if len(history) == LOOKBACK:
                variants = forecast_variants(history)
                mu = variants[strategy_key]
                sigma = build_covariance(history)
                cost_mult = float(cost_lookup.get(date, 1.0))
                next_w, status = optimize_weights(mu, sigma, current_w, cost_mult)
                turnover = 0.5 * float(np.abs(next_w - current_w).sum())
                trade_cost = TRADING_COST * cost_mult * turnover
                weight_rows.append(
                    pd.Series(
                        {
                            "date": date,
                            "strategy": name,
                            "cost_multiplier": cost_mult,
                            "turnover": turnover,
                            "solver_status": status,
                            **{asset: next_w[i] for i, asset in enumerate(assets)},
                        }
                    )
                )
                forecast_rows.append(
                    pd.Series(
                        {
                            "date": date,
                            "strategy": name,
                            **{asset: mu.iloc[i] for i, asset in enumerate(assets)},
                        }
                    )
                )
                current_w = next_w
            else:
                weight_rows.append(
                    pd.Series(
                        {
                            "date": date,
                            "strategy": name,
                            "cost_multiplier": np.nan,
                            "turnover": np.nan,
                            "solver_status": "insufficient_history",
                            **{asset: np.nan for asset in assets},
                        }
                    )
                )

        net_ret = gross_ret - trade_cost
        realized_rows.append(
            pd.Series(
                {
                    "date": date,
                    "strategy": name,
                    "gross_return": gross_ret,
                    "transaction_cost": trade_cost,
                    "net_return": net_ret,
                    "wealth": np.nan,
                    "turnover": turnover,
                    "solver_status": status,
                }
            )
        )
        daily_rows.append(realized_rows[-1])

    daily = pd.DataFrame(daily_rows)
    daily["wealth"] = (1.0 + daily["net_return"]).cumprod()
    weights = pd.DataFrame(weight_rows).dropna(how="all") if weight_rows else pd.DataFrame(columns=["date", "strategy"])
    forecasts = pd.DataFrame(forecast_rows).dropna(how="all") if forecast_rows else pd.DataFrame(columns=["date", "strategy"])

    realized_next_5d = forward_average_daily_returns(returns, rebalance_start, horizon=5)
    quality_rows = []
    if not forecasts.empty:
        forecast_series = forecasts.set_index("date").drop(columns=["strategy"])
        ic = information_coefficient(forecast_series, realized_next_5d)
        ic["strategy"] = name
        ic["forecast_variant"] = strategy_key
        quality_rows.append(ic)
    forecast_quality = pd.DataFrame(quality_rows)
    return BacktestResult(name=name, daily=daily, weights=weights, forecast_quality=forecast_quality)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    returns, regime = load_inputs()

    rebalance_start = pd.Timestamp("2013-01-08")
    strategy_map = {
        "raw_mean_252": "RawMean_HMMMPC",
        "momentum_12_1_scaled": "Momentum12_1Scaled_HMMMPC",
        "vol_adj_momentum_scaled": "VolAdjMomentum_HMMMPC",
        "shrink_blend": "ShrinkBlend_HMMMPC",
    }

    all_results = []
    all_weights = []
    all_quality = []

    for variant_key, strategy_name in strategy_map.items():
        result = backtest_strategy(strategy_name, returns, regime, rebalance_start, variant_key)
        all_results.append(summarize_performance(result.daily))
        if not result.weights.empty:
            all_weights.append(result.weights)
        if not result.forecast_quality.empty:
            all_quality.append(result.forecast_quality)

    performance = pd.DataFrame(all_results)
    performance = performance[
        [
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
    ]

    external = pd.read_csv(EXTERNAL_BASELINE_PATH)
    external = external[external["strategy"].isin(["EqualWeight_weekly", "Markowitz_CVXPY_weekly"])]
    external = external[
        [
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
    ]
    external["source"] = "external_baseline"
    performance["source"] = "robust_return_forecast"
    combined = pd.concat([external, performance], ignore_index=True, sort=False)
    combined.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    performance.to_csv(RESULTS_DIR / "forecast_strategy_performance.csv", index=False)
    combined.sort_values(["source", "strategy"]).to_csv(RESULTS_DIR / "combined_strategy_table.csv", index=False)

    if all_weights:
        pd.concat(all_weights, ignore_index=True).to_csv(RESULTS_DIR / "strategy_weights.csv", index=False)
    if all_quality:
        pd.concat(all_quality, ignore_index=True).to_csv(RESULTS_DIR / "forecast_quality.csv", index=False)

    # Excess performance versus the existing project baselines.
    baseline = external.set_index("strategy")
    excess_rows = []
    for _, row in performance.iterrows():
        for base_name in baseline.index:
            base = baseline.loc[base_name]
            excess_rows.append(
                {
                    "strategy": row["strategy"],
                    "baseline": base_name,
                    "annualized_return_diff": float(row["annualized_return"] - base["annualized_return"]),
                    "annualized_volatility_diff": float(row["annualized_volatility"] - base["annualized_volatility"]),
                    "sharpe_diff": float(row["sharpe"] - base["sharpe"]),
                    "max_drawdown_diff": float(row["max_drawdown"] - base["max_drawdown"]),
                    "final_wealth_ratio": float(row["final_wealth"] / base["final_wealth"]),
                    "avg_rebalance_turnover_diff": float(row["avg_rebalance_turnover"] - base["avg_rebalance_turnover"]),
                    "total_cost_diff": float(row["total_cost"] - base["total_cost"]),
                }
            )
    pd.DataFrame(excess_rows).to_csv(RESULTS_DIR / "excess_vs_external_baselines.csv", index=False)

    # Keep one simple text summary around for quick inspection.
    summary = performance.sort_values("sharpe", ascending=False)
    summary.to_csv(RESULTS_DIR / "ranked_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
