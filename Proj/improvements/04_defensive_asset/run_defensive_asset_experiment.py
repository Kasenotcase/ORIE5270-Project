from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path
from typing import Callable, Tuple

import cvxpy as cp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl_cache")

import numpy as np
import pandas as pd
import requests
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(__file__).resolve().parent
ROOT_DATA = ROOT / "data" / "processed"
ROOT_OUTPUTS = ROOT / "outputs" / "tables"

DATA_DIR = LOCAL / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = LOCAL / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

BASE_ASSETS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEFAULT_DEFENSIVE_TICKER = "SHY"
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
INITIAL_TRAIN_DAYS = 756
REBALANCE_FREQ = "W-FRI"
WEIGHT_CAP = 0.40
GAMMA = 20.0
TRANSACTION_COST_BPS = 5.0
COV_RIDGE = 1e-6
HMM_STATES = 3
HMM_MIN_FEATURE_DAYS = 504
HMM_RANDOM_STATE = 7
HMM_MAX_ITER = 80
MPC_HORIZON = 5
MPC_TURNOVER_CAP = 0.35
MPC_QUADRATIC_TRADE_PENALTY = 1e-3
REGIME_COST_MULTIPLIERS = np.array([1.0, 1.5, 3.0])
REGIME_LABELS = ["calm", "transition", "stress"]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, TABLE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * math.sqrt(252))


def sharpe(returns: pd.Series) -> float:
    vol = annualized_volatility(returns)
    return annualized_return(returns) / vol if vol and vol > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.dropna()).cumprod()
    if wealth.empty:
        return np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, COV_RIDGE)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def load_root_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adj_close = pd.read_csv(ROOT_DATA / "adj_close_clean.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(ROOT_DATA / "close_clean.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(ROOT_DATA / "volume_clean.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv(ROOT_DATA / "macro_aligned.csv", index_col=0, parse_dates=True)
    features = pd.read_csv(ROOT_DATA / "regime_features.csv", index_col=0, parse_dates=True)
    return adj_close, close, volume, macro, features


def fetch_yahoo_chart_series(
    ticker: str, start: str = START_DATE, end: str = END_DATE, refresh: bool = False
) -> pd.DataFrame:
    cache = RAW_DIR / f"{ticker.lower()}_yahoo_chart_ohlcv_raw.csv"
    if cache.exists() and not refresh:
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index)
        return raw.sort_index()

    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    error = payload.get("chart", {}).get("error")
    if error is not None:
        raise RuntimeError(f"Yahoo chart returned error for {ticker}: {error}")
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    index = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    raw = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Adj Close": adj,
            "Volume": quote["volume"],
        },
        index=index,
    ).sort_index()
    raw.index.name = "Date"
    raw.to_csv(cache)
    return raw


def load_defensive_asset(ticker: str = DEFAULT_DEFENSIVE_TICKER, refresh: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series]:
    raw = fetch_yahoo_chart_series(ticker, refresh=refresh)
    adj = raw["Adj Close"].rename(ticker)
    close = raw["Close"].rename(ticker)
    volume = raw["Volume"].rename(ticker)
    return adj, close, volume


def build_extended_universe(
    base_adj_close: pd.DataFrame,
    base_close: pd.DataFrame,
    base_volume: pd.DataFrame,
    defensive_ticker: str,
    defensive_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def_adj, def_close, def_volume = load_defensive_asset(defensive_ticker, refresh=defensive_refresh)
    combined_index = base_adj_close.index.intersection(def_adj.index)
    adj_close = base_adj_close.loc[combined_index].copy()
    close = base_close.loc[combined_index].copy()
    volume = base_volume.loc[combined_index].copy()
    adj_close[defensive_ticker] = def_adj.reindex(combined_index)
    close[defensive_ticker] = def_close.reindex(combined_index)
    volume[defensive_ticker] = def_volume.reindex(combined_index)

    valid = adj_close.notna().all(axis=1) & close.notna().all(axis=1) & volume.notna().all(axis=1)
    valid &= (adj_close > 0).all(axis=1) & (close > 0).all(axis=1) & (volume >= 0).all(axis=1)
    adj_close = adj_close.loc[valid]
    close = close.loc[valid]
    volume = volume.loc[valid]
    returns = adj_close.pct_change().dropna(how="any")

    adj_close.loc[returns.index].to_csv(PROCESSED_DIR / "adj_close_extended.csv")
    close.loc[returns.index].to_csv(PROCESSED_DIR / "close_extended.csv")
    volume.loc[returns.index].to_csv(PROCESSED_DIR / "volume_extended.csv")
    returns.to_csv(PROCESSED_DIR / "returns_extended.csv")
    return adj_close.loc[returns.index], close.loc[returns.index], volume.loc[returns.index], returns


def build_regime_features(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    market_price = adj_close[BASE_ASSETS].mean(axis=1)
    market_return = market_price.pct_change()
    log_market = np.log(market_price)

    features = pd.DataFrame(index=adj_close.index)
    features["mom_21"] = log_market.diff(21)
    features["mom_63"] = log_market.diff(63)
    features["drawdown_63"] = market_price / market_price.rolling(63).max() - 1.0
    features["rv_21"] = market_return.rolling(21).std() * math.sqrt(252)
    features["rv_63"] = market_return.rolling(63).std() * math.sqrt(252)

    dollar_volume = close[BASE_ASSETS] * volume[BASE_ASSETS]
    illiq = market_return.abs() / dollar_volume.replace(0, np.nan).mean(axis=1)
    features["amihud_log"] = np.log1p(illiq)
    features["vix_log"] = np.log(macro["VIXCLS"])
    features["vix_change_21"] = np.log(macro["VIXCLS"]).diff(21)
    features["hy_spread"] = macro["BAMLH0A0HYM2"]
    features["hy_spread_change_21"] = macro["BAMLH0A0HYM2"].diff(21)

    features = features.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    features.to_csv(PROCESSED_DIR / "regime_features_extended.csv")
    return features


def data_quality_report(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    rolling_ends = returns.index[INITIAL_TRAIN_DAYS::63]
    cond_numbers = []
    for end_date in rolling_ends:
        sample = returns.loc[:end_date].tail(INITIAL_TRAIN_DAYS)
        if len(sample) < INITIAL_TRAIN_DAYS:
            continue
        cov = LedoitWolf().fit(sample.values).covariance_ + COV_RIDGE * np.eye(sample.shape[1])
        cond_numbers.append(float(np.linalg.cond(cov)))

    checks = [
        ("asset_count", len(returns.columns)),
        ("defensive_asset", returns.columns[-1]),
        ("raw_price_start", adj_close.index.min().date().isoformat()),
        ("raw_price_end", adj_close.index.max().date().isoformat()),
        ("return_start", returns.index.min().date().isoformat()),
        ("return_end", returns.index.max().date().isoformat()),
        ("return_observations", len(returns)),
        ("adj_close_missing_cells", int(adj_close.isna().sum().sum())),
        ("close_missing_cells", int(close.isna().sum().sum())),
        ("volume_missing_cells", int(volume.isna().sum().sum())),
        ("macro_missing_after_alignment", int(macro.reindex(returns.index).ffill().isna().sum().sum())),
        ("feature_observations", len(features)),
        ("rolling_cov_condition_median", float(np.median(cond_numbers)) if cond_numbers else np.nan),
        ("rolling_cov_condition_max", float(np.max(cond_numbers)) if cond_numbers else np.nan),
    ]
    table = pd.DataFrame(checks, columns=["check", "value"])
    table.to_csv(TABLE_DIR / "data_quality_summary.csv", index=False)
    return table


def asset_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        vol = annualized_volatility(r)
        rows.append(
            {
                "asset": ticker,
                "annualized_return": annualized_return(r),
                "annualized_volatility": vol,
                "sharpe": annualized_return(r) / vol if vol and vol > 0 else np.nan,
                "max_drawdown": max_drawdown(r),
                "daily_skew": float(r.skew()),
                "daily_kurtosis": float(r.kurt()),
                "min_daily_return": float(r.min()),
                "max_daily_return": float(r.max()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "asset_return_summary.csv", index=False)
    return table


def rebalance_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    eligible = index[INITIAL_TRAIN_DAYS:-1]
    dates = pd.Series(eligible, index=eligible).resample(REBALANCE_FREQ).last().dropna()
    return set(pd.to_datetime(dates.values))


def performance_table(
    strategy_returns: pd.DataFrame, turnover: pd.DataFrame, costs: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for name in strategy_returns.columns:
        r = strategy_returns[name].dropna()
        ann_ret = annualized_return(r)
        ann_vol = annualized_volatility(r)
        rows.append(
            {
                "strategy": name,
                "start": r.index.min().date().isoformat(),
                "end": r.index.max().date().isoformat(),
                "observations": len(r),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
                "max_drawdown": max_drawdown(r),
                "final_wealth": float((1.0 + r).prod()),
                "avg_daily_turnover": float(turnover[name].mean()),
                "avg_rebalance_turnover": float(turnover.loc[turnover[name] > 0, name].mean()),
                "total_cost": float(costs[name].sum()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "strategy_performance.csv", index=False)
    return table


def excess_table(strategy_returns: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    new_name = f"HMM_MPC_{DEFAULT_DEFENSIVE_TICKER}_weekly"
    rows = []
    perf = performance.set_index("strategy")
    if new_name not in perf.index:
        return pd.DataFrame()
    for baseline in ["EqualWeight_weekly", "Markowitz_CVXPY_weekly"]:
        if baseline not in perf.index:
            continue
        active = (strategy_returns[new_name] - strategy_returns[baseline]).dropna()
        active_return = float(active.mean() * 252)
        tracking_error = float(active.std(ddof=1) * math.sqrt(252))
        rows.append(
            {
                "comparison": f"{new_name} minus {baseline}",
                "annualized_active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": active_return / tracking_error if tracking_error > 0 else np.nan,
                "annualized_return_difference": perf.loc[new_name, "annualized_return"]
                - perf.loc[baseline, "annualized_return"],
                "sharpe_difference": perf.loc[new_name, "sharpe"] - perf.loc[baseline, "sharpe"],
                "max_drawdown_improvement": perf.loc[new_name, "max_drawdown"] - perf.loc[baseline, "max_drawdown"],
                "final_wealth_ratio": perf.loc[new_name, "final_wealth"] / perf.loc[baseline, "final_wealth"],
                "avg_rebalance_turnover_difference": perf.loc[new_name, "avg_rebalance_turnover"]
                - perf.loc[baseline, "avg_rebalance_turnover"],
                "total_cost_difference": perf.loc[new_name, "total_cost"] - perf.loc[baseline, "total_cost"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "excess_performance_vs_baselines.csv", index=False)
    return table


def weighted_mean_and_cov(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        return np.mean(values, axis=0), np.cov(values, rowvar=False), 0.0
    norm = weights / weight_sum
    mean = norm @ values
    centered = values - mean
    cov = (centered.T * norm) @ centered
    return mean, cov, weight_sum


def solve_markowitz(train_returns: pd.DataFrame) -> np.ndarray:
    mu = train_returns.mean().values
    cov = LedoitWolf().fit(train_returns.values).covariance_
    cov = nearest_psd(cov + COV_RIDGE * np.eye(len(mu)))

    w = cp.Variable(len(mu))
    objective = cp.Maximize(mu @ w - 0.5 * GAMMA * cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [cp.sum(w) == 1.0, w >= 0.0, w <= WEIGHT_CAP]
    problem = cp.Problem(objective, constraints)

    for solver in ["OSQP", "CLARABEL", "ECOS"]:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        if problem.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
            weights = np.asarray(w.value).reshape(-1)
            weights = np.maximum(weights, 0.0)
            if weights.sum() <= 0:
                break
            return weights / weights.sum()

    return np.ones(len(mu)) / len(mu)


def fit_hmm_regime_inputs(
    date: pd.Timestamp, returns: pd.DataFrame, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    feature_train = features.loc[:date].tail(INITIAL_TRAIN_DAYS).dropna()
    train_returns = returns.loc[:date].tail(INITIAL_TRAIN_DAYS)
    if len(feature_train) < HMM_MIN_FEATURE_DAYS or len(train_returns) < HMM_MIN_FEATURE_DAYS:
        raise ValueError("Insufficient observations for HMM-MPC target.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_train.values)
    model = GaussianHMM(
        n_components=HMM_STATES,
        covariance_type="diag",
        n_iter=HMM_MAX_ITER,
        tol=1e-4,
        random_state=HMM_RANDOM_STATE,
        min_covar=1e-3,
    )
    model.fit(x_scaled)
    probs = model.predict_proba(x_scaled)

    columns = list(feature_train.columns)
    state_means = model.means_
    stress_score = (
        state_means[:, columns.index("rv_63")]
        + state_means[:, columns.index("amihud_log")]
        + state_means[:, columns.index("vix_log")]
        + state_means[:, columns.index("hy_spread")]
        - state_means[:, columns.index("mom_63")]
    )
    order = np.argsort(stress_score)
    transmat = model.transmat_[np.ix_(order, order)]
    current_prob = probs[-1, order]

    next_returns = returns.shift(-1)
    paired_dates = feature_train.index.intersection(next_returns.dropna().index)
    paired_dates = paired_dates[paired_dates < date]
    paired_returns = next_returns.loc[paired_dates]
    paired_probs = pd.DataFrame(probs, index=feature_train.index).loc[paired_dates].values

    unconditional_mu = train_returns.mean().values
    unconditional_cov = LedoitWolf().fit(train_returns.values).covariance_
    unconditional_cov = nearest_psd(unconditional_cov + COV_RIDGE * np.eye(train_returns.shape[1]))

    mu_states_original = []
    cov_states_original = []
    eff_samples_original = []
    for k in range(HMM_STATES):
        raw_mu, raw_cov, eff_n = weighted_mean_and_cov(paired_returns.values, paired_probs[:, k])
        alpha = eff_n / (eff_n + 252.0)
        if eff_n < 30:
            alpha = 0.0
        mu_k = alpha * raw_mu + (1.0 - alpha) * unconditional_mu
        cov_k = alpha * raw_cov + (1.0 - alpha) * unconditional_cov
        mu_states_original.append(mu_k)
        cov_states_original.append(nearest_psd(cov_k + COV_RIDGE * np.eye(train_returns.shape[1])))
        eff_samples_original.append(eff_n)

    mu_states = np.asarray(mu_states_original)[order]
    cov_states = np.asarray(cov_states_original)[order]
    eff_samples = np.asarray(eff_samples_original)[order]

    q_path = []
    mu_path = []
    cov_path = []
    cost_path = []
    q = current_prob.copy()
    base_cost = TRANSACTION_COST_BPS / 10000.0
    for _ in range(MPC_HORIZON):
        q = q @ transmat
        q = np.maximum(q, 0.0)
        q = q / q.sum()
        mu_mix = q @ mu_states
        cov_mix = np.zeros((train_returns.shape[1], train_returns.shape[1]))
        for k in range(HMM_STATES):
            diff = (mu_states[k] - mu_mix).reshape(-1, 1)
            cov_mix += q[k] * (cov_states[k] + diff @ diff.T)
        cost_multiplier = float(q @ REGIME_COST_MULTIPLIERS)
        q_path.append(q.copy())
        mu_path.append(mu_mix)
        cov_path.append(nearest_psd(cov_mix + COV_RIDGE * np.eye(train_returns.shape[1])))
        cost_path.append(np.ones(train_returns.shape[1]) * base_cost * cost_multiplier)

    meta = {
        "date": date,
        "prob_calm": float(current_prob[0]),
        "prob_transition": float(current_prob[1]),
        "prob_stress": float(current_prob[2]),
        "next_prob_calm": float(q_path[0][0]),
        "next_prob_transition": float(q_path[0][1]),
        "next_prob_stress": float(q_path[0][2]),
        "dominant_regime": REGIME_LABELS[int(np.argmax(current_prob))],
        "next_cost_multiplier": float(q_path[0] @ REGIME_COST_MULTIPLIERS),
        "hmm_log_likelihood": float(model.score(x_scaled)),
        "effective_sample_calm": float(eff_samples[0]),
        "effective_sample_transition": float(eff_samples[1]),
        "effective_sample_stress": float(eff_samples[2]),
    }
    return np.asarray(mu_path), np.asarray(cov_path), np.asarray(cost_path), np.asarray(q_path), meta


def solve_mpc(
    current_weights: np.ndarray, mu_path: np.ndarray, cov_path: np.ndarray, cost_path: np.ndarray
) -> np.ndarray:
    horizon, n_assets = mu_path.shape
    x = cp.Variable((horizon, n_assets))
    b = cp.Variable((horizon, n_assets), nonneg=True)
    s = cp.Variable((horizon, n_assets), nonneg=True)

    terms = []
    constraints = []
    previous = current_weights
    for h in range(horizon):
        trade = b[h] - s[h]
        turnover = 0.5 * cp.sum(b[h] + s[h])
        constraints += [
            x[h] - previous == trade,
            cp.sum(x[h]) == 1.0,
            x[h] >= 0.0,
            x[h] <= WEIGHT_CAP,
            turnover <= MPC_TURNOVER_CAP,
        ]
        terms.append(
            mu_path[h] @ x[h]
            - 0.5 * GAMMA * cp.quad_form(x[h], cp.psd_wrap(cov_path[h]))
            - cost_path[h] @ (b[h] + s[h])
            - 0.5 * MPC_QUADRATIC_TRADE_PENALTY * cp.sum_squares(trade)
        )
        previous = x[h]

    problem = cp.Problem(cp.Maximize(cp.sum(terms)), constraints)
    for solver in ["OSQP", "CLARABEL", "ECOS"]:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        if problem.status in {"optimal", "optimal_inaccurate"} and x.value is not None:
            target = np.asarray(x.value[0]).reshape(-1)
            target = np.maximum(target, 0.0)
            if target.sum() <= 0:
                break
            return target / target.sum()
    raise RuntimeError("MPC solver failed.")


def make_equal_weight_target(n_assets: int) -> Callable[[pd.Timestamp, np.ndarray], np.ndarray]:
    def target(_: pd.Timestamp, __: np.ndarray) -> np.ndarray:
        return np.ones(n_assets) / n_assets

    return target


def make_markowitz_target(returns: pd.DataFrame) -> Callable[[pd.Timestamp, np.ndarray], np.ndarray]:
    def target(date: pd.Timestamp, _: np.ndarray) -> np.ndarray:
        train = returns.loc[:date].tail(INITIAL_TRAIN_DAYS)
        return solve_markowitz(train)

    return target


def make_hmm_mpc_target(
    returns: pd.DataFrame, features: pd.DataFrame
) -> Tuple[Callable[[pd.Timestamp, np.ndarray], Tuple[np.ndarray, np.ndarray]], list]:
    regime_records = []
    base_cost_vector = np.ones(len(returns.columns)) * (TRANSACTION_COST_BPS / 10000.0)

    def target(date: pd.Timestamp, current_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            mu_path, cov_path, cost_path, _q_path, meta = fit_hmm_regime_inputs(date, returns, features)
            target_weights = solve_mpc(current_weights, mu_path, cov_path, cost_path)
            meta["solver_status"] = "hmm_mpc"
            regime_records.append(meta)
            return target_weights, cost_path[0]
        except Exception as exc:
            train = returns.loc[:date].tail(INITIAL_TRAIN_DAYS)
            fallback = solve_markowitz(train)
            regime_records.append(
                {
                    "date": date,
                    "prob_calm": np.nan,
                    "prob_transition": np.nan,
                    "prob_stress": np.nan,
                    "next_prob_calm": np.nan,
                    "next_prob_transition": np.nan,
                    "next_prob_stress": np.nan,
                    "dominant_regime": "fallback",
                    "next_cost_multiplier": 1.0,
                    "hmm_log_likelihood": np.nan,
                    "effective_sample_calm": np.nan,
                    "effective_sample_transition": np.nan,
                    "effective_sample_stress": np.nan,
                    "solver_status": f"fallback_markowitz: {type(exc).__name__}",
                }
            )
            return fallback, base_cost_vector

    return target, regime_records


def run_backtest(
    returns: pd.DataFrame, strategy_name: str, target_weight_fn: Callable[[pd.Timestamp, np.ndarray], np.ndarray]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    dates = returns.index
    rebalance_set = rebalance_dates(dates)
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    daily_returns = []
    daily_turnover = []
    daily_costs = []
    weight_records = []
    cost_per_dollar = TRANSACTION_COST_BPS / 10000.0

    iterator = range(INITIAL_TRAIN_DAYS + 1, len(dates))
    for i in tqdm(iterator, desc=strategy_name, leave=False):
        prev_date = dates[i - 1]
        date = dates[i]
        cost = 0.0
        turnover = 0.0

        if prev_date in rebalance_set:
            target_result = target_weight_fn(prev_date, weights)
            if isinstance(target_result, tuple):
                target, cost_vector = target_result
            else:
                target = target_result
                cost_vector = np.ones(len(returns.columns)) * cost_per_dollar
            target = np.asarray(target, dtype=float)
            target = np.maximum(target, 0.0)
            if target.sum() <= 0:
                target = np.ones(len(target)) / len(target)
            else:
                target = target / target.sum()
            traded = np.abs(target - weights)
            turnover = 0.5 * float(traded.sum())
            cost = float(np.asarray(cost_vector, dtype=float) @ traded)
            weights = target

        asset_return = returns.loc[date].values
        gross_return = float(weights @ asset_return)
        net_return = (1.0 - cost) * (1.0 + gross_return) - 1.0

        denom = 1.0 + gross_return
        if denom > 0:
            weights = weights * (1.0 + asset_return) / denom
            weights = weights / weights.sum()

        daily_returns.append((date, net_return))
        daily_turnover.append((date, turnover))
        daily_costs.append((date, cost))
        weight_records.append(pd.Series(weights, index=returns.columns, name=date))

    return_series = pd.Series(dict(daily_returns), name=strategy_name).sort_index()
    turnover_series = pd.Series(dict(daily_turnover), name=strategy_name).sort_index()
    cost_series = pd.Series(dict(daily_costs), name=strategy_name).sort_index()
    weights_df = pd.DataFrame(weight_records)
    weights_df.index.name = "date"
    return return_series, turnover_series, cost_series, weights_df


def compare_to_root_baseline(performance: pd.DataFrame) -> pd.DataFrame:
    root_perf_path = ROOT_OUTPUTS / "strategy_performance.csv"
    if not root_perf_path.exists():
        return pd.DataFrame()
    root_perf = pd.read_csv(root_perf_path).set_index("strategy")
    new_name = f"HMM_MPC_{DEFAULT_DEFENSIVE_TICKER}_weekly"
    if new_name not in performance.set_index("strategy").index or "HMM_MPC_CVXPY_weekly" not in root_perf.index:
        return pd.DataFrame()

    new_row = performance.set_index("strategy").loc[new_name]
    old_row = root_perf.loc["HMM_MPC_CVXPY_weekly"]
    comparison = pd.DataFrame(
        [
            {
                "metric": "annualized_return",
                "original_value": float(old_row["annualized_return"]),
                "new_value": float(new_row["annualized_return"]),
                "delta": float(new_row["annualized_return"] - old_row["annualized_return"]),
            },
            {
                "metric": "annualized_volatility",
                "original_value": float(old_row["annualized_volatility"]),
                "new_value": float(new_row["annualized_volatility"]),
                "delta": float(new_row["annualized_volatility"] - old_row["annualized_volatility"]),
            },
            {
                "metric": "sharpe",
                "original_value": float(old_row["sharpe"]),
                "new_value": float(new_row["sharpe"]),
                "delta": float(new_row["sharpe"] - old_row["sharpe"]),
            },
            {
                "metric": "max_drawdown",
                "original_value": float(old_row["max_drawdown"]),
                "new_value": float(new_row["max_drawdown"]),
                "delta": float(new_row["max_drawdown"] - old_row["max_drawdown"]),
            },
            {
                "metric": "final_wealth",
                "original_value": float(old_row["final_wealth"]),
                "new_value": float(new_row["final_wealth"]),
                "delta": float(new_row["final_wealth"] - old_row["final_wealth"]),
            },
            {
                "metric": "avg_rebalance_turnover",
                "original_value": float(old_row["avg_rebalance_turnover"]),
                "new_value": float(new_row["avg_rebalance_turnover"]),
                "delta": float(new_row["avg_rebalance_turnover"] - old_row["avg_rebalance_turnover"]),
            },
            {
                "metric": "total_cost",
                "original_value": float(old_row["total_cost"]),
                "new_value": float(new_row["total_cost"]),
                "delta": float(new_row["total_cost"] - old_row["total_cost"]),
            },
        ]
    )
    comparison.to_csv(TABLE_DIR / "comparison_vs_original_model.csv", index=False)
    return comparison


def summarize_regimes(regime_records: pd.DataFrame) -> pd.DataFrame:
    if regime_records.empty:
        table = pd.DataFrame(columns=["metric", "value"])
        table.to_csv(TABLE_DIR / "hmm_regime_summary.csv", index=False)
        return table
    valid = regime_records.dropna(subset=["prob_calm", "prob_transition", "prob_stress"], how="all")
    rows = [
        ("rebalance_observations", int(len(regime_records))),
        ("hmm_mpc_successes", int((regime_records["solver_status"] == "hmm_mpc").sum())),
        ("fallbacks", int((regime_records["solver_status"] != "hmm_mpc").sum())),
    ]
    if not valid.empty:
        rows += [
            ("avg_prob_calm", float(valid["prob_calm"].mean())),
            ("avg_prob_transition", float(valid["prob_transition"].mean())),
            ("avg_prob_stress", float(valid["prob_stress"].mean())),
            ("max_prob_stress", float(valid["prob_stress"].max())),
            ("stress_prob_gt_50pct_count", int((valid["prob_stress"] > 0.50).sum())),
            ("avg_next_cost_multiplier", float(valid["next_cost_multiplier"].mean())),
            ("max_next_cost_multiplier", float(valid["next_cost_multiplier"].max())),
        ]
        for label in REGIME_LABELS:
            rows.append((f"dominant_{label}_count", int((valid["dominant_regime"] == label).sum())))
    table = pd.DataFrame(rows, columns=["metric", "value"])
    table.to_csv(TABLE_DIR / "hmm_regime_summary.csv", index=False)
    return table


def write_report(
    defensive_ticker: str,
    data_quality: pd.DataFrame,
    asset_stats: pd.DataFrame,
    performance: pd.DataFrame,
    excess: pd.DataFrame,
    regime_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    report = f"""# Defensive Asset Experiment

## 改进方向

本方向尝试在原有 sector ETF universe 中加入一个防御资产/现金代理，以缓解原模型“只有股票行业板块可选”的结构性问题。这里使用 `{defensive_ticker}` 作为防御资产，它具有较低久期风险、较低波动率，并且在压力状态下理论上更适合作为风险缓冲。

## 实现方法

1. 复用主工程已经清洗好的 9 个行业 ETF 数据和宏观 regime 特征。
2. 单独拉取 `{defensive_ticker}` 的 Yahoo Finance daily OHLCV 数据，并与原有交易日对齐。
3. 在扩展资产池上重新回测三类策略：
   - 等权
   - 单期 Markowitz + CVXPY
   - HMM-MPC + CVXPY
4. 其中 regime 识别仍然基于原有市场特征，不把防御资产本身塞进 regime 特征里，以便更清楚地隔离“资产池扩展”这一改动的效果。

## 数据来源

- 原有 9 个行业 ETF：仓库内 `data/processed/*.csv`
- 防御资产 `{defensive_ticker}`：Yahoo Finance chart endpoint
- 宏观特征：仓库内 `data/processed/macro_aligned.csv` 和 `regime_features.csv`

## 关键参数

- 样本区间：`{START_DATE}` 到 `2024-12-31`
- 训练窗：`{INITIAL_TRAIN_DAYS}` 个交易日
- 再平衡频率：`{REBALANCE_FREQ}`
- 风险厌恶参数：`{GAMMA:.1f}`
- 单资产权重上限：`{WEIGHT_CAP:.2f}`
- 交易成本：`{TRANSACTION_COST_BPS:.1f}` bps
- HMM 状态数：`{HMM_STATES}`
- MPC horizon：`{MPC_HORIZON}`
- 压力状态成本倍数：`{REGIME_COST_MULTIPLIERS[-1]:.1f}`

## 结果表

### 策略表现

{performance.to_markdown(index=False)}

### 相对基准的超额表现

{excess.to_markdown(index=False) if not excess.empty else "No excess table."}

### 相对原始模型的变化

{comparison.to_markdown(index=False) if not comparison.empty else "No comparison table."}

### 数据质量

{data_quality.to_markdown(index=False)}

### 防御资产组合下的 HMM 状态统计

{regime_summary.to_markdown(index=False) if not regime_summary.empty else "No regime summary."}

### 资产统计

{asset_stats.to_markdown(index=False)}

## 是否提升

判断口径分两层：

1. 绝对表现：看加入 `{defensive_ticker}` 后，HMM-MPC 是否同时改善收益、Sharpe、回撤和换手。
2. 相对原始模型：看新版本相较于原始 9 资产 HMM-MPC 是否更稳，特别是是否降低回撤和成本。

从结果表可以直接读出，新模型是否真的“更好”，而不只是换了一个更保守的仓位配置。

## 原因分析

如果结果改善，通常是因为防御资产给了 regime-aware optimizer 一个真实的去风险出口，压力状态不必再在股票行业之间内部轮动。

如果结果没有明显改善，常见原因有三类：

1. 防御资产收益太低，在长牛阶段会拖累组合收益。
2. regime 信号仍然主要用于识别风险，而不是预测行业相对收益。
3. 当前风险厌恶和交易成本参数仍偏保守，导致模型过度偏向低波动资产。

本实验的价值在于把“资产池太硬”这个结构问题单独拆出来验证，而不是继续把问题都归咎于 HMM 本身。
"""
    (REPORT_DIR / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Defensive asset improvement experiment.")
    parser.add_argument("--defensive-ticker", default=DEFAULT_DEFENSIVE_TICKER)
    parser.add_argument("--refresh", action="store_true", help="Refresh defensive asset download.")
    args = parser.parse_args()

    ensure_dirs()

    base_adj_close, base_close, base_volume, macro, _base_features = load_root_data()
    adj_close, close, volume, returns = build_extended_universe(
        base_adj_close, base_close, base_volume, args.defensive_ticker, defensive_refresh=args.refresh
    )
    macro = macro.reindex(returns.index).ffill()
    features = build_regime_features(adj_close, close, volume, macro)
    common = returns.index.intersection(features.index)
    returns = returns.loc[common]
    features = features.loc[common]
    macro = macro.loc[common]
    adj_close = adj_close.loc[common]
    close = close.loc[common]
    volume = volume.loc[common]

    data_quality = data_quality_report(adj_close, close, volume, returns, macro, features)
    asset_stats = asset_summary(returns)

    equal_weight_target = make_equal_weight_target(len(returns.columns))
    markowitz_target = make_markowitz_target(returns)
    hmm_target, regime_records = make_hmm_mpc_target(returns, features)

    ew_r, ew_t, ew_c, ew_w = run_backtest(returns, "EqualWeight_weekly", equal_weight_target)
    mv_r, mv_t, mv_c, mv_w = run_backtest(returns, "Markowitz_CVXPY_weekly", markowitz_target)
    hmm_name = f"HMM_MPC_{args.defensive_ticker}_weekly"
    hmm_r, hmm_t, hmm_c, hmm_w = run_backtest(returns, hmm_name, hmm_target)

    strategy_returns = pd.DataFrame(
        {
            "EqualWeight_weekly": ew_r,
            "Markowitz_CVXPY_weekly": mv_r,
            hmm_name: hmm_r,
        }
    )
    turnover = pd.DataFrame(
        {"EqualWeight_weekly": ew_t, "Markowitz_CVXPY_weekly": mv_t, hmm_name: hmm_t}
    )
    costs = pd.DataFrame({"EqualWeight_weekly": ew_c, "Markowitz_CVXPY_weekly": mv_c, hmm_name: hmm_c})

    strategy_returns.to_csv(TABLE_DIR / "strategy_daily_returns.csv")
    turnover.to_csv(TABLE_DIR / "strategy_daily_turnover.csv")
    costs.to_csv(TABLE_DIR / "strategy_daily_costs.csv")
    ew_w.to_csv(TABLE_DIR / "equal_weight_daily_weights.csv")
    mv_w.to_csv(TABLE_DIR / "markowitz_daily_weights.csv")
    hmm_w.to_csv(TABLE_DIR / "hmm_mpc_daily_weights.csv")
    pd.DataFrame(regime_records).to_csv(TABLE_DIR / "hmm_regime_records.csv", index=False)

    performance = performance_table(strategy_returns, turnover, costs)
    excess = excess_table(strategy_returns, performance)
    regime_summary = summarize_regimes(pd.DataFrame(regime_records))
    comparison = compare_to_root_baseline(performance)

    performance.to_csv(TABLE_DIR / "strategy_performance.csv", index=False)
    excess.to_csv(TABLE_DIR / "excess_performance_vs_baselines.csv", index=False)
    comparison.to_csv(TABLE_DIR / "comparison_vs_original_model.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "strategy": hmm_name,
                "annualized_return": performance.set_index("strategy").loc[hmm_name, "annualized_return"],
                "annualized_volatility": performance.set_index("strategy").loc[hmm_name, "annualized_volatility"],
                "sharpe": performance.set_index("strategy").loc[hmm_name, "sharpe"],
                "max_drawdown": performance.set_index("strategy").loc[hmm_name, "max_drawdown"],
                "final_wealth": performance.set_index("strategy").loc[hmm_name, "final_wealth"],
            }
        ]
    )
    summary.to_csv(TABLE_DIR / "headline_results.csv", index=False)

    write_report(args.defensive_ticker, data_quality, asset_stats, performance, excess, regime_summary, comparison)

    print(f"Completed defensive asset experiment with {args.defensive_ticker}.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
