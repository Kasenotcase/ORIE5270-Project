from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Callable, Tuple

import cvxpy as cp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

PYTHON_ENV = "/opt/anaconda3/envs/pytorch_py=3.8/bin/python"

ASSETS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"  # yfinance end date is exclusive.
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
    for path in [RAW_DIR, PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def field_frame(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex dataframe from yfinance.")
    if field in data.columns.get_level_values(1):
        frame = data.xs(field, axis=1, level=1)
    elif field in data.columns.get_level_values(0):
        frame = data.xs(field, axis=1, level=0)
    else:
        raise KeyError(f"Field {field} not found in downloaded data.")
    frame = frame.reindex(columns=ASSETS)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame


def fetch_price_data(refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_path = PROCESSED_DIR / "adj_close.csv"
    close_path = PROCESSED_DIR / "close.csv"
    volume_path = PROCESSED_DIR / "volume.csv"

    if not refresh and price_path.exists() and close_path.exists() and volume_path.exists():
        adj_close = pd.read_csv(price_path, index_col=0, parse_dates=True)
        close = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        return adj_close, close, volume

    try:
        data = yf.download(
            ASSETS,
            start=START_DATE,
            end=END_DATE,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise RuntimeError("yfinance returned no ETF data.")
        data.to_csv(RAW_DIR / "yfinance_ohlcv_raw.csv")
        adj_close = field_frame(data, "Adj Close")
        close = field_frame(data, "Close")
        volume = field_frame(data, "Volume")
        (PROCESSED_DIR / "price_data_source.txt").write_text("yfinance download\n", encoding="utf-8")
    except Exception as exc:
        print(f"yfinance failed ({exc}); falling back to Yahoo Finance chart endpoint.")
        adj_close, close, volume = fetch_yahoo_chart_data()
        (PROCESSED_DIR / "price_data_source.txt").write_text(
            "Yahoo Finance chart endpoint fallback\n", encoding="utf-8"
        )

    adj_close.to_csv(price_path)
    close.to_csv(close_path)
    volume.to_csv(volume_path)
    return adj_close, close, volume


def fetch_yahoo_chart_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {}
    period1 = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
    period2 = int(pd.Timestamp(END_DATE, tz="UTC").timestamp())
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in ASSETS:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        error = payload.get("chart", {}).get("error")
        if error is not None:
            raise RuntimeError(f"Yahoo chart error for {ticker}: {error}")
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
        frame = pd.DataFrame(
            {
                "Open": quote["open"],
                "High": quote["high"],
                "Low": quote["low"],
                "Close": quote["close"],
                "Adj Close": adj,
                "Volume": quote["volume"],
            },
            index=index,
        )
        frames[ticker] = frame

    raw = pd.concat(frames, axis=1).sort_index()
    raw.to_csv(RAW_DIR / "yahoo_chart_ohlcv_raw.csv")

    adj_close = pd.DataFrame({ticker: frames[ticker]["Adj Close"] for ticker in ASSETS})
    close = pd.DataFrame({ticker: frames[ticker]["Close"] for ticker in ASSETS})
    volume = pd.DataFrame({ticker: frames[ticker]["Volume"] for ticker in ASSETS})
    adj_close.index.name = "Date"
    close.index.name = "Date"
    volume.index.name = "Date"
    return adj_close.sort_index(), close.sort_index(), volume.sort_index()


def fetch_fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    path = RAW_DIR / f"fred_{series_id}.csv"
    if not refresh and path.exists():
        raw = pd.read_csv(path)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        raw = pd.read_csv(url)
        raw.to_csv(path, index=False)

    date_col = "DATE" if "DATE" in raw.columns else "observation_date"
    raw[date_col] = pd.to_datetime(raw[date_col])
    values = pd.to_numeric(raw[series_id].replace(".", np.nan), errors="coerce")
    series = pd.Series(values.values, index=raw[date_col], name=series_id).sort_index()
    return series


def fetch_macro_data(refresh: bool = False) -> pd.DataFrame:
    macro_path = PROCESSED_DIR / "macro.csv"
    if not refresh and macro_path.exists():
        return pd.read_csv(macro_path, index_col=0, parse_dates=True)

    macro = pd.concat(
        [
            fetch_fred_series("VIXCLS", refresh=refresh),
            fetch_fred_series("BAMLH0A0HYM2", refresh=refresh),
        ],
        axis=1,
    )
    macro.to_csv(macro_path)
    return macro


def clean_data(
    adj_close: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, macro: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adj_close = adj_close.replace([np.inf, -np.inf], np.nan)
    close = close.replace([np.inf, -np.inf], np.nan)
    volume = volume.replace([np.inf, -np.inf], np.nan)

    valid = adj_close.notna().all(axis=1) & close.notna().all(axis=1) & volume.notna().all(axis=1)
    valid &= (adj_close > 0).all(axis=1) & (close > 0).all(axis=1) & (volume >= 0).all(axis=1)

    adj_close = adj_close.loc[valid]
    close = close.loc[valid]
    volume = volume.loc[valid]

    returns = adj_close.pct_change().dropna(how="any")
    macro = macro.reindex(adj_close.index).ffill()
    macro = macro.loc[returns.index]
    close = close.loc[returns.index]
    volume = volume.loc[returns.index]

    adj_close.loc[returns.index].to_csv(PROCESSED_DIR / "adj_close_clean.csv")
    close.to_csv(PROCESSED_DIR / "close_clean.csv")
    volume.to_csv(PROCESSED_DIR / "volume_clean.csv")
    returns.to_csv(PROCESSED_DIR / "returns.csv")
    macro.to_csv(PROCESSED_DIR / "macro_aligned.csv")
    return adj_close.loc[returns.index], close, volume, returns, macro


def build_regime_features(
    adj_close: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame, macro: pd.DataFrame
) -> pd.DataFrame:
    market_price = adj_close.mean(axis=1)
    market_return = market_price.pct_change()
    log_market = np.log(market_price)

    features = pd.DataFrame(index=returns.index)
    features["mom_21"] = log_market.diff(21)
    features["mom_63"] = log_market.diff(63)
    features["drawdown_63"] = market_price / market_price.rolling(63).max() - 1.0
    features["rv_21"] = market_return.rolling(21).std() * math.sqrt(252)
    features["rv_63"] = market_return.rolling(63).std() * math.sqrt(252)

    dollar_volume = close * volume
    illiq = returns.abs() / dollar_volume.replace(0, np.nan)
    features["amihud_log"] = np.log1p(illiq.mean(axis=1))

    features["vix_log"] = np.log(macro["VIXCLS"])
    features["vix_change_21"] = np.log(macro["VIXCLS"]).diff(21)
    features["hy_spread"] = macro["BAMLH0A0HYM2"]
    features["hy_spread_change_21"] = macro["BAMLH0A0HYM2"].diff(21)

    features = features.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    features.to_csv(PROCESSED_DIR / "regime_features.csv")
    return features


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * math.sqrt(252))


def performance_table(
    strategy_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    filename: str = "strategy_performance.csv",
) -> pd.DataFrame:
    rows = []
    for name in strategy_returns.columns:
        r = strategy_returns[name].dropna()
        ann_ret = annualized_return(r)
        ann_vol = annualized_volatility(r)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rows.append(
            {
                "strategy": name,
                "start": r.index.min().date().isoformat(),
                "end": r.index.max().date().isoformat(),
                "observations": len(r),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown(r),
                "final_wealth": float((1.0 + r).prod()),
                "avg_daily_turnover": float(turnover[name].mean()),
                "avg_rebalance_turnover": float(turnover.loc[turnover[name] > 0, name].mean()),
                "total_cost": float(costs[name].sum()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / filename, index=False)
    return table


def active_performance_table(strategy_returns: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    proposed_name = "HMM_MPC_CVXPY_weekly"
    if proposed_name not in strategy_returns.columns:
        return pd.DataFrame()

    proposed_perf = performance.set_index("strategy").loc[proposed_name]
    rows = []
    for baseline in ["EqualWeight_weekly", "Markowitz_CVXPY_weekly"]:
        if baseline not in strategy_returns.columns:
            continue
        active = (strategy_returns[proposed_name] - strategy_returns[baseline]).dropna()
        active_return = float(active.mean() * 252)
        tracking_error = float(active.std(ddof=1) * math.sqrt(252))
        base_perf = performance.set_index("strategy").loc[baseline]
        rows.append(
            {
                "comparison": f"{proposed_name} minus {baseline}",
                "annualized_active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": active_return / tracking_error if tracking_error > 0 else np.nan,
                "annualized_return_difference": proposed_perf["annualized_return"] - base_perf["annualized_return"],
                "sharpe_difference": proposed_perf["sharpe"] - base_perf["sharpe"],
                "max_drawdown_improvement": proposed_perf["max_drawdown"] - base_perf["max_drawdown"],
                "final_wealth_ratio": proposed_perf["final_wealth"] / base_perf["final_wealth"],
                "avg_rebalance_turnover_difference": (
                    proposed_perf["avg_rebalance_turnover"] - base_perf["avg_rebalance_turnover"]
                ),
                "total_cost_difference": proposed_perf["total_cost"] - base_perf["total_cost"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "excess_performance_vs_baselines.csv", index=False)
    return table


def hmm_regime_summary(regime_records: pd.DataFrame) -> pd.DataFrame:
    if regime_records.empty:
        return pd.DataFrame(columns=["metric", "value"])
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


def asset_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        ann_vol = annualized_volatility(r)
        rows.append(
            {
                "asset": ticker,
                "annualized_return": annualized_return(r),
                "annualized_volatility": ann_vol,
                "sharpe": annualized_return(r) / ann_vol if ann_vol > 0 else np.nan,
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


def data_quality_report(
    raw_adj_close: pd.DataFrame,
    raw_close: pd.DataFrame,
    raw_volume: pd.DataFrame,
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
        ("asset_count", len(ASSETS)),
        ("raw_price_start", raw_adj_close.index.min().date().isoformat()),
        ("raw_price_end", raw_adj_close.index.max().date().isoformat()),
        ("clean_return_start", returns.index.min().date().isoformat()),
        ("clean_return_end", returns.index.max().date().isoformat()),
        ("return_observations", len(returns)),
        ("raw_adj_close_missing_cells", int(raw_adj_close.isna().sum().sum())),
        ("raw_close_missing_cells", int(raw_close.isna().sum().sum())),
        ("raw_volume_missing_cells", int(raw_volume.isna().sum().sum())),
        ("zero_volume_cells_clean_sample", int((raw_volume.loc[returns.index] == 0).sum().sum())),
        ("abs_daily_return_gt_20pct_cells", int((returns.abs() > 0.20).sum().sum())),
        ("macro_missing_after_alignment", int(macro.isna().sum().sum())),
        ("feature_start", features.index.min().date().isoformat()),
        ("feature_end", features.index.max().date().isoformat()),
        ("feature_observations", len(features)),
        ("rolling_cov_condition_median", float(np.median(cond_numbers)) if cond_numbers else np.nan),
        ("rolling_cov_condition_max", float(np.max(cond_numbers)) if cond_numbers else np.nan),
    ]
    table = pd.DataFrame(checks, columns=["check", "value"])
    table.to_csv(TABLE_DIR / "data_quality_summary.csv", index=False)
    return table


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, COV_RIDGE)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


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
            weights = weights / weights.sum()
            return weights

    return np.ones(len(mu)) / len(mu)


def weighted_mean_and_cov(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        mean = np.mean(values, axis=0)
        cov = np.cov(values, rowvar=False)
        return mean, cov, 0.0

    norm_weights = weights / weight_sum
    mean = norm_weights @ values
    centered = values - mean
    cov = (centered.T * norm_weights) @ centered
    return mean, cov, weight_sum


def fit_hmm_regime_inputs(
    date: pd.Timestamp,
    returns: pd.DataFrame,
    features: pd.DataFrame,
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
    unconditional_cov = nearest_psd(unconditional_cov + COV_RIDGE * np.eye(len(ASSETS)))

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
        cov_states_original.append(nearest_psd(cov_k + COV_RIDGE * np.eye(len(ASSETS))))
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
    for _h in range(MPC_HORIZON):
        q = q @ transmat
        q = np.maximum(q, 0.0)
        q = q / q.sum()
        mu_mix = q @ mu_states
        cov_mix = np.zeros((len(ASSETS), len(ASSETS)))
        for k in range(HMM_STATES):
            diff = (mu_states[k] - mu_mix).reshape(-1, 1)
            cov_mix += q[k] * (cov_states[k] + diff @ diff.T)
        cost_multiplier = float(q @ REGIME_COST_MULTIPLIERS)
        q_path.append(q.copy())
        mu_path.append(mu_mix)
        cov_path.append(nearest_psd(cov_mix + COV_RIDGE * np.eye(len(ASSETS))))
        cost_path.append(np.ones(len(ASSETS)) * base_cost * cost_multiplier)

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
    current_weights: np.ndarray,
    mu_path: np.ndarray,
    cov_path: np.ndarray,
    cost_path: np.ndarray,
) -> np.ndarray:
    horizon, n_assets = mu_path.shape
    x = cp.Variable((horizon, n_assets))
    b = cp.Variable((horizon, n_assets), nonneg=True)
    s = cp.Variable((horizon, n_assets), nonneg=True)

    objective_terms = []
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
        objective_terms.append(
            mu_path[h] @ x[h]
            - 0.5 * GAMMA * cp.quad_form(x[h], cp.psd_wrap(cov_path[h]))
            - cost_path[h] @ (b[h] + s[h])
            - 0.5 * MPC_QUADRATIC_TRADE_PENALTY * cp.sum_squares(trade)
        )
        previous = x[h]

    problem = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints)
    for solver in ["OSQP", "CLARABEL", "ECOS"]:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        if problem.status in {"optimal", "optimal_inaccurate"} and x.value is not None:
            target = np.asarray(x.value[0]).reshape(-1)
            target = np.maximum(target, 0.0)
            target = target / target.sum()
            return target
    raise RuntimeError("MPC solver failed.")


def make_hmm_mpc_target(
    returns: pd.DataFrame,
    features: pd.DataFrame,
) -> Tuple[Callable[[pd.Timestamp, np.ndarray], Tuple[np.ndarray, np.ndarray]], list]:
    regime_records = []
    base_cost_vector = np.ones(len(ASSETS)) * (TRANSACTION_COST_BPS / 10000.0)

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


def rebalance_dates(index: pd.DatetimeIndex) -> set:
    eligible = index[INITIAL_TRAIN_DAYS:-1]
    dates = pd.Series(eligible, index=eligible).resample(REBALANCE_FREQ).last().dropna()
    return set(pd.to_datetime(dates.values))


def run_backtest(
    returns: pd.DataFrame,
    strategy_name: str,
    target_weight_fn: Callable[[pd.Timestamp, np.ndarray], np.ndarray],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    dates = returns.index
    rebalance_set = rebalance_dates(dates)
    weights = np.ones(len(ASSETS)) / len(ASSETS)
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
                cost_vector = np.ones(len(ASSETS)) * cost_per_dollar
            target = np.asarray(target, dtype=float)
            target = np.maximum(target, 0.0)
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
        weight_records.append(pd.Series(weights, index=ASSETS, name=date))

    return_series = pd.Series(dict(daily_returns), name=strategy_name).sort_index()
    turnover_series = pd.Series(dict(daily_turnover), name=strategy_name).sort_index()
    cost_series = pd.Series(dict(daily_costs), name=strategy_name).sort_index()
    weights_df = pd.DataFrame(weight_records)
    weights_df.index.name = "date"
    return return_series, turnover_series, cost_series, weights_df


def plot_outputs(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    macro: pd.DataFrame,
    strategy_returns: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    asset_wealth = (1.0 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    for col in asset_wealth.columns:
        plt.plot(asset_wealth.index, asset_wealth[col], linewidth=1.0, label=col)
    plt.title("Sector ETF Cumulative Wealth")
    plt.ylabel("Growth of $1")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "asset_cumulative_returns.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="vlag", center=0.0, square=True)
    plt.title("Daily Return Correlation")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "asset_return_correlation.png", dpi=160)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(macro.index, macro["VIXCLS"], color="tab:red", linewidth=0.9, label="VIXCLS")
    ax1.set_ylabel("VIX")
    ax2 = ax1.twinx()
    ax2.plot(
        macro.index,
        macro["BAMLH0A0HYM2"],
        color="tab:blue",
        linewidth=0.9,
        label="High-yield OAS",
    )
    ax2.set_ylabel("High-yield OAS")
    ax1.set_title("Daily Macro Stress Proxies")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "macro_stress_proxies.png", dpi=160)
    plt.close(fig)

    feature_cols = ["mom_63", "rv_63", "amihud_log", "vix_log", "hy_spread"]
    axes = features[feature_cols].plot(subplots=True, figsize=(12, 9), linewidth=0.8, title="Regime Feature Diagnostics")
    for ax in np.ravel(axes):
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "regime_feature_diagnostics.png", dpi=160)
    plt.close()

    wealth = (1.0 + strategy_returns).cumprod()
    plt.figure(figsize=(12, 6))
    for col in wealth.columns:
        plt.plot(wealth.index, wealth[col], linewidth=1.3, label=col)
    plt.title("Strategy Cumulative Wealth")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "strategy_cumulative_wealth.png", dpi=160)
    plt.close()

    drawdowns = wealth / wealth.cummax() - 1.0
    plt.figure(figsize=(12, 5))
    for col in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[col], linewidth=1.1, label=col)
    plt.title("Strategy Drawdowns")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "strategy_drawdowns.png", dpi=160)
    plt.close()


def plot_hmm_mpc_outputs(regime_records: pd.DataFrame, proposed_weights: pd.DataFrame) -> None:
    if not regime_records.empty:
        probs = regime_records.set_index("date")[["prob_calm", "prob_transition", "prob_stress"]]
        probs = probs.dropna(how="all")
        if not probs.empty:
            plt.figure(figsize=(12, 5))
            plt.stackplot(
                probs.index,
                probs["prob_calm"],
                probs["prob_transition"],
                probs["prob_stress"],
                labels=["calm", "transition", "stress"],
                colors=["#4c78a8", "#f2cf5b", "#d95f5f"],
                alpha=0.85,
            )
            plt.title("HMM Filtered Regime Probabilities at Rebalance Dates")
            plt.ylabel("Probability")
            plt.ylim(0, 1)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "hmm_regime_probabilities.png", dpi=160)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(probs.index, probs["prob_stress"], color="#b22222", linewidth=1.2)
            plt.title("HMM Stress Probability")
            plt.ylabel("Stress probability")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "hmm_stress_probability.png", dpi=160)
            plt.close()

    if not proposed_weights.empty:
        monthly_weights = proposed_weights.resample("M").last()
        plt.figure(figsize=(12, 6))
        plt.stackplot(monthly_weights.index, monthly_weights.T, labels=monthly_weights.columns, alpha=0.9)
        plt.title("HMM-MPC Portfolio Weights")
        plt.ylabel("Weight")
        plt.ylim(0, 1)
        plt.legend(ncol=3, fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "hmm_mpc_weights.png", dpi=160)
        plt.close()


def write_report(
    data_quality: pd.DataFrame,
    asset_stats: pd.DataFrame,
    performance: pd.DataFrame,
    excess_performance: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> None:
    source_path = PROCESSED_DIR / "price_data_source.txt"
    price_data_source = source_path.read_text(encoding="utf-8").strip() if source_path.exists() else "unknown"

    perf_md = performance.copy()
    for col in [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "final_wealth",
        "avg_daily_turnover",
        "avg_rebalance_turnover",
        "total_cost",
    ]:
        perf_md[col] = perf_md[col].map(lambda x: f"{x:.4f}")

    excess_md = excess_performance.copy()
    if not excess_md.empty:
        for col in [
            "annualized_active_return",
            "tracking_error",
            "information_ratio",
            "annualized_return_difference",
            "sharpe_difference",
            "max_drawdown_improvement",
            "final_wealth_ratio",
            "avg_rebalance_turnover_difference",
            "total_cost_difference",
        ]:
            excess_md[col] = excess_md[col].map(lambda x: f"{x:.4f}")

    asset_md = asset_stats.copy()
    for col in [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "daily_skew",
        "daily_kurtosis",
        "min_daily_return",
        "max_daily_return",
    ]:
        asset_md[col] = asset_md[col].map(lambda x: f"{x:.4f}")

    regime_md = regime_summary.copy()
    if not regime_md.empty:
        regime_md["value"] = regime_md["value"].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    report = f"""# Empirical Strategy Report

## Framework

- Python environment: `{PYTHON_ENV}`
- Price data source used: `{price_data_source}`
- Macro data source used: `FRED CSV, VIXCLS and BAMLH0A0HYM2`
- Asset universe: `{", ".join(ASSETS)}`
- Data window: `{START_DATE}` to `2024-12-31`
- Training scheme: rolling `{INITIAL_TRAIN_DAYS}` trading-day estimation window
- Rebalance frequency: `{REBALANCE_FREQ}`
- Transaction cost assumption: `{TRANSACTION_COST_BPS}` basis points per dollar traded
- Markowitz solver: CVXPY with OSQP first and CLARABEL/ECOS fallback
- Markowitz constraints: long-only, full investment, max weight `{WEIGHT_CAP:.2f}`
- Markowitz risk aversion: `{GAMMA:.1f}`
- HMM states: `{HMM_STATES}`, sorted as calm, transition, stress
- HMM estimation: rolling `{INITIAL_TRAIN_DAYS}` trading-day feature window, diagonal Gaussian emissions
- MPC horizon: `{MPC_HORIZON}` trading days
- MPC turnover cap per rebalance: `{MPC_TURNOVER_CAP:.2f}`
- Regime cost multipliers: calm `{REGIME_COST_MULTIPLIERS[0]:.1f}`, transition `{REGIME_COST_MULTIPLIERS[1]:.1f}`, stress `{REGIME_COST_MULTIPLIERS[2]:.1f}`

The empirical design first establishes clean baseline strategies and then evaluates a proposed HMM-MPC strategy. The HMM is estimated only from information available through the rebalance date. Its regime probabilities determine conditional return, covariance, and trading-cost inputs. The optimizer remains a convex multi-period CVXPY problem.

## Data Quality Summary

{data_quality.to_markdown(index=False)}

## Asset Return Summary

{asset_md.to_markdown(index=False)}

## Strategy Performance

{perf_md.to_markdown(index=False)}

## Excess Performance of HMM-MPC

{excess_md.to_markdown(index=False) if not excess_md.empty else "No excess-performance table was generated."}

## HMM Regime Diagnostics

{regime_md.to_markdown(index=False) if not regime_md.empty else "No HMM regime diagnostics were generated."}

## Generated Outputs

- `data/processed/returns.csv`
- `data/processed/regime_features.csv`
- `data/processed/price_data_source.txt`
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/asset_return_summary.csv`
- `outputs/tables/baseline_performance.csv`
- `outputs/tables/strategy_performance.csv`
- `outputs/tables/excess_performance_vs_baselines.csv`
- `outputs/tables/hmm_regime_records.csv`
- `outputs/tables/hmm_regime_summary.csv`
- `outputs/tables/baseline_daily_returns.csv`
- `outputs/tables/baseline_daily_turnover.csv`
- `outputs/tables/baseline_daily_costs.csv`
- `outputs/tables/strategy_daily_returns.csv`
- `outputs/tables/strategy_daily_turnover.csv`
- `outputs/tables/strategy_daily_costs.csv`
- `outputs/tables/hmm_mpc_daily_weights.csv`
- `outputs/figures/asset_cumulative_returns.png`
- `outputs/figures/asset_return_correlation.png`
- `outputs/figures/macro_stress_proxies.png`
- `outputs/figures/regime_feature_diagnostics.png`
- `outputs/figures/strategy_cumulative_wealth.png`
- `outputs/figures/strategy_drawdowns.png`
- `outputs/figures/hmm_regime_probabilities.png`
- `outputs/figures/hmm_stress_probability.png`
- `outputs/figures/hmm_mpc_weights.png`
"""
    (REPORT_DIR / "empirical_summary.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "baseline_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download data again even if cached files exist.")
    args = parser.parse_args()

    ensure_dirs()

    print("Downloading or loading data...")
    raw_adj_close, raw_close, raw_volume = fetch_price_data(refresh=args.refresh)
    raw_macro = fetch_macro_data(refresh=args.refresh)

    print("Cleaning data and building features...")
    adj_close, close, volume, returns, macro = clean_data(raw_adj_close, raw_close, raw_volume, raw_macro)
    features = build_regime_features(adj_close, close, volume, returns, macro)

    print("Running diagnostics and descriptive statistics...")
    dq = data_quality_report(raw_adj_close, raw_close, raw_volume, returns, macro, features)
    asset_stats = asset_summary(returns)

    def equal_weight_target(_date: pd.Timestamp, _weights: np.ndarray) -> np.ndarray:
        return np.ones(len(ASSETS)) / len(ASSETS)

    def markowitz_target(date: pd.Timestamp, _weights: np.ndarray) -> np.ndarray:
        train = returns.loc[:date].tail(INITIAL_TRAIN_DAYS)
        return solve_markowitz(train)

    print("Running baseline backtests...")
    ew_r, ew_to, ew_cost, ew_w = run_backtest(returns, "EqualWeight_weekly", equal_weight_target)
    mv_r, mv_to, mv_cost, mv_w = run_backtest(returns, "Markowitz_CVXPY_weekly", markowitz_target)

    baseline_returns = pd.concat([ew_r, mv_r], axis=1).dropna()
    baseline_turnover = pd.concat([ew_to, mv_to], axis=1).reindex(baseline_returns.index)
    baseline_costs = pd.concat([ew_cost, mv_cost], axis=1).reindex(baseline_returns.index)

    baseline_returns.to_csv(TABLE_DIR / "baseline_daily_returns.csv")
    baseline_turnover.to_csv(TABLE_DIR / "baseline_daily_turnover.csv")
    baseline_costs.to_csv(TABLE_DIR / "baseline_daily_costs.csv")
    ew_w.to_csv(TABLE_DIR / "equal_weight_daily_weights.csv")
    mv_w.to_csv(TABLE_DIR / "markowitz_daily_weights.csv")

    baseline_performance = performance_table(
        baseline_returns, baseline_turnover, baseline_costs, filename="baseline_performance.csv"
    )

    print("Running HMM-MPC proposed strategy...")
    hmm_mpc_target, regime_records = make_hmm_mpc_target(returns, features)
    mpc_r, mpc_to, mpc_cost, mpc_w = run_backtest(returns, "HMM_MPC_CVXPY_weekly", hmm_mpc_target)

    strategy_returns = pd.concat([ew_r, mv_r, mpc_r], axis=1).dropna()
    turnover = pd.concat([ew_to, mv_to, mpc_to], axis=1).reindex(strategy_returns.index)
    costs = pd.concat([ew_cost, mv_cost, mpc_cost], axis=1).reindex(strategy_returns.index)

    strategy_returns.to_csv(TABLE_DIR / "strategy_daily_returns.csv")
    turnover.to_csv(TABLE_DIR / "strategy_daily_turnover.csv")
    costs.to_csv(TABLE_DIR / "strategy_daily_costs.csv")
    mpc_w.to_csv(TABLE_DIR / "hmm_mpc_daily_weights.csv")

    regime_records_df = pd.DataFrame(regime_records)
    if not regime_records_df.empty:
        regime_records_df["date"] = pd.to_datetime(regime_records_df["date"])
    regime_records_df.to_csv(TABLE_DIR / "hmm_regime_records.csv", index=False)

    performance = performance_table(strategy_returns, turnover, costs, filename="strategy_performance.csv")
    excess_performance = active_performance_table(strategy_returns, performance)
    regime_summary = hmm_regime_summary(regime_records_df)

    print("Generating plots and report...")
    plot_outputs(returns, features, macro, strategy_returns)
    plot_hmm_mpc_outputs(regime_records_df, mpc_w)
    write_report(dq, asset_stats, performance, excess_performance, regime_summary)

    print("\nBaseline performance:")
    print(baseline_performance.to_string(index=False))
    print("\nStrategy performance:")
    print(performance.to_string(index=False))
    print("\nExcess performance:")
    print(excess_performance.to_string(index=False))
    print(f"\nReport written to: {REPORT_DIR / 'empirical_summary.md'}")


if __name__ == "__main__":
    main()
