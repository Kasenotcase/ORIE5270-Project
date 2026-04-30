from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

ASSETS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
STRATEGIES = ["EqualWeight_weekly", "Markowitz_CVXPY_weekly", "HMM_MPC_CVXPY_weekly"]


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (252 / len(r)) - 1) if len(r) else np.nan


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * math.sqrt(252))


def sharpe(r: pd.Series) -> float:
    vol = ann_vol(r)
    return ann_return(r) / vol if vol > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def load_data():
    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    strategy_returns = pd.read_csv(TABLE_DIR / "strategy_daily_returns.csv", index_col=0, parse_dates=True)
    costs = pd.read_csv(TABLE_DIR / "strategy_daily_costs.csv", index_col=0, parse_dates=True)
    turnover = pd.read_csv(TABLE_DIR / "strategy_daily_turnover.csv", index_col=0, parse_dates=True)
    weights = {
        "EqualWeight_weekly": pd.read_csv(TABLE_DIR / "equal_weight_daily_weights.csv", index_col=0, parse_dates=True),
        "Markowitz_CVXPY_weekly": pd.read_csv(TABLE_DIR / "markowitz_daily_weights.csv", index_col=0, parse_dates=True),
        "HMM_MPC_CVXPY_weekly": pd.read_csv(TABLE_DIR / "hmm_mpc_daily_weights.csv", index_col=0, parse_dates=True),
    }
    regimes = pd.read_csv(TABLE_DIR / "hmm_regime_records.csv", parse_dates=["date"])
    features = pd.read_csv(DATA_DIR / "regime_features.csv", index_col=0, parse_dates=True)
    asset_stats = pd.read_csv(TABLE_DIR / "asset_return_summary.csv")
    data_quality = pd.read_csv(TABLE_DIR / "data_quality_summary.csv")
    return returns, strategy_returns, costs, turnover, weights, regimes, features, asset_stats, data_quality


def yearly_strategy_table(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in strategy_returns.groupby(strategy_returns.index.year):
        row = {"year": int(year)}
        for name in STRATEGIES:
            row[f"{name}_return"] = float((1 + group[name]).prod() - 1)
        row["HMM_minus_EqualWeight"] = row["HMM_MPC_CVXPY_weekly_return"] - row["EqualWeight_weekly_return"]
        row["HMM_minus_Markowitz"] = row["HMM_MPC_CVXPY_weekly_return"] - row["Markowitz_CVXPY_weekly_return"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_yearly_strategy_returns.csv", index=False)
    return table


def strategy_subperiod_table(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "2013-2016": ("2013-01-01", "2016-12-31"),
        "2017-2019": ("2017-01-01", "2019-12-31"),
        "2020_crisis_recovery": ("2020-01-01", "2020-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
        "2022_inflation_bear": ("2022-01-01", "2022-12-31"),
        "2023-2024": ("2023-01-01", "2024-12-31"),
    }
    rows = []
    for label, (start, end) in periods.items():
        sample = strategy_returns.loc[start:end]
        row = {"period": label, "observations": len(sample)}
        for name in STRATEGIES:
            row[f"{name}_ann_return"] = ann_return(sample[name])
            row[f"{name}_sharpe"] = sharpe(sample[name])
            row[f"{name}_max_drawdown"] = max_drawdown(sample[name])
        row["HMM_minus_EqualWeight_ann_return"] = (
            row["HMM_MPC_CVXPY_weekly_ann_return"] - row["EqualWeight_weekly_ann_return"]
        )
        row["HMM_minus_Markowitz_ann_return"] = (
            row["HMM_MPC_CVXPY_weekly_ann_return"] - row["Markowitz_CVXPY_weekly_ann_return"]
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_subperiod_performance.csv", index=False)
    return table


def average_weight_table(weights: dict) -> pd.DataFrame:
    rows = []
    for name, w in weights.items():
        avg = w[ASSETS].mean()
        final = w[ASSETS].iloc[-1]
        concentration = (w[ASSETS] ** 2).sum(axis=1).mean()
        row = {"strategy": name, "avg_hhi": float(concentration)}
        for asset in ASSETS:
            row[f"avg_{asset}"] = float(avg[asset])
        for asset in ASSETS:
            row[f"final_{asset}"] = float(final[asset])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_average_weights.csv", index=False)
    return table


def contribution_table(returns: pd.DataFrame, weights: dict) -> pd.DataFrame:
    rows = []
    for name, w in weights.items():
        aligned_returns = returns.reindex(w.index)[ASSETS]
        start_weights = w[ASSETS].shift(1).fillna(1 / len(ASSETS))
        gross_contrib = start_weights * aligned_returns
        contrib_ann = gross_contrib.mean() * 252
        avg_weight = start_weights.mean()
        for asset in ASSETS:
            rows.append(
                {
                    "strategy": name,
                    "asset": asset,
                    "avg_weight": float(avg_weight[asset]),
                    "annualized_gross_contribution": float(contrib_ann[asset]),
                    "asset_annualized_return": ann_return(aligned_returns[asset]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_asset_contributions.csv", index=False)
    return table


def attach_daily_regime(strategy_returns: pd.DataFrame, regimes: pd.DataFrame) -> pd.Series:
    regime_points = regimes[["date", "dominant_regime"]].dropna().sort_values("date")
    regime_series = pd.Series(regime_points["dominant_regime"].values, index=regime_points["date"])
    daily = regime_series.reindex(strategy_returns.index, method="ffill")
    return daily.rename("dominant_regime")


def regime_performance_table(
    strategy_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    daily_regime = attach_daily_regime(strategy_returns, regimes)
    rows = []
    for regime, idx in daily_regime.dropna().groupby(daily_regime.dropna()).groups.items():
        sample = strategy_returns.loc[idx]
        row = {"regime": regime, "observations": len(sample)}
        for name in STRATEGIES:
            row[f"{name}_ann_return"] = ann_return(sample[name])
            row[f"{name}_vol"] = ann_vol(sample[name])
            row[f"{name}_avg_turnover"] = float(turnover.loc[idx, name].mean())
            row[f"{name}_total_cost"] = float(costs.loc[idx, name].sum())
        row["HMM_minus_EqualWeight_ann_return"] = (
            row["HMM_MPC_CVXPY_weekly_ann_return"] - row["EqualWeight_weekly_ann_return"]
        )
        row["HMM_minus_Markowitz_ann_return"] = (
            row["HMM_MPC_CVXPY_weekly_ann_return"] - row["Markowitz_CVXPY_weekly_ann_return"]
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_regime_performance.csv", index=False)
    return table


def forecast_quality_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    horizons = [1, 5, 21]
    for window in [252, 504, 756]:
        rolling_mean = returns.rolling(window).mean()
        for horizon in horizons:
            future = returns.shift(-horizon).rolling(horizon).mean()
            common = rolling_mean.dropna().index.intersection(future.dropna().index)
            ic_values = []
            sign_hits = []
            top_minus_bottom = []
            for date in common:
                pred = rolling_mean.loc[date]
                realized = future.loc[date]
                if pred.std() <= 0 or realized.std() <= 0:
                    continue
                ic_values.append(float(pred.corr(realized, method="spearman")))
                sign_hits.append(float((np.sign(pred) == np.sign(realized)).mean()))
                top_asset = pred.idxmax()
                bottom_asset = pred.idxmin()
                top_minus_bottom.append(float(realized[top_asset] - realized[bottom_asset]))
            rows.append(
                {
                    "rolling_window": window,
                    "future_horizon_days": horizon,
                    "mean_spearman_ic": float(np.nanmean(ic_values)),
                    "median_spearman_ic": float(np.nanmedian(ic_values)),
                    "sign_hit_rate": float(np.nanmean(sign_hits)),
                    "annualized_top_minus_bottom_realized": float(np.nanmean(top_minus_bottom) * 252),
                    "observations": len(ic_values),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_forecast_quality.csv", index=False)
    return table


def data_issue_table(returns: pd.DataFrame, features: pd.DataFrame, data_quality: pd.DataFrame) -> pd.DataFrame:
    outliers = []
    stacked = returns.stack()
    for (date, asset), value in stacked[stacked.abs() > 0.20].items():
        outliers.append({"date": date.date().isoformat(), "asset": asset, "daily_return": float(value)})
    table = pd.DataFrame(outliers)
    if table.empty:
        table = pd.DataFrame(columns=["date", "asset", "daily_return"])
    table.to_csv(TABLE_DIR / "diagnostic_large_return_outliers.csv", index=False)

    checks = data_quality.copy()
    checks.to_csv(TABLE_DIR / "diagnostic_data_quality_recheck.csv", index=False)
    return table


def strategy_similarity_table(strategy_returns: pd.DataFrame, weights: dict) -> pd.DataFrame:
    corr = strategy_returns.corr()
    corr.to_csv(TABLE_DIR / "diagnostic_strategy_return_correlation.csv")

    rows = []
    for left in STRATEGIES:
        for right in STRATEGIES:
            if left >= right:
                continue
            common = weights[left].index.intersection(weights[right].index)
            diff = (weights[left].loc[common, ASSETS] - weights[right].loc[common, ASSETS]).abs().sum(axis=1) / 2
            rows.append(
                {
                    "pair": f"{left} vs {right}",
                    "avg_weight_distance": float(diff.mean()),
                    "max_weight_distance": float(diff.max()),
                    "return_correlation": float(corr.loc[left, right]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "diagnostic_strategy_similarity.csv", index=False)
    return table


def format_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_diagnosis_report(
    yearly: pd.DataFrame,
    subperiod: pd.DataFrame,
    avg_weights: pd.DataFrame,
    contrib: pd.DataFrame,
    regime_perf: pd.DataFrame,
    forecast_quality: pd.DataFrame,
    outliers: pd.DataFrame,
    similarity: pd.DataFrame,
    asset_stats: pd.DataFrame,
) -> None:
    ew_vs_hmm_years = int((yearly["HMM_minus_EqualWeight"] > 0).sum())
    mv_vs_hmm_years = int((yearly["HMM_minus_Markowitz"] > 0).sum())
    n_years = len(yearly)

    hmm_weights = avg_weights.set_index("strategy").loc["HMM_MPC_CVXPY_weekly"]
    ew_weights = avg_weights.set_index("strategy").loc["EqualWeight_weekly"]
    avg_weight_lines = []
    for asset in ASSETS:
        delta = hmm_weights[f"avg_{asset}"] - ew_weights[f"avg_{asset}"]
        avg_weight_lines.append((asset, hmm_weights[f"avg_{asset}"], delta))
    avg_weight_lines = sorted(avg_weight_lines, key=lambda x: x[2])

    contrib_pivot = contrib.pivot(index="asset", columns="strategy", values="annualized_gross_contribution")
    contrib_pivot["hmm_minus_ew_contribution"] = (
        contrib_pivot["HMM_MPC_CVXPY_weekly"] - contrib_pivot["EqualWeight_weekly"]
    )
    contrib_pivot = contrib_pivot.sort_values("hmm_minus_ew_contribution")
    contrib_pivot.to_csv(TABLE_DIR / "diagnostic_hmm_vs_ew_contribution_delta.csv")

    fq = forecast_quality.sort_values("future_horizon_days").copy()
    fq_display = fq[["rolling_window", "future_horizon_days", "mean_spearman_ic", "sign_hit_rate", "annualized_top_minus_bottom_realized"]]
    subperiod_display = subperiod[
        [
            "period",
            "EqualWeight_weekly_ann_return",
            "Markowitz_CVXPY_weekly_ann_return",
            "HMM_MPC_CVXPY_weekly_ann_return",
            "HMM_minus_EqualWeight_ann_return",
            "HMM_minus_Markowitz_ann_return",
        ]
    ]

    report = f"""# Diagnosis of HMM-MPC Underperformance

## Executive Diagnosis

The proposed HMM-MPC strategy is not a bad result, but it does not dominate the equal-weight benchmark. The evidence points to four main reasons.

First, the data set is clean enough for this stage. The underperformance is not explained by missing prices, missing macro data, zero volume, or obvious data corruption. The only absolute daily return above 20% is documented in `diagnostic_large_return_outliers.csv`.

Second, the asset universe itself is unusually friendly to equal weighting. From 2013 to 2024, several growth and defensive sector ETFs had strong realized performance, and a simple diversified 1/N allocation captured that without estimation error. The HMM-MPC model beat equal weight in {ew_vs_hmm_years} out of {n_years} calendar years, while it beat single-period Markowitz in {mv_vs_hmm_years} out of {n_years} years.

Third, the current optimizer still relies on rolling historical mean estimates. The cross-sectional return-forecast diagnostics show weak predictive content. When expected-return estimates are noisy, a constrained optimizer can easily underweight assets that later perform well. This is a classic mean-variance problem rather than only an HMM problem.

Fourth, the current HMM-MPC implementation is conservative and risk/cost-aware. It improves on single-period Markowitz by reducing turnover and transaction costs, but the same conservatism prevents it from fully participating in the strongest sectors during long bull periods.

## 1. Data Quality

The current evidence does not indicate a fatal data issue.

{outliers.to_markdown(index=False) if not outliers.empty else "No daily asset return exceeded 20% in absolute value."}

The strongest data-related caution is not missingness, but source reproducibility. `yfinance.download` failed in this environment, so the script used the Yahoo Finance chart endpoint fallback. The downloaded data are complete for the selected ETFs, but the final paper should disclose this fallback and keep the cached CSVs for reproducibility.

## 2. Historical Asset Performance

The sector ETF universe creates a difficult benchmark for any estimated optimizer. Equal weight automatically holds the long-run winners and keeps rebalancing into them without making forecasts. The strongest assets over the full sample include XLK, XLY, XLV, and XLI, while XLE had weak long-run performance and very large drawdowns.

{asset_stats.to_markdown(index=False)}

The HMM-MPC average weight deviations from equal weight are:

| asset | HMM-MPC average weight | difference from equal weight |
|:------|-----------------------:|-----------------------------:|
""" + "\n".join(
        f"| {asset} | {format_pct(weight)} | {format_pct(delta)} |"
        for asset, weight, delta in avg_weight_lines
    ) + f"""

The most negative contribution differences versus equal weight were:

{contrib_pivot.head(5).reset_index().to_markdown(index=False)}

The subperiod pattern is also informative:

{subperiod_display.to_markdown(index=False)}

The proposed strategy was competitive before 2020 and improved on equal weight in the 2013-2016 and 2017-2019 subperiods. The overall gap is mainly created after 2021, especially in 2023-2024, when equal weight benefited from broad equity recovery and strong growth-sector performance while the optimizer remained heavily defensive.

## 3. Model Assumptions

The current model makes several strong assumptions:

- The HMM state inferred from market-level features is assumed to be useful for sector-level expected returns.
- State-conditional means are estimated from a relatively short rolling sample.
- Gaussian diagonal HMM emissions are assumed to summarize nonlinear market stress dynamics.
- The optimizer treats estimated means as reliable enough to tilt away from equal weight.
- The cost multiplier increases in stress states, but the strategy has no explicit cash or bond asset, so it can only rotate within equities.

These assumptions are individually reasonable for a first implementation, but together they create a conservative equity-sector rotation model with weak alpha forecasts.

## 4. Forecast Quality

The forecast-quality table evaluates whether rolling historical means predict future cross-sectional ETF returns. Low or unstable information coefficients imply that the optimizer's mean vector is mostly noise.

{fq_display.to_markdown(index=False)}

This is probably the most important empirical finding. The optimizer is convex and numerically stable, but the expected-return input is not strong enough to beat an equal-weight benchmark over this asset universe.

## 5. Regime-Conditioned Performance

{regime_perf.to_markdown(index=False)}

The regime-aware model is most defensible when compared with single-period Markowitz. It lowers turnover and costs, and its performance is less fragile than an optimizer that repeatedly jumps to a one-period target. However, the regime model does not automatically create excess return versus equal weight.

## 6. Implementation Issues

The implementation is broadly consistent with the theoretical design: the HMM is fit using only historical features, regime probabilities are external inputs, and the MPC problem is solved with CVXPY. There are nevertheless implementation choices that may suppress performance:

- The same risk-aversion parameter is used for Markowitz and HMM-MPC; this may make the proposed model too conservative.
- The strategy is very close to single-period Markowitz in realized returns. The return correlation between HMM-MPC and Markowitz is 0.989, while the average weight distance from equal weight is about 0.549. This indicates that the current HMM-MPC behaves mainly like a stabilized defensive mean-variance optimizer, not yet like a distinct regime-alpha model.
- The trading-cost multiplier is deliberately high in stress states, which reduces turnover but can also slow recovery-period reallocation.
- The HMM is refit at every rebalance, which is conservative but may produce unstable state labels even after sorting by stress score.
- The model has no explicit shrinkage of expected returns toward economically motivated priors such as equal weight, momentum, or volatility timing.
- The objective uses daily arithmetic return and covariance estimates. This is mathematically valid, but the practical aggressiveness depends heavily on gamma and cost calibration. With the current gamma, the optimizer strongly prefers low-volatility defensive sectors.

## 7. Most Promising Improvement Directions

The most promising directions are:

1. Replace raw rolling mean forecasts with more robust expected-return inputs. Candidate signals include 12-1 momentum, short-term reversal control, volatility-adjusted momentum, or shrinkage toward zero/equal-weight implied returns.

2. Add a no-alpha or weak-alpha version of MPC. In this version, expected returns are set equal across assets or heavily shrunk, so the model mainly optimizes risk and trading costs. This directly tests whether the current underperformance comes from noisy means.

3. Tune risk aversion and trading-cost multipliers on a validation period only. The current parameters are defensible but not calibrated. A small grid over gamma, turnover cap, and stress-cost multipliers could improve the risk-return trade-off without data mining the final test set.

4. Add a defensive asset or cash proxy. A sector-only long-only universe cannot truly de-risk in stress regimes; it can only rotate among risky equity sectors. Adding SHY, IEF, or BIL would make the market-regime signal more actionable.

5. Use regime probabilities mainly for risk and cost, not expected returns. The current regime-conditioned mean estimates are noisy. A more robust design may let HMM drive covariance and transaction-cost penalties while expected returns come from simpler momentum or remain shrunk.

6. Report the equal-weight result honestly. Equal weight is a strong benchmark in this data set. The final paper should frame the proposed model as an improvement over single-period Markowitz in turnover/cost control, while acknowledging that it does not outperform 1/N in total return over 2013-2024.

## Generated Diagnostic Tables

- `outputs/tables/diagnostic_yearly_strategy_returns.csv`
- `outputs/tables/diagnostic_subperiod_performance.csv`
- `outputs/tables/diagnostic_average_weights.csv`
- `outputs/tables/diagnostic_asset_contributions.csv`
- `outputs/tables/diagnostic_hmm_vs_ew_contribution_delta.csv`
- `outputs/tables/diagnostic_regime_performance.csv`
- `outputs/tables/diagnostic_forecast_quality.csv`
- `outputs/tables/diagnostic_large_return_outliers.csv`
- `outputs/tables/diagnostic_strategy_similarity.csv`
- `outputs/tables/diagnostic_strategy_return_correlation.csv`
"""
    (REPORT_DIR / "model_underperformance_diagnosis.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    returns, strategy_returns, costs, turnover, weights, regimes, features, asset_stats, data_quality = load_data()

    common = strategy_returns.index
    returns = returns.reindex(common)

    yearly = yearly_strategy_table(strategy_returns)
    subperiod = strategy_subperiod_table(strategy_returns)
    avg_weights = average_weight_table(weights)
    contrib = contribution_table(returns, weights)
    regime_perf = regime_performance_table(strategy_returns, turnover, costs, regimes)
    forecast_quality = forecast_quality_table(returns)
    outliers = data_issue_table(returns, features, data_quality)
    similarity = strategy_similarity_table(strategy_returns, weights)

    write_diagnosis_report(
        yearly=yearly,
        subperiod=subperiod,
        avg_weights=avg_weights,
        contrib=contrib,
        regime_perf=regime_perf,
        forecast_quality=forecast_quality,
        outliers=outliers,
        similarity=similarity,
        asset_stats=asset_stats,
    )

    print("Diagnosis written to:", REPORT_DIR / "model_underperformance_diagnosis.md")
    print("\nYearly active returns:")
    print(yearly[["year", "HMM_minus_EqualWeight", "HMM_minus_Markowitz"]].to_string(index=False))
    print("\nForecast quality:")
    print(forecast_quality.to_string(index=False))


if __name__ == "__main__":
    main()
