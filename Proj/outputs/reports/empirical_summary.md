# Empirical Strategy Report

## Framework

- Python environment: `/opt/anaconda3/envs/pytorch_py=3.8/bin/python`
- Price data source used: `Yahoo Finance chart endpoint fallback`
- Macro data source used: `FRED CSV, VIXCLS and BAMLH0A0HYM2`
- Asset universe: `XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY`
- Data window: `2010-01-01` to `2024-12-31`
- Training scheme: rolling `756` trading-day estimation window
- Rebalance frequency: `W-FRI`
- Transaction cost assumption: `5.0` basis points per dollar traded
- Markowitz solver: CVXPY with OSQP first and CLARABEL/ECOS fallback
- Markowitz constraints: long-only, full investment, max weight `0.40`
- Markowitz risk aversion: `20.0`
- HMM states: `3`, sorted as calm, transition, stress
- HMM estimation: rolling `756` trading-day feature window, diagonal Gaussian emissions
- MPC horizon: `5` trading days
- MPC turnover cap per rebalance: `0.35`
- Regime cost multipliers: calm `1.0`, transition `1.5`, stress `3.0`

The empirical design first establishes clean baseline strategies and then evaluates a proposed HMM-MPC strategy. The HMM is estimated only from information available through the rebalance date. Its regime probabilities determine conditional return, covariance, and trading-cost inputs. The optimizer remains a convex multi-period CVXPY problem.

## Data Quality Summary


| check                           | value              |
| ------------------------------- | ------------------ |
| asset_count                     | 9                  |
| raw_price_start                 | 2010-01-04         |
| raw_price_end                   | 2024-12-31         |
| clean_return_start              | 2010-01-05         |
| clean_return_end                | 2024-12-31         |
| return_observations             | 3773               |
| raw_adj_close_missing_cells     | 0                  |
| raw_close_missing_cells         | 0                  |
| raw_volume_missing_cells        | 0                  |
| zero_volume_cells_clean_sample  | 0                  |
| abs_daily_return_gt_20pct_cells | 1                  |
| macro_missing_after_alignment   | 0                  |
| feature_start                   | 2010-04-07         |
| feature_end                     | 2024-12-31         |
| feature_observations            | 3710               |
| rolling_cov_condition_median    | 54.77676803444878  |
| rolling_cov_condition_max       | 106.06411258840812 |


## Asset Return Summary


| asset | annualized_return | annualized_volatility | sharpe | max_drawdown | daily_skew | daily_kurtosis | min_daily_return | max_daily_return |
| ----- | ----------------- | --------------------- | ------ | ------------ | ---------- | -------------- | ---------------- | ---------------- |
| XLB   | 0.0855            | 0.2086                | 0.41   | -0.3727      | -0.2779    | 6.8454         | -0.1101          | 0.1176           |
| XLE   | 0.0598            | 0.2744                | 0.2178 | -0.7126      | -0.3835    | 13.0069        | -0.2014          | 0.1604           |
| XLF   | 0.1179            | 0.2214                | 0.5324 | -0.4286      | -0.2233    | 11.3896        | -0.1371          | 0.1316           |
| XLI   | 0.1295            | 0.1945                | 0.6661 | -0.4233      | -0.3228    | 10.4004        | -0.1134          | 0.1265           |
| XLK   | 0.1825            | 0.2129                | 0.8573 | -0.3356      | -0.2217    | 8.772          | -0.1381          | 0.1173           |
| XLP   | 0.1042            | 0.1371                | 0.7603 | -0.2451      | -0.2978    | 13.8339        | -0.094           | 0.0851           |
| XLU   | 0.099             | 0.1759                | 0.5631 | -0.3607      | -0.002     | 15.796         | -0.1136          | 0.1279           |
| XLV   | 0.1221            | 0.1609                | 0.7588 | -0.284       | -0.2692    | 7.7642         | -0.0986          | 0.0771           |
| XLY   | 0.1582            | 0.2006                | 0.7889 | -0.3967      | -0.5536    | 7.4365         | -0.1267          | 0.0938           |


## Strategy Performance


| strategy               | start      | end        | observations | annualized_return | annualized_volatility | sharpe | max_drawdown | final_wealth | avg_daily_turnover | avg_rebalance_turnover | total_cost |
| ---------------------- | ---------- | ---------- | ------------ | ----------------- | --------------------- | ------ | ------------ | ------------ | ------------------ | ---------------------- | ---------- |
| EqualWeight_weekly     | 2013-01-08 | 2024-12-31 | 3016         | 0.1265            | 0.1624                | 0.779  | -0.3698      | 4.159        | 0.0012             | 0.0058                 | 0.0036     |
| Markowitz_CVXPY_weekly | 2013-01-08 | 2024-12-31 | 3016         | 0.0987            | 0.1454                | 0.679  | -0.3251      | 3.0862       | 0.0084             | 0.0406                 | 0.0254     |
| HMM_MPC_CVXPY_weekly   | 2013-01-08 | 2024-12-31 | 3016         | 0.107             | 0.1498                | 0.7139 | -0.3265      | 3.374        | 0.0038             | 0.0182                 | 0.0174     |


## Excess Performance of HMM-MPC


| comparison                                        | annualized_active_return | tracking_error | information_ratio | annualized_return_difference | sharpe_difference | max_drawdown_improvement | final_wealth_ratio | avg_rebalance_turnover_difference | total_cost_difference |
| ------------------------------------------------- | ------------------------ | -------------- | ----------------- | ---------------------------- | ----------------- | ------------------------ | ------------------ | --------------------------------- | --------------------- |
| HMM_MPC_CVXPY_weekly minus EqualWeight_weekly     | -0.0195                  | 0.0638         | -0.3051           | -0.0195                      | -0.0651           | 0.0433                   | 0.8112             | 0.0125                            | 0.0138                |
| HMM_MPC_CVXPY_weekly minus Markowitz_CVXPY_weekly | 0.0081                   | 0.0224         | 0.3607            | 0.0082                       | 0.0349            | -0.0015                  | 1.0933             | -0.0224                           | -0.008                |


## HMM Regime Diagnostics


| metric                     | value  |
| -------------------------- | ------ |
| rebalance_observations     | 626    |
| hmm_mpc_successes          | 626    |
| fallbacks                  | 0      |
| avg_prob_calm              | 0.5236 |
| avg_prob_transition        | 0.3075 |
| avg_prob_stress            | 0.1689 |
| max_prob_stress            | 1      |
| stress_prob_gt_50pct_count | 105    |
| avg_next_cost_multiplier   | 1.4906 |
| max_next_cost_multiplier   | 3      |
| dominant_calm_count        | 329    |
| dominant_transition_count  | 192    |
| dominant_stress_count      | 105    |


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

