# V3 Rolling Regime-MPC Experiment

## 改进目标

V3 是对 V2 的直接修正：不再用 `2013-2018` 固定验证集选出一组永久参数，而是从 `2013-01-08` 开始完整滚动运行。每个调仓点只用过去已经实现的候选策略收益来选择当前参数，从而避免把某一段历史的市场风格锁死到未来。

## 核心变化

- 起始时间仍为 `2013-01-08`，与主实证和 baseline 口径一致。
- 参数网格保持小规模，共 `12` 个候选，避免滚动选择过慢。
- 每个候选都是一条实时 paper-trading 策略，候选本身只使用当时可见的历史收益和 HMM regime probability。
- V3 每次调仓使用过去 `756` 个交易日的候选表现打分；早期不足 `252` 个交易日时，使用强 momentum 默认候选。
- 打分函数不仅看 Sharpe，也加入相对等权的 active return 和 information ratio，避免重新选出过度防御的参数。
- alpha 不再是固定全局 shrink，而是 regime-conditioned：calm 时保留更多 12-1 momentum，stress 时自动降低 alpha 强度。

## 候选参数


| 参数                  | 候选值               |
| ------------------- | ----------------- |
| `gamma`             | `5.0, 8.0`        |
| `base_alpha`        | `0.25, 0.75, 1.0` |
| `turnover_cap`      | `0.10, 0.25`      |
| `stress_multiplier` | `1.0`             |
| MPC horizon         | `3`               |
| 单资产权重上限             | `60%`             |


## 无未来数据泄漏控制

- 候选策略收益是按时间顺序逐日生成的 paper-trading 结果。
- 在调仓日 `t`，V3 只使用 `t` 之前已经实现的候选收益打分。
- 调仓权重只使用 `t` 及以前的收益历史和 HMM 下一期概率，并在下一交易日生效。
- 所有策略都从 2013 年开始运行；测试段只是对 2019-2024 收益进行切片评估。

## 全样本结果


| strategy               | start      | end        | observations | annualized_return | annualized_volatility | sharpe | max_drawdown | final_wealth | avg_daily_turnover | avg_rebalance_turnover | total_cost |
| ---------------------- | ---------- | ---------- | ------------ | ----------------- | --------------------- | ------ | ------------ | ------------ | ------------------ | ---------------------- | ---------- |
| EqualWeight_weekly     | 2013-01-08 | 2024-12-31 | 3016         | 0.1265            | 0.1624                | 0.779  | -0.3698      | 4.159        | 0.0012             | 0.0058                 | 0.0036     |
| Markowitz_CVXPY_weekly | 2013-01-08 | 2024-12-31 | 3016         | 0.0987            | 0.1454                | 0.679  | -0.3251      | 3.0862       | 0.0084             | 0.0406                 | 0.0254     |
| HMM_MPC_CVXPY_weekly   | 2013-01-08 | 2024-12-31 | 3016         | 0.107             | 0.1498                | 0.7139 | -0.3265      | 3.374        | 0.0038             | 0.0182                 | 0.0174     |
| V3_Rolling_Regime_MPC  | 2013-01-08 | 2024-12-31 | 3016         | 0.144             | 0.1643                | 0.8763 | -0.3345      | 5.0015       | 0.0028             | 0.0136                 | 0.0104     |


全样本超额表现：


| comparison                                         | annualized_active_return | tracking_error | information_ratio | annualized_return_difference | sharpe_difference | max_drawdown_improvement | final_wealth_ratio | avg_rebalance_turnover_difference | total_cost_difference |
| -------------------------------------------------- | ------------------------ | -------------- | ----------------- | ---------------------------- | ----------------- | ------------------------ | ------------------ | --------------------------------- | --------------------- |
| V3_Rolling_Regime_MPC minus EqualWeight_weekly     | 0.0157                   | 0.0697         | 0.2258            | 0.0175                       | 0.0973            | 0.0353                   | 1.2026             | 0.0078                            | 0.0067                |
| V3_Rolling_Regime_MPC minus Markowitz_CVXPY_weekly | 0.0433                   | 0.0718         | 0.6031            | 0.0452                       | 0.1973            | -0.0094                  | 1.6206             | -0.0271                           | -0.0151               |
| V3_Rolling_Regime_MPC minus HMM_MPC_CVXPY_weekly   | 0.0352                   | 0.0683         | 0.5153            | 0.037                        | 0.1624            | -0.0079                  | 1.4824             | -0.0047                           | -0.0071               |


## 2019-2024 测试切片


| strategy               | start      | end        | observations | annualized_return | annualized_volatility | sharpe | max_drawdown | final_wealth | avg_daily_turnover | avg_rebalance_turnover | total_cost |
| ---------------------- | ---------- | ---------- | ------------ | ----------------- | --------------------- | ------ | ------------ | ------------ | ------------------ | ---------------------- | ---------- |
| EqualWeight_weekly     | 2019-01-02 | 2024-12-31 | 1510         | 0.1477            | 0.1939                | 0.7615 | -0.3698      | 2.2828       | 0.0014             | 0.0069                 | 0.0022     |
| Markowitz_CVXPY_weekly | 2019-01-02 | 2024-12-31 | 1510         | 0.0954            | 0.1695                | 0.5629 | -0.3251      | 1.7263       | 0.0067             | 0.0322                 | 0.0101     |
| HMM_MPC_CVXPY_weekly   | 2019-01-02 | 2024-12-31 | 1510         | 0.0993            | 0.1755                | 0.5655 | -0.3265      | 1.7632       | 0.0035             | 0.0168                 | 0.0084     |
| V3_Rolling_Regime_MPC  | 2019-01-02 | 2024-12-31 | 1510         | 0.1618            | 0.1924                | 0.8408 | -0.3345      | 2.4556       | 0.0038             | 0.0185                 | 0.0071     |


测试切片超额表现：


| comparison                                         | annualized_active_return | tracking_error | information_ratio | annualized_return_difference | sharpe_difference | max_drawdown_improvement | final_wealth_ratio | avg_rebalance_turnover_difference | total_cost_difference |
| -------------------------------------------------- | ------------------------ | -------------- | ----------------- | ---------------------------- | ----------------- | ------------------------ | ------------------ | --------------------------------- | --------------------- |
| V3_Rolling_Regime_MPC minus EqualWeight_weekly     | 0.0119                   | 0.0846         | 0.1403            | 0.0141                       | 0.0794            | 0.0353                   | 1.0757             | 0.0116                            | 0.005                 |
| V3_Rolling_Regime_MPC minus Markowitz_CVXPY_weekly | 0.063                    | 0.0923         | 0.6824            | 0.0664                       | 0.278             | -0.0094                  | 1.4225             | -0.0137                           | -0.003                |
| V3_Rolling_Regime_MPC minus HMM_MPC_CVXPY_weekly   | 0.0585                   | 0.0885         | 0.6603            | 0.0625                       | 0.2753            | -0.0079                  | 1.3928             | 0.0017                            | -0.0012               |


## 参数选择诊断

V3 全样本年化收益为 `14.40%`，Sharpe 为 `0.8763`，最大回撤为 `-33.45%`。相较 V2，V3 的关键改进是允许 strong momentum 候选重新进入模型，同时让参数随近期历史表现滚动变化。

选择次数最多的候选：


| selected_candidate | gamma | base_alpha | turnover_cap | selection_count |
| ------------------ | ----- | ---------- | ------------ | --------------- |
| g5_a100_cap25_s1   | 5     | 1          | 0.25         | 183             |
| g8_a100_cap25_s1   | 8     | 1          | 0.25         | 104             |
| g5_a025_cap10_s1   | 5     | 0.25       | 0.1          | 98              |
| g5_a025_cap25_s1   | 5     | 0.25       | 0.25         | 81              |
| g8_a025_cap25_s1   | 8     | 0.25       | 0.25         | 74              |
| g5_a075_cap10_s1   | 5     | 0.75       | 0.1          | 40              |
| g5_a100_cap10_s1   | 5     | 1          | 0.1          | 17              |
| g8_a075_cap25_s1   | 8     | 0.75       | 0.25         | 17              |
| g8_a025_cap10_s1   | 8     | 0.25       | 0.1          | 12              |


候选自身全样本 paper-trading 表现：


| candidate        | start      | end        | observations | annualized_return | annualized_volatility | sharpe | max_drawdown | final_wealth | avg_rebalance_turnover | total_cost |
| ---------------- | ---------- | ---------- | ------------ | ----------------- | --------------------- | ------ | ------------ | ------------ | ---------------------- | ---------- |
| g8_a100_cap25_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1289            | 0.17                  | 0.7581 | -0.3286      | 4.2679       | 0.0252                 | 0.0188     |
| g8_a075_cap25_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1258            | 0.1663                | 0.7566 | -0.3309      | 4.1304       | 0.0199                 | 0.0148     |
| g8_a075_cap10_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1247            | 0.167                 | 0.7469 | -0.3299      | 4.0815       | 0.0192                 | 0.0144     |
| g5_a025_cap25_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1111            | 0.1491                | 0.7452 | -0.3307      | 3.5287       | 0.0032                 | 0.0023     |
| g5_a025_cap10_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1111            | 0.1491                | 0.7449 | -0.3307      | 3.527        | 0.0032                 | 0.0023     |
| g8_a100_cap10_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.126             | 0.1703                | 0.7401 | -0.3301      | 4.1395       | 0.0237                 | 0.0178     |
| g8_a025_cap25_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1064            | 0.1441                | 0.7389 | -0.3352      | 3.3556       | 0.0031                 | 0.0026     |
| g5_a075_cap10_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.13              | 0.1768                | 0.7355 | -0.331       | 4.3182       | 0.0196                 | 0.0143     |
| g8_a025_cap10_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.106             | 0.1442                | 0.7354 | -0.3353      | 3.3397       | 0.0031                 | 0.0026     |
| g5_a075_cap25_s1 | 2013-01-08 | 2024-12-31 | 3016         | 0.1283            | 0.1767                | 0.7263 | -0.3305      | 4.2417       | 0.021                  | 0.0152     |


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

