# V2 Combined HMM-MPC Experiment

## 改进目标

V2 把前三个可行方向合并为一个新的实证版本：方向 1 的稳健 12-1 截面动量预期收益、方向 2 的 weak-alpha 缩放、方向 3 的验证集调参。该文件夹是独立实验，不修改项目原始 baseline 或原始 HMM-MPC 代码。

## 模型设定

- 资产池：`XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY`。
- 预测信号：每次调仓在历史 `252` 个交易日内计算 12-1 动量，即跳过最近 `21` 个交易日，只使用更早窗口的收益信息。
- weak-alpha：对截面动量信号做去均值和缩放后乘以 `alpha_shrink`，避免把噪声均值当成强 alpha。
- 风险估计：滚动 `252` 日协方差，使用 `15%` 对角收缩，并根据 HMM 下一期 stress/transition 概率做温和风险放大。
- 优化器：CVXPY 多期凸优化，MPC horizon 为 `5` 个交易日，约束为 long-only、full investment、单资产权重上限 `45%`、每次调仓 turnover cap。
- 交易成本：基础成本为 `0.05%` 每美元交易额，并随 HMM stress 概率由 `stress_multiplier` 放大。

## 防止未来数据泄漏

- 验证集：`2013-01-08` 至 `2018-12-31`。
- 测试集：`2019-01-01` 至 `2024-12-31`。
- 参数只按照验证集 Sharpe 选择；测试集结果使用冻结参数。
- 每个测试日的组合权重只由前一交易日及以前的数据决定。调仓决策日为 `decision_date`，实际生效日为下一交易日 `effective_date`。
- HMM regime records 使用项目原始 walk-forward 输出；V2 只读取相应调仓日已经生成的下一期概率，不重新用测试期未来信息拟合参数。

## 选中的 V2 参数

|   gamma |   alpha_shrink |   turnover_cap |   stress_multiplier |
|--------:|---------------:|---------------:|--------------------:|
|       8 |           0.25 |            0.1 |                   1 |

验证集选中参数的表现：

| strategy   | start      | end        |   observations |   annualized_return |   annualized_volatility |   sharpe |   max_drawdown |   final_wealth |   avg_daily_turnover |   avg_rebalance_turnover |   total_cost |   gamma |   alpha_shrink |   turnover_cap |   stress_multiplier |
|:-----------|:-----------|:-----------|---------------:|--------------------:|------------------------:|---------:|---------------:|---------------:|---------------------:|-------------------------:|-------------:|--------:|---------------:|---------------:|--------------------:|
| V2_HMM_MPC | 2013-01-08 | 2018-12-31 |           1506 |              0.1116 |                  0.1164 |   0.9593 |        -0.1572 |         1.8822 |                0.001 |                   0.0049 |       0.0018 |       8 |           0.25 |            0.1 |                   1 |

## 测试集表现

| strategy               | start      | end        |   observations |   annualized_return |   annualized_volatility |   sharpe |   max_drawdown |   final_wealth |   avg_daily_turnover |   avg_rebalance_turnover |   total_cost |
|:-----------------------|:-----------|:-----------|---------------:|--------------------:|------------------------:|---------:|---------------:|---------------:|---------------------:|-------------------------:|-------------:|
| EqualWeight_weekly     | 2019-01-02 | 2024-12-31 |           1510 |              0.1477 |                  0.1939 |   0.7615 |        -0.3698 |         2.2828 |               0.0014 |                   0.0069 |       0.0022 |
| Markowitz_CVXPY_weekly | 2019-01-02 | 2024-12-31 |           1510 |              0.0954 |                  0.1695 |   0.5629 |        -0.3251 |         1.7263 |               0.0067 |                   0.0322 |       0.0101 |
| HMM_MPC_CVXPY_weekly   | 2019-01-02 | 2024-12-31 |           1510 |              0.0993 |                  0.1755 |   0.5655 |        -0.3265 |         1.7632 |               0.0035 |                   0.0168 |       0.0084 |
| V2_HMM_MPC             | 2019-01-02 | 2024-12-31 |           1510 |              0.1132 |                  0.1707 |   0.663  |        -0.3262 |         1.9012 |               0.002  |                   0.0099 |       0.0043 |

测试集相对 baseline 和原始模型的超额表现：

| comparison                              |   annualized_active_return |   tracking_error |   information_ratio |   annualized_return_difference |   sharpe_difference |   max_drawdown_improvement |   final_wealth_ratio |   avg_rebalance_turnover_difference |   total_cost_difference |
|:----------------------------------------|---------------------------:|-----------------:|--------------------:|-------------------------------:|--------------------:|---------------------------:|---------------------:|------------------------------------:|------------------------:|
| V2_HMM_MPC minus EqualWeight_weekly     |                    -0.0348 |           0.0736 |             -0.4734 |                        -0.0345 |             -0.0985 |                     0.0436 |               0.8328 |                              0.003  |                  0.0021 |
| V2_HMM_MPC minus Markowitz_CVXPY_weekly |                     0.0163 |           0.044  |              0.3704 |                         0.0178 |              0.1001 |                    -0.0011 |               1.1013 |                             -0.0223 |                 -0.0058 |
| V2_HMM_MPC minus HMM_MPC_CVXPY_weekly   |                     0.0118 |           0.0452 |              0.2601 |                         0.0139 |              0.0975 |                     0.0004 |               1.0783 |                             -0.007  |                 -0.0041 |

## 全样本描述性结果

全样本结果用于和前面实验保持口径一致，但不用于选参。

| strategy               | start      | end        |   observations |   annualized_return |   annualized_volatility |   sharpe |   max_drawdown |   final_wealth |   avg_daily_turnover |   avg_rebalance_turnover |   total_cost |
|:-----------------------|:-----------|:-----------|---------------:|--------------------:|------------------------:|---------:|---------------:|---------------:|---------------------:|-------------------------:|-------------:|
| EqualWeight_weekly     | 2013-01-08 | 2024-12-31 |           3016 |              0.1265 |                  0.1624 |   0.779  |        -0.3698 |         4.159  |               0.0012 |                   0.0058 |       0.0036 |
| Markowitz_CVXPY_weekly | 2013-01-08 | 2024-12-31 |           3016 |              0.0987 |                  0.1454 |   0.679  |        -0.3251 |         3.0862 |               0.0084 |                   0.0406 |       0.0254 |
| HMM_MPC_CVXPY_weekly   | 2013-01-08 | 2024-12-31 |           3016 |              0.107  |                  0.1498 |   0.7139 |        -0.3265 |         3.374  |               0.0038 |                   0.0182 |       0.0174 |
| V2_HMM_MPC             | 2013-01-08 | 2024-12-31 |           3016 |              0.1112 |                  0.1461 |   0.7613 |        -0.3278 |         3.5326 |               0.0014 |                   0.0069 |       0.0056 |

全样本超额表现：

| comparison                              |   annualized_active_return |   tracking_error |   information_ratio |   annualized_return_difference |   sharpe_difference |   max_drawdown_improvement |   final_wealth_ratio |   avg_rebalance_turnover_difference |   total_cost_difference |
|:----------------------------------------|---------------------------:|-----------------:|--------------------:|-------------------------------:|--------------------:|---------------------------:|---------------------:|------------------------------------:|------------------------:|
| V2_HMM_MPC minus EqualWeight_weekly     |                    -0.0162 |           0.0608 |             -0.266  |                        -0.0153 |             -0.0177 |                     0.042  |               0.8494 |                              0.0012 |                  0.002  |
| V2_HMM_MPC minus Markowitz_CVXPY_weekly |                     0.0114 |           0.037  |              0.3076 |                         0.0125 |              0.0824 |                    -0.0027 |               1.1447 |                             -0.0337 |                 -0.0199 |
| V2_HMM_MPC minus HMM_MPC_CVXPY_weekly   |                     0.0033 |           0.0381 |              0.0863 |                         0.0043 |              0.0475 |                    -0.0013 |               1.047  |                             -0.0113 |                 -0.0118 |

## 结果解读

V2 测试集年化收益为 `11.32%`，Sharpe 为 `0.6630`，最大回撤为 `-32.62%`。与原始 HMM-MPC 相比，V2 的核心变化不是提高模型复杂度，而是降低均值估计噪声：使用更稳的 12-1 截面动量替代滚动均值，再通过 weak-alpha 缩放控制信号强度，并让风险厌恶、换手约束、stress 成本放大都由验证集决定。

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

| start      | end        |   observations |   annualized_return |   annualized_volatility |   sharpe |   max_drawdown |   final_wealth |   avg_daily_turnover |   avg_rebalance_turnover |   total_cost |   gamma |   alpha_shrink |   turnover_cap |   stress_multiplier | strategy   |
|:-----------|:-----------|---------------:|--------------------:|------------------------:|---------:|---------------:|---------------:|---------------------:|-------------------------:|-------------:|--------:|---------------:|---------------:|--------------------:|:-----------|
| 2013-01-08 | 2018-12-31 |           1506 |              0.1116 |                  0.1164 |   0.9593 |        -0.1572 |         1.8822 |               0.001  |                   0.0049 |       0.0018 |       8 |           0.25 |           0.1  |                   1 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1119 |                  0.1167 |   0.9591 |        -0.1598 |         1.885  |               0.0009 |                   0.0046 |       0.0015 |       8 |           0.25 |           0.1  |                   3 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1104 |                  0.1161 |   0.9512 |        -0.1593 |         1.8697 |               0.0009 |                   0.0043 |       0.0014 |       8 |           0.25 |           0.05 |                   3 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1101 |                  0.1159 |   0.9508 |        -0.1569 |         1.8673 |               0.001  |                   0.0047 |       0.0018 |       8 |           0.25 |           0.05 |                   1 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1102 |                  0.1199 |   0.9189 |        -0.1791 |         1.8674 |               0.001  |                   0.0047 |       0.0015 |       5 |           0.25 |           0.1  |                   3 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1087 |                  0.1185 |   0.9175 |        -0.1721 |         1.8529 |               0.0009 |                   0.0044 |       0.0014 |       5 |           0.25 |           0.05 |                   3 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1091 |                  0.1197 |   0.9113 |        -0.1784 |         1.8567 |               0.0011 |                   0.0052 |       0.0019 |       5 |           0.25 |           0.1  |                   1 | V2_HMM_MPC |
| 2013-01-08 | 2018-12-31 |           1506 |              0.1078 |                  0.1184 |   0.9108 |        -0.1719 |         1.8441 |               0.001  |                   0.0048 |       0.0017 |       5 |           0.25 |           0.05 |                   1 | V2_HMM_MPC |
