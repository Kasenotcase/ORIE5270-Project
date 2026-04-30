# 改进方向 1：更稳健的收益预测输入

## 改进方向

原始版本在 HMM-MPC 中直接使用 `252` 日 rolling mean 作为期望收益输入。这个做法的问题在于：

1. 均值估计对短期噪声很敏感；
2. 行业 ETF 的横截面收益经常带有明显的趋势和反转结构，单纯 rolling mean 容易把近期噪声当成信号；
3. 在长仓、全投资、带交易成本的优化里，`mu` 的微小偏差会放大成不必要的换手和错误配置。

所以这个改进方向的目标是：**不用原始 rolling mean，改成更稳健的 return inputs，再送进同一个 CVXPY 优化器里**。

## 实现方法

本实验没有修改项目主代码，只在本子目录里做独立实验，读取已有缓存数据：

- `data/processed/returns.csv`
- `outputs/tables/hmm_regime_records.csv`
- `outputs/tables/strategy_performance.csv`

我实现了 4 个期望收益输入版本：

1. `raw_mean_252`
2. `momentum_12_1_scaled`
3. `vol_adj_momentum_scaled`
4. `shrink_blend`

其中最关键的是 `momentum_12_1_scaled`：

- 使用 `252` 日 lookback
- 跳过最近 `21` 个交易日，避免短期反转噪声
- 用过去窗口的平均对数收益作为 momentum signal
- 再按横截面标准化到和 raw mean 相近的尺度

优化器保持原来那套 convex 结构：

```text
max_w  mu_t^T w - gamma * w^T Sigma_t w - tc * turnover
```

约束：

- `sum(w) = 1`
- `w >= 0`

其中：

- `gamma = 5.0`
- `tc = 5 bps`
- 协方差矩阵做了 `15%` diagonal shrinkage，再投影到 PSD
- 每周按 `hmm_regime_records.csv` 的 rebalance 日期执行

## 关键参数

| 参数 | 取值 |
|---|---:|
| 回看窗口 | 252 日 |
| Skip window | 21 日 |
| 风险厌恶系数 `gamma` | 5.0 |
| 交易成本 | 0.0005 |
| 协方差 shrinkage | 0.15 |
| 评估区间 | 2013-01-08 到 2024-12-31 |

## 结果表

实验结果保存在这里：

- [forecast_strategy_performance.csv](./results/forecast_strategy_performance.csv)
- [forecast_quality.csv](./results/forecast_quality.csv)
- [excess_vs_external_baselines.csv](./results/excess_vs_external_baselines.csv)
- [comparison_table.csv](./results/comparison_table.csv)
- [strategy_weights.csv](./results/strategy_weights.csv)

### 主结果

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Final Wealth |
|---|---:|---:|---:|---:|---:|
| `RawMean_HMMMPC` | 13.03% | 18.19% | 0.765 | -32.89% | 4.332 |
| `Momentum12_1Scaled_HMMMPC` | 13.61% | 17.94% | 0.801 | -33.71% | 4.603 |
| `VolAdjMomentum_HMMMPC` | 12.11% | 17.00% | 0.758 | -33.61% | 3.928 |
| `ShrinkBlend_HMMMPC` | 12.72% | 17.47% | 0.773 | -33.13% | 4.190 |

### 相对既有 baseline 的超额表现

最好的版本是 `Momentum12_1Scaled_HMMMPC`：

- 相对 `EqualWeight_weekly`，年化收益 `+0.96%`，Sharpe `+0.022`
- 相对 `Markowitz_CVXPY_weekly`，年化收益 `+3.73%`，Sharpe `+0.122`
- `forecast_quality.csv` 里，mean Spearman IC 也从 `0.024` 提升到 `0.030`

## 是否提升

**有提升，而且提升是清晰的。**

在这次实验里，`momentum_12_1` 版本是最稳定的赢家。它比原始 rolling mean 更好，也超过了项目里已有的两个 baseline（等权和单期 Markowitz）在年化收益和 Sharpe 上的表现。

需要诚实说明的是：

- 它的最大回撤并没有明显优于 Markowitz；
- `vol_adj_momentum` 过于保守，收益和 Sharpe 都没有打赢等权；
- `shrink_blend` 比 raw mean 更稳，但不如纯 `12-1 momentum` 干净。

## 原因分析

我对结果的理解是：

1. **raw rolling mean 太噪声化。** 过去一年日收益的简单均值，对 sector ETF 来说经常被短期波动和均值回归干扰。
2. **12-1 momentum 更符合行业 ETF 的结构。** 跳过最近 21 天后，信号更像趋势跟随，而不是把短期反转当成长期预期。
3. **风险调整不一定总是更好。** `vol_adj_momentum` 虽然更稳，但它把高波动、高弹性的行业压得太厉害，结果偏防御，收益被拖下去。
4. **blending 有帮助，但过度保守会削弱 alpha。** `shrink_blend` 比 raw mean 稳一点，但没有纯 momentum 那么直接。

这一步的结论很适合写进论文：**模型的收益输入不应该用原始 rolling mean，而应该用更稳健的 momentum-style signal，再通过标准化和 shrinkage 送入优化器。**

## 运行方式

在项目根目录下执行：

```bash
/opt/anaconda3/envs/pytorch_py=3.8/bin/python improvements/01_robust_return_forecasts/run_experiment.py
```

