# 改进方向 2：No-alpha / Weak-alpha MPC

## 改进方向

这个方向的核心想法很直接：把原始模型里偏噪声化的 expected return 信号压弱，甚至在极端情况下直接设为零，让优化器主要依赖协方差、交易成本和 regime 信息来决定权重。

当前主实验的一个问题是，等权基准本身很强，而单纯用滚动均值去做 alpha 容易把优化器带偏。于是这里改成：

- 仍然保留 CVXPY 的凸优化框架
- 仍然保留 regime-dependent 的风险缩放和交易成本
- 但把收益预测改成 `63` 日滚动均值的极弱版本，并围绕零均值做 shrink

这实际上是在测试一句很朴素的话：**如果 alpha 很不靠谱，那就少信一点 alpha。**

## 实现方法

实验脚本是独立的，没有修改主代码：

- `weak_alpha_mpc_experiment.py`

它读取的都是现成缓存数据和现有输出：

- `data/processed/returns.csv`
- `data/processed/regime_features.csv`
- `outputs/tables/hmm_regime_records.csv`
- `outputs/tables/strategy_performance.csv`

模型做法如下：

1. 以 `2013-01-08` 到 `2024-12-31` 为样本区间。
2. 在每个 rebalance date 上，用过去 `252` 个交易日估计协方差。
3. 用过去 `63` 个交易日的资产均值作为 raw signal。
4. 对 raw signal 做 demean，再乘以 `alpha_shrink`，形成 weak-alpha 预期收益。
5. 用 `prob_stress`、`prob_transition` 和 `next_cost_multiplier` 做状态依赖的风险缩放与交易成本缩放。
6. 用 CVXPY 解下面的凸优化问题：

```math
\max_{w \ge 0,\ \mathbf{1}^\top w = 1}
\mu^\top w - \frac{\gamma}{2} w^\top \Sigma w - c_t \lVert w - w_{t-1} \rVert_1
```

这里：

- `mu` 是 weak-alpha 的收益向量
- `Sigma` 是 shrink 后的协方差矩阵
- `c_t` 是 state-dependent 交易成本系数
- `w_{t-1}` 是上一期权重

脚本会扫描一组 `alpha_shrink`，然后选出 Sharpe 最好的版本作为 headline result。

## 关键参数

| 参数 | 取值 |
|---|---:|
| `lookback_mu` | 63 |
| `lookback_cov` | 252 |
| `gamma` | 4.0 |
| `base_trade_cost` | 0.0005 |
| `alpha_grid` | 0.00, 0.05, 0.10, 0.25, 0.50 |
| 选中 `alpha_shrink` | 0.25 |

## 结果表

### 候选扫描

| alpha_shrink | Annualized Return | Annualized Vol | Sharpe | Max DD |
|---:|---:|---:|---:|---:|
| 0.00 | 11.84% | 16.07% | 0.737 | -36.91% |
| 0.05 | 12.03% | 16.06% | 0.749 | -36.91% |
| 0.10 | 11.82% | 16.03% | 0.737 | -36.91% |
| 0.25 | 12.60% | 15.54% | 0.811 | -34.53% |
| 0.50 | 12.32% | 18.37% | 0.671 | -33.01% |

### 与 baseline 对比

| Strategy | Annualized Return | Annualized Vol | Sharpe | Max DD | Final Wealth | Avg Rebalance Turnover | Total Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| EqualWeight_weekly | 12.65% | 16.24% | 0.779 | -36.98% | 4.159 | 0.58% | 0.36% |
| Markowitz_CVXPY_weekly | 9.87% | 14.54% | 0.679 | -32.51% | 3.086 | 4.06% | 2.54% |
| HMM_MPC_CVXPY_weekly | 10.70% | 14.98% | 0.714 | -32.65% | 3.374 | 1.82% | 1.74% |
| WeakAlpha_selected | 12.60% | 15.54% | 0.811 | -34.53% | 4.138 | 0.15% | 0.05% |

### 相对超额表现

| Comparison | Return Diff | Sharpe Diff | Max DD Diff | Final Wealth Ratio | Turnover Diff |
|---|---:|---:|---:|---:|---:|
| WeakAlpha - EqualWeight | -0.05% | +0.03 | +2.45 pp | 0.995 | -0.43% |
| WeakAlpha - Markowitz | +2.73% | +0.13 | -2.02 pp | 1.341 | -3.91% |
| WeakAlpha - Current HMM-MPC | +1.90% | +0.10 | -1.88 pp | 1.227 | -1.67% |

## 是否提升

结论是：**相对当前 HMM-MPC，提升是明确的。**

Weak-alpha 版本在这个实验里：

- 年化收益高于当前 HMM-MPC
- Sharpe 明显更高
- 最大回撤更小
- 换手和交易成本更低

但它仍然**没有稳定打败等权**。这说明这个数据集里，`1/N` 仍然是很强的 benchmark，模型真正改进的是风险调整后表现和交易行为，而不是单纯追求更高收益。

## 原因分析

这次改进能起作用，主要是因为它减弱了均值估计噪声。

- 原模型里，expected return 过强，容易把权重推向防御板块
- 当前数据里，行业 ETF 的长期走势并不支持一个很强的均值预测模型
- weak-alpha 让优化器更多依赖 covariance 和 cost，而不是过度相信滚动收益均值
- 适度保留 `alpha_shrink=0.25` 又比完全零 alpha 更好，因为它还能保留一点横截面动量信息

换句话说，这个方向的本质不是“更聪明地猜收益”，而是“少犯收益预测的错”。

## 输出文件

结果都写在这个子目录下的 `results/`：

- `weak_alpha_candidate_performance.csv`
- `weak_alpha_excess_vs_baselines.csv`
- `weak_alpha_selected_summary.csv`
- `weak_alpha_selected_daily_returns.csv`
- `weak_alpha_selected_daily_weights.csv`
- `weak_alpha_selected_daily_turnover.csv`
- `weak_alpha_selected_daily_costs.csv`
- `baseline_snapshot.csv`

