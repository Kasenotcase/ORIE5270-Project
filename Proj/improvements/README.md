# 四个改进方向实验总览

本目录包含四个互相独立的小实验。每个实验都在自己的子文件夹中运行，没有修改主工程已有代码。

## 结果总览

| 方向 | 子文件夹 | 核心结果 | 初步结论 |
|---|---|---|---|
| 1. Robust return forecasts | `01_robust_return_forecasts` | `Momentum12_1Scaled_HMMMPC` 年化收益 `13.61%`，Sharpe `0.801` | 最有希望，已经超过等权和 Markowitz |
| 2. Weak-alpha MPC | `02_weak_alpha_mpc` | `alpha_shrink=0.25` 年化收益 `12.60%`，Sharpe `0.811` | 非常稳健，说明减少 noisy alpha 依赖是关键 |
| 3. Validation tuning | `03_validation_tuning` | 测试期年化收益 `14.18%`，Sharpe `0.770` | 参数校准有效，尤其能改善原 HMM-MPC |
| 4. Defensive asset SHY | `04_defensive_asset` | 年化收益 `6.06%`，Sharpe `0.607`，最大回撤 `-21.03%` | 单独使用不提升收益，但显著降低回撤 |

## 最重要发现

1. 原模型最大的问题不是 CVXPY 或 HMM 框架，而是预期收益输入太弱、太容易被噪声驱动。
2. Momentum/vol-scaled forecast 和 weak-alpha shrinkage 都能明显改善表现。
3. 参数校准也很有用，但必须严格区分 validation period 和 test period。
4. 防御资产方向不能直接套当前参数，否则模型会过度保守；它更适合作为 stress-regime-only 风险出口。

## 后续建议

最推荐合并的方向是：

```text
Momentum or weak-alpha expected return
+ validation-tuned gamma / turnover cap
+ HMM controls risk and transaction cost
+ optional stress-only defensive asset exposure
```

也就是说，最终模型不应该继续使用 raw rolling mean；应当把 HMM 的作用收敛到 regime-aware risk/cost control，并使用更稳健的 alpha 输入。
