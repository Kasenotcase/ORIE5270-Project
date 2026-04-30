# 方向4：加入防御资产 SHY 的实验说明

## 改进方向

本实验在原有 9 个行业 ETF 的基础上加入防御资产 `SHY`，目的是检验：如果组合有一个低波动、类现金/短债资产，HMM-MPC 是否能在压力状态下真正降低权益风险，而不是只能在股票行业内部轮动。

## 实现方法

- 原有 9 个行业 ETF、宏观变量和 regime features 读取主工程已经清洗好的缓存数据。
- `SHY` 使用 Yahoo Finance chart endpoint 单独下载，并缓存在本子目录。
- 扩展资产池为 10 个资产：`XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, SHY`。
- 重新回测三类策略：
  - `EqualWeight_weekly`
  - `Markowitz_CVXPY_weekly`
  - `HMM_MPC_SHY_weekly`
- HMM regime 特征仍然基于原有市场/宏观变量，不把 `SHY` 本身作为 regime 输入，避免把资产池变化和 regime 识别变化混在一起。

## 关键结果


| 策略                     | 年化收益   | 年化波动   | Sharpe | 最大回撤    | 最终财富  |
| ---------------------- | ------ | ------ | ------ | ------- | ----- |
| EqualWeight_weekly     | 11.04% | 14.69% | 0.752  | -33.66% | 3.411 |
| Markowitz_CVXPY_weekly | 5.84%  | 9.50%  | 0.616  | -21.87% | 1.945 |
| HMM_MPC_SHY_weekly     | 6.06%  | 9.98%  | 0.607  | -21.03% | 1.992 |


相对原始 9 资产 HMM-MPC：


| 指标     | 原始模型    | 加 SHY 后 | 变化        |
| ------ | ------- | ------- | --------- |
| 年化收益   | 10.70%  | 6.06%   | -4.64 pp  |
| 年化波动   | 14.98%  | 9.98%   | -5.00 pp  |
| Sharpe | 0.714   | 0.607   | -0.107    |
| 最大回撤   | -32.65% | -21.03% | +11.62 pp |
| 最终财富   | 3.374   | 1.992   | -1.382    |


## 是否提升

没有全面提升。

加入 `SHY` 后，最大回撤明显改善，从原始模型的约 `-32.65%` 改善到 `-21.03%`，说明防御资产确实提供了去风险出口。但收益和 Sharpe 明显下降，最终财富也低于原始模型和等权基准。

## 原因分析

这个结果说明“加入防御资产”方向本身是有逻辑的，但不能直接用当前参数套进去。`SHY` 的长期年化收益只有约 `1.13%`，波动也很低，优化器在当前风险厌恶和成本参数下容易过度偏向低波动资产，牺牲了长期权益风险溢价。

因此，防御资产方向更适合与以下改动结合：

- 降低风险厌恶参数；
- 给 `SHY` 设置单独较低权重上限；
- 只在 stress regime 提高 `SHY` 的可用权重；
- 用 momentum 或 weak-alpha 版本改善权益资产的预期收益输入。

## 输出文件

- `run_defensive_asset_experiment.py`
- `data/raw/shy_yahoo_chart_ohlcv_raw.csv`
- `data/processed/returns_extended.csv`
- `outputs/tables/strategy_performance.csv`
- `outputs/tables/excess_performance_vs_baselines.csv`
- `outputs/tables/comparison_vs_original_model.csv`
- `outputs/tables/headline_results.csv`
- `outputs/reports/README.md`

