# 03_validation_tuning

## 改进方向

这一方向对应“基于验证集调参”的改进：不改变主模型结构，只对三个最容易影响实盘表现的超参数做系统搜索：

1. `gamma`：风险厌恶系数，控制收益和方差之间的权衡。
2. `turnover_cap`：单次再平衡的换手上限，防止模型过度交易。
3. `stress_multiplier`：压力状态下的交易成本放大系数，避免在高风险阶段频繁换仓。

目标不是把模型写得更复杂，而是在当前可执行框架内，通过验证期选择一组更稳的参数，让测试期表现更稳定、更可解释。

## 实现方法

脚本：`run_validation_tuning.py`

实现逻辑如下：

1. 读取 `data/processed/returns.csv` 作为资产收益数据。
2. 读取 `outputs/tables/hmm_regime_records.csv` 作为 regime 预测缓存。
3. 如果 regime 缓存可用，则直接使用其中的 `next_prob_stress` 作为压力状态概率；如果不可用，则退化为基于 `regime_features.csv` 的简化代理。
4. 用滚动窗口估计均值和协方差，协方差做简单收缩并投影到 PSD。
5. 在每个再平衡日求解一个 CVXPY 凸优化问题：

   $$\max_w \mu^T w - \gamma w^T \Sigma w - c_t \cdot \frac{1}{2}\lVert w-w_{prev} \rVert_1$$

   其中 `c_t` 会随压力概率和 `stress_multiplier` 动态变化。
6. 在验证期做小网格搜索，按验证期 Sharpe 选择最优参数。
7. 用选中的参数固定跑测试期，并与已有 baseline 结果做对照。

## 关键参数

脚本中使用的网格如下：

- `gamma in {1.0, 3.0, 10.0}`
- `turnover_cap in {0.05, 0.10, 0.20}`
- `stress_multiplier in {0.0, 1.0, 3.0}`

说明：

- `turnover_cap` 是单次再平衡允许的最大 turnover。
- 交易成本基准设为 5 bps。
- 采用周频再平衡，保持和现有实证框架一致。

## 验证期

- 验证期：`2013-01-11` 到 `2018-12-31`

这一段用于选参，不参与最终测试结果汇报。

## 测试期

- 测试期：`2019-01-01` 到 `2024-12-31`

这一段只用验证期选出的参数跑一次，作为最终测试表现。

## 结果表

运行后结果保存在 `results/tables/` 下，主要文件包括：

- `validation_grid_results.csv`
- `selected_parameters.csv`
- `validation_selected_metrics.csv`
- `test_selected_metrics.csv`
- `reference_strategy_metrics_test.csv`
- `tuned_excess_vs_references.csv`
- `summary_metrics.csv`

## 结果

验证期选出的最优参数为：

- `gamma = 10.0`
- `turnover_cap = 0.05`
- `stress_multiplier = 3.0`

对应表现：

- 验证期 Sharpe: `0.8186`
- 测试期 Sharpe: `0.7699`
- 测试期年化收益: `14.18%`
- 测试期年化波动: `18.42%`
- 测试期最大回撤: `-32.46%`
- 测试期平均换手: `1.76%`

和参考策略对比，测试期结果如下：

- 相比原始 `HMM_MPC_CVXPY_weekly`，年化收益高约 `4.25` 个百分点，Sharpe 高约 `0.204`，最终财富约高 `25.4%`，换手更低，交易成本也更低。
- 相比 `EqualWeight_weekly`，年化收益略低约 `0.59` 个百分点，但 Sharpe 略高，最大回撤更小，说明调参后模型在风险控制和交易稳定性上更有优势。

## 是否提升

这次改进是有提升的，至少相对原始 HMM-MPC 是明确提升的。重点看三件事：

1. 相比原始 HMM-MPC，Sharpe 是否更高。
2. 相比原始 HMM-MPC，换手和交易成本是否更低。
3. 相比等权基准，是否至少在风险控制和交易稳定性上更有优势。

从结果看，调参后模型已经把原始 HMM-MPC 的弱点补了一部分，尤其是换手和成本控制。它没有彻底击败等权，但已经从“明显落后”变成“接近甚至在 Sharpe 上略有优势”的状态。

## 原因分析

如果调参后提升有限，可能原因主要有四个：

1. 当前资产池本身就比较强，等权在样本里已经很有竞争力。
2. 收益预测信号仍然偏弱，单靠调 `gamma` 和成本参数只能改善交易行为，难以凭空制造 alpha。
3. regime 概率更多影响的是风险和交易成本，而不是截面收益排序。
4. 训练窗口和验证窗口有限，参数对样本切分比较敏感。

所以这一路线的定位是“稳健性校准”，不是结构性重写。它更适合和其他改进方向一起使用。
