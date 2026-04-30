# Theoretical Model: State-Dependent Multi-Period Trading as a Convex Program

## Abstract

The optimization layer defines a regime-aware multi-period portfolio strategy. At each rebalancing date, the model plans a sequence of trades over a finite horizon and executes only the first trade. Expected returns, covariance matrices, and transaction-cost coefficients are produced by the market-regime layer and treated as fixed inputs. The decision variables are portfolio weights and buy/sell trades. With long-only constraints, full investment, linear turnover costs, and positive semidefinite covariance matrices, the model is a convex quadratic program that can be solved directly in CVXPY.

## 1. Trading Timeline

Let \(t\) denote a daily rebalancing date and \(H\) the planning horizon. The empirical implementation should use a short horizon, for example \(H=5\) trading days, because regime and return forecasts become unreliable at longer horizons.

At date \(t\), the investor observes current holdings \(w_t \in \mathbb{R}^N\), market-regime features through date \(t\), and the current HMM posterior distribution. The regime layer produces forecasts

$$
\left\{
\hat{\mu}_{t,h},
\hat{\Sigma}_{t,h},
\hat{a}_{t,h}
\right\}_{h=1}^{H},
$$

where \(\hat{\mu}_{t,h} \in \mathbb{R}^N\) is the forecast return vector, \(\hat{\Sigma}_{t,h} \in \mathbb{S}_+^N\) is the forecast covariance matrix, and \(\hat{a}_{t,h} \in \mathbb{R}_+^N\) is the per-dollar trading-cost vector for horizon step \(h\).

The optimizer chooses a planned path of post-trade weights. Only the first trade is executed. At \(t+1\), realized returns update the portfolio, the regime model is updated, and a new optimization problem is solved. This receding-horizon rule is the model predictive control structure.

## 2. Decision Variables

For each horizon step \(h=1,\dots,H\), define:

$$
x_h \in \mathbb{R}^N
$$

as the post-trade portfolio weights planned for step \(h\). Trades are represented by nonnegative buy and sell variables:

$$
b_h \in \mathbb{R}_+^N,
\quad
s_h \in \mathbb{R}_+^N.
$$

The net trade is

$$
z_h = b_h - s_h.
$$

Let \(x_0 = w_t\). The planned holdings satisfy

$$
x_h - x_{h-1} = b_h - s_h,
\quad h=1,\dots,H.
$$

The turnover at step \(h\) is

$$
\text{TO}_h=
\frac{1}{2}\mathbf{1}^\top(b_h+s_h).
$$

The split buy/sell formulation converts the absolute-value trading cost into a linear term, which is convenient for CVXPY and standard quadratic-program solvers.

## 3. Objective Function

The objective maximizes forecast risk-adjusted return net of trading costs:

$$
\max_{\{x_h,b_h,s_h\}_{h=1}^{H}}
\sum_{h=1}^{H}
\beta^{h-1}
\left[
\hat{\mu}_{t,h}^\top x_h-
\frac{\gamma}{2}
x_h^\top \hat{\Sigma}_{t,h}x_h-
\hat{a}_{t,h}^\top(b_h+s_h)-
\frac{\kappa}{2}\|b_h-s_h\|_2^2
\right].
$$

The terms have the following roles:

- \(\hat{\mu}_{t,h}^\top x_h\) rewards forecast return.
- \(x_h^\top \hat{\Sigma}_{t,h}x_h\) penalizes portfolio risk.
- \(\hat{a}_{t,h}^\top(b_h+s_h)\) penalizes turnover and represents linear transaction costs.
- \(\|b_h-s_h\|_2^2\) is an optional small quadratic trading penalty that stabilizes allocations and discourages abrupt changes.
- \(\beta \in (0,1]\) discounts later horizon steps. A simple implementation can set \(\beta=1\).

The state dependence enters only through \(\hat{\mu}_{t,h}\), \(\hat{\Sigma}_{t,h}\), and \(\hat{a}_{t,h}\). In stress regimes, the HMM increases the probability weight on high-volatility and illiquid states, which raises covariance estimates and trading-cost coefficients.

## 4. Constraints

The basic feasible set is:

$$
\mathbf{1}^\top x_h = 1,
\quad h=1,\dots,H,
$$

$$
0 \le x_h \le \bar{x}\mathbf{1},
\quad h=1,\dots,H,
$$

$$
x_h - x_{h-1} = b_h - s_h,
\quad h=1,\dots,H,
$$

$$
b_h \ge 0,\quad s_h \ge 0,
\quad h=1,\dots,H,
$$

$$
\frac{1}{2}\mathbf{1}^\top(b_h+s_h) \le \bar{\tau},
\quad h=1,\dots,H.
$$

The full-investment constraint \(\mathbf{1}^\top x_h=1\) and the trade-linking constraint imply self-financing in weight space. The long-only constraint avoids leverage and shorting. The upper bound \(\bar{x}\), such as \(\bar{x}=0.40\), prevents the optimizer from concentrating too heavily in one ETF. The turnover cap \(\bar{\tau}\) is optional but recommended for numerical stability and realistic trading behavior.

In the backtest, transaction costs should also be deducted from realized portfolio value. If the executed first trade is \(z_1=b_1-s_1\), the realized cost rate can be

$$
c_t^{\text{realized}}=
\hat{a}_{t,1}^\top(b_1+s_1).
$$

The next-day wealth update is then

$$
V_{t+1}=
V_t
\left(1-c_t^{\text{realized}}\right)
\left(1+R_{t+1}^\top x_1\right),
$$

where \(R_{t+1}\) is the realized close-to-close return vector after the trade.

## 5. Convexity

The model is convex in the standard disciplined convex programming sense. The objective is concave because it is the sum of linear return terms minus convex quadratic risk terms, linear nonnegative trading-cost terms, and convex quadratic trading-penalty terms. The constraints are affine or convex bound constraints. Therefore, maximizing this concave objective over the feasible set is a convex optimization problem.

The key requirements are:

$$
\hat{\Sigma}_{t,h} \succeq 0
\quad \text{and} \quad
\hat{a}_{t,h} \ge 0
\quad \text{for all } h.
$$

The regime layer must project or shrink covariance matrices so that they are positive semidefinite. If numerical estimation produces a small negative eigenvalue, the implementation should replace the matrix by its nearest positive semidefinite approximation or add a small ridge term \(\eta I\).

## 6. CVXPY Formulation

The following pseudocode gives the direct CVXPY structure:

```python
import cvxpy as cp

H, n = mu.shape[0], mu.shape[1]
x = cp.Variable((H, n))
b = cp.Variable((H, n), nonneg=True)
s = cp.Variable((H, n), nonneg=True)

objective_terms = []
constraints = []
prev = w0

for h in range(H):
    trade = b[h] - s[h]
    turnover = 0.5 * cp.sum(b[h] + s[h])

    constraints += [
        x[h] - prev == trade,
        cp.sum(x[h]) == 1,
        x[h] >= 0,
        x[h] <= weight_cap,
        turnover <= turnover_cap,
    ]

    objective_terms.append(
        beta**h * (
            mu[h] @ x[h]
            - 0.5 * gamma * cp.quad_form(x[h], Sigma[h])
            - cost[h] @ (b[h] + s[h])
            - 0.5 * kappa * cp.sum_squares(trade)
        )
    )

    prev = x[h]

problem = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints)
problem.solve(solver="OSQP")
first_trade = (b.value[0] - s.value[0])
post_trade_weight = x.value[0]
```

This formulation is a quadratic program when \(\hat{\Sigma}_{t,h}\) is fixed and positive semidefinite. OSQP is a natural solver. CLARABEL or ECOS can be used as alternatives if solver status is not optimal.

## 7. Parameter Choices

The empirical section should begin with conservative parameters:

$$
H = 5,
\quad
\beta = 1,
\quad
\bar{x} = 0.40,
\quad
\bar{\tau} = 0.50,
\quad
a_i^0 = 0.0005.
$$

The risk-aversion parameter \(\gamma\) should be selected by a small validation grid, for example

$$
\gamma \in \{5, 10, 20, 40, 80\}.
$$

The quadratic trading penalty \(\kappa\) can be set to a small value, such as \(10^{-4}\), or omitted in the first implementation. The regime stress multipliers should be fixed before evaluating the final out-of-sample period. A simple default is:

$$
m_{\text{calm}}=1.0,\quad
m_{\text{transition}}=1.5,\quad
m_{\text{stress}}=3.0.
$$

The empirical section should not tune these multipliers directly on the final test performance. They should be treated as design parameters motivated by liquidity risk.

## 8. Relationship to Baselines

The proposed model nests simpler strategies.

Equal weight is

$$
w_t^{\text{EW}} = \frac{1}{N}\mathbf{1}.
$$

The single-period Markowitz baseline is recovered by setting \(H=1\), using regime-agnostic estimates \((\mu_0,\Sigma_0)\), and setting trading costs to zero or to a constant:

$$
\max_x
\quad
\mu_0^\top x-
\frac{\gamma}{2}x^\top \Sigma_0 x
\quad
\text{s.t.}
\quad
\mathbf{1}^\top x=1,\quad 0 \le x \le \bar{x}\mathbf{1}.
$$

The proposed model differs from this baseline in three ways: it plans multiple future trades, it uses regime-mixture forecasts of return and risk, and it raises trading penalties when the regime layer detects stress or illiquidity.

## 9. Empirical Algorithm

At each date \(t\), the empirical procedure is:

1. Update daily price, volume, VIX, and credit-spread data through \(t\).
2. Compute standardized regime features using only data available through \(t\).
3. Fit or update the HMM on an expanding or rolling window.
4. Compute \(q_{t+h\mid t}\) for \(h=1,\dots,H\).
5. Construct \(\hat{\mu}_{t,h}\), \(\hat{\Sigma}_{t,h}\), and \(\hat{a}_{t,h}\).
6. Solve the CVXPY multi-period convex program.
7. Execute only the first trade \(z_1\).
8. Deduct transaction costs and apply realized next-day returns.
9. Repeat at \(t+1\).

This algorithm creates a direct bridge from the market-regime document to the empirical code. The regime model is responsible for state probabilities and conditional parameters; the optimization model is responsible for disciplined trading decisions.

## 10. Evaluation Metrics

The final empirical section should report both performance and trading behavior:

$$
\text{Annualized Return},
\quad
\text{Annualized Volatility},
\quad
\text{Sharpe Ratio},
\quad
\text{Maximum Drawdown},
$$

$$
\text{Average Daily Turnover},
\quad
\text{Realized Transaction Cost},
\quad
\text{Turnover During Stress Regimes}.
$$

The most important diagnostic is not only whether the proposed model has a higher Sharpe ratio, but whether it reduces turnover during high-cost regimes while maintaining or improving drawdown control. This is the empirical implication of state-dependent trading costs.

## References

Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., and Speth, J. (2017). Multi-period trading via convex optimization. *Foundations and Trends in Optimization*, 3(1), 1-76. https://doi.org/10.1561/2400000023

Diamond, S., and Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex optimization. *Journal of Machine Learning Research*, 17(83), 1-5. https://jmlr.org/papers/v17/15-408.html

Garleanu, N., and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance*, 68(6), 2309-2340. https://doi.org/10.1111/jofi.12080

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x

Nystrup, P., Madsen, H., and Lindstrom, E. (2018). Dynamic portfolio optimization across hidden market regimes. *Quantitative Finance*, 18(1), 83-95. https://doi.org/10.1080/14697688.2017.1342857
