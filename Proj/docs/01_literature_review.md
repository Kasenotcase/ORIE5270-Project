# Literature Review: Regime-Aware Multi-Period Trading via Convex Optimization

## Abstract

This review positions the project within the literature on portfolio optimization, multi-period trading, market regimes, and liquidity-sensitive transaction costs. The central methodological choice is to combine a model predictive control formulation of multi-period trading with a hidden Markov model for daily market regime identification. The optimization layer remains convex and implementable in CVXPY: expected returns, covariance matrices, and trading-cost coefficients are treated as exogenous forecasts, while portfolio weights and trades are the decision variables. The review therefore emphasizes work that supports a feasible empirical design rather than a broad survey of all asset-pricing or machine-learning approaches.

## 1. From Single-Period Mean-Variance Optimization to Dynamic Trading

Modern portfolio selection begins with Markowitz (1952), who formalized the trade-off between expected return and variance in a single-period setting. In its simplest long-only version, the investor chooses portfolio weights \(w \in \mathbb{R}^N\) to solve

$$
\max_{w} \quad \mu^\top w - \gamma w^\top \Sigma w
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \ge 0,
$$

where \(\mu\) is the expected return vector, \(\Sigma\) is the covariance matrix, and \(\gamma > 0\) controls risk aversion. This model remains the natural baseline because it is interpretable and directly expressible as a quadratic program. Its limitation is equally clear: a single-period mean-variance optimizer ignores the path of future rebalancing, the cost of moving from the current portfolio to the target portfolio, and the fact that market conditions are not stationary.

Empirical studies have also shown that the unconstrained sample mean-variance optimizer can be unstable when expected returns and covariances are estimated from noisy historical data. DeMiguel, Garlappi, and Uppal (2009) demonstrate that naive diversification can be difficult to beat out of sample when estimation error is large. This does not invalidate optimization; rather, it motivates a conservative implementation with a small asset universe, regularized covariance estimates, long-only constraints, turnover controls, and simple forecast models.

## 2. Multi-Period Convex Trading and Model Predictive Control

The main theoretical foundation for this project is the multi-period convex trading framework of Boyd et al. (2017). Their contribution is to express trading as a sequence of convex optimization problems that balance expected return, risk, transaction cost, and holding cost. The multi-period version is a model predictive control problem: at each decision date, the optimizer plans trades over a finite horizon, executes only the first trade, and then re-solves when new information arrives. This receding-horizon structure is well matched to daily portfolio management because forecasts and market conditions are updated continuously.

The practical importance of the framework is that it separates prediction from optimization. Forecasts of future returns, risk, and trading costs can come from any model, provided they enter the optimizer as fixed numerical inputs at the time of solving. This separation is essential for maintaining convexity. In the present project, hidden market regimes affect \(\hat{\mu}_{t+h}\), \(\hat{\Sigma}_{t+h}\), and the trading-cost vector \(a_{t+h}\), but the regime itself is not a decision variable. The optimizer only sees a sequence of predicted parameters.

Transaction costs are central in multi-period trading because frequent rebalancing can erase gross alpha. Garleanu and Pedersen (2013) show that with predictable returns and trading costs, the optimal policy trades partially toward a moving target rather than fully jumping to the current Markowitz portfolio. This result supports the intuition behind MPC: an investor should account for future target changes and avoid excessive turnover. Boyd et al. (2017) provide a computationally tractable way to implement this idea with convex optimization, while CVXPY provides the modeling language needed to translate the problem into code (Diamond and Boyd, 2016).

## 3. Market Regimes and Nonstationary Return Distributions

Financial return distributions change across time. Hamilton (1989) introduced a Markov regime-switching framework in which unobserved states drive changes in the data-generating process. In finance, regime models are useful because equity returns, volatility, correlations, and liquidity conditions behave differently during calm expansions, transitional drawdowns, and crisis periods. Ang and Timmermann (2012) review how regime changes affect asset allocation, risk premia, and financial decision-making.

Hidden Markov models are particularly appropriate for this project because they combine probabilistic state classification with a transition matrix. Instead of assigning each day to a fixed cluster and then treating the next regime as a deterministic label, an HMM produces posterior probabilities over states. If \(p_t\) denotes the probability distribution over regimes at date \(t\) and \(P\) is the transition matrix, the \(h\)-step-ahead regime distribution is

$$
p_{t+h \mid t} = p_t P^h.
$$

This probability forecast is useful for MPC because the optimizer needs parameters for several future periods, not only a one-day label. Regime probabilities can be used to form mixture forecasts of means and covariances, which smooths abrupt parameter changes and avoids the instability of hard regime filtering.

Nystrup, Madsen, and Lindstrom (2018) provide the closest precedent for the present project. They combine forecasts from a hidden Markov model with model predictive control for dynamic portfolio optimization and show that regime-based MPC can improve drawdown control relative to static allocation rules. Their study supports the central design principle here: regime identification should not replace portfolio optimization; it should feed conditional estimates into a disciplined multi-period optimizer.

## 4. Liquidity, Volatility, and State-Dependent Trading Costs

The project departs from a standard regime-aware allocation model by making transaction costs state dependent. This choice is motivated by the literature on liquidity risk. Amihud (2002) introduced a daily illiquidity measure based on the ratio of absolute return to dollar trading volume. The measure is simple, public-data-compatible, and well suited to daily data:

$$
\text{ILLIQ}_{i,t} = \frac{|r_{i,t}|}{\text{DollarVolume}_{i,t}}.
$$

The economic interpretation is that a larger price movement per dollar traded indicates poorer liquidity. Acharya and Pedersen (2005) show that liquidity risk is priced, while Brunnermeier and Pedersen (2009) describe how market liquidity and funding liquidity can reinforce one another during stress. These studies justify the project's assumption that trading should be penalized more heavily in crisis or illiquid regimes.

Volatility is another natural state variable. The VIX index is a forward-looking measure of expected S&P 500 volatility derived from options prices, and daily VIX observations are publicly available through FRED as `VIXCLS`. Credit spreads also provide useful information about systemic stress. The ICE BofA US High Yield Index Option-Adjusted Spread, available on FRED as `BAMLH0A0HYM2`, is a daily proxy for credit-market risk appetite. Combining realized volatility, VIX, credit spreads, trend, drawdown, and Amihud illiquidity gives the HMM a compact set of economically interpretable daily features.

## 5. Recent Regime-Aware and Decision-Aware Directions

Recent work has continued to move toward explicit regime awareness and closer integration between prediction and optimization. Zhang et al. (2025) propose RegimeFolio, an arXiv preprint that uses VIX-based regime segmentation, sector-specific forecasting, and adaptive mean-variance allocation. The paper is relevant as a recent example of regime-aware sector allocation, although it should be treated as a preprint rather than a settled benchmark.

Linghu, Liu, and Deng (2025) propose Integrated Prediction and Multi-period Portfolio Optimization, also an arXiv preprint, which embeds a differentiable multi-period optimization layer into a learning system. This work is conceptually related because it recognizes that forecasts should be evaluated by decision quality, not only prediction error. The present project does not implement an end-to-end differentiable learner because that would add unnecessary complexity for the empirical scope. Instead, it keeps the prediction layer and optimization layer modular, which is more transparent and easier to validate in a moderate-scale empirical study.

## 6. Project Positioning

The proposed project occupies a tractable middle ground. It is richer than a single-period Markowitz model because it includes horizon planning, transaction costs, and time-varying regimes. It is less complex than full end-to-end learning because it uses a small daily feature set, a low-dimensional ETF universe, and a convex optimization layer that can be solved reliably with CVXPY.

The key research hypothesis is:

$$
\text{Regime-aware MPC with state-dependent trading costs improves net risk-adjusted performance and drawdown control relative to regime-agnostic baselines.}
$$

The hypothesis is empirical, not assumed true. The literature motivates the design, but the backtest must evaluate whether the regime model improves net performance after transaction costs. A credible empirical section should therefore compare at least three strategies: an equal-weight portfolio, a single-period Markowitz optimizer, and the proposed HMM-informed multi-period convex optimizer.

## 7. Implications for the Empirical Design

The literature implies several implementation constraints.

First, the asset universe should remain small. Sector ETFs are preferable to hundreds of individual equities because covariance estimation is more stable and the interpretation of regime-dependent allocation is clearer. For a long 2010-2024 sample, nine long-history Select Sector SPDR ETFs are the most practical primary universe; the full eleven-sector GICS universe can be used only over a shorter sample unless proxy ETFs are introduced.

Second, all regime features should be daily and publicly observable. Adjusted prices and volume can be collected from Yahoo Finance through `yfinance`; VIX and high-yield spreads can be collected from FRED. The HMM should use rolling or expanding estimation windows to avoid look-ahead bias.

Third, regime estimates must remain outside the optimizer. Conditional means, covariances, and trading-cost multipliers are estimated before solving the CVXPY problem. The optimizer then solves a deterministic convex program at each rebalancing date.

Fourth, transaction costs should be modeled conservatively. A linear turnover penalty, possibly multiplied by a regime stress coefficient, is sufficient for a convex quadratic program. More realistic nonlinear market-impact models can be added later, but they are not necessary for the core research question.

## References

Acharya, V. V., and Pedersen, L. H. (2005). Asset pricing with liquidity risk. *Journal of Financial Economics*, 77(2), 375-410. https://doi.org/10.1016/j.jfineco.2004.06.007

Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56. https://doi.org/10.1016/S1386-4181(01)00024-6

Ang, A., and Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313-337. https://doi.org/10.1146/annurev-financial-110311-101808

Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., and Speth, J. (2017). Multi-period trading via convex optimization. *Foundations and Trends in Optimization*, 3(1), 1-76. https://doi.org/10.1561/2400000023

Brunnermeier, M. K., and Pedersen, L. H. (2009). Market liquidity and funding liquidity. *The Review of Financial Studies*, 22(6), 2201-2238. https://doi.org/10.1093/rfs/hhn098

DeMiguel, V., Garlappi, L., and Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915-1953. https://doi.org/10.1093/rfs/hhm075

Diamond, S., and Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex optimization. *Journal of Machine Learning Research*, 17(83), 1-5. https://jmlr.org/papers/v17/15-408.html

Garleanu, N., and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance*, 68(6), 2309-2340. https://doi.org/10.1111/jofi.12080

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384. https://doi.org/10.2307/1912559

Ledoit, O., and Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4), 110-119. https://doi.org/10.3905/jpm.2004.110

Linghu, Y., Liu, Z., and Deng, Q. (2025). Integrated prediction and multi-period portfolio optimization. arXiv:2512.11273. https://doi.org/10.48550/arXiv.2512.11273

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x

Nystrup, P., Madsen, H., and Lindstrom, E. (2018). Dynamic portfolio optimization across hidden market regimes. *Quantitative Finance*, 18(1), 83-95. https://doi.org/10.1080/14697688.2017.1342857

Zhang, Y., Goel, D., Ahmad, H., and Szabo, C. (2025). RegimeFolio: A regime aware ML system for sectoral portfolio optimization in dynamic markets. arXiv:2510.14986. https://doi.org/10.48550/arXiv.2510.14986
