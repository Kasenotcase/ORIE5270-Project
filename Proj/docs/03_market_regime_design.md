# Market Regime Design for Daily Regime-Aware Portfolio Optimization

## Abstract

The market regime layer is designed to produce stable, interpretable, daily regime probabilities for a multi-period convex portfolio optimizer. The objective is not to build a complex macro-financial forecasting system, but to identify economically meaningful shifts in trend, volatility, liquidity, and credit stress. The regime model uses public daily data, a small set of economically motivated features, and a hidden Markov model with two to four latent states. Its outputs are \(h\)-step-ahead regime probabilities, regime-conditioned expected returns, regime-conditioned covariance matrices, and state-dependent trading-cost multipliers. These outputs are exogenous inputs to a CVXPY optimization problem.

## 1. Empirical Scope and Data Universe

The preferred empirical universe is a set of liquid U.S. sector ETFs. A full eleven-sector GICS ETF universe is conceptually attractive, but the communication services and real estate Select Sector SPDR ETFs have shorter histories than the older sector funds. To preserve a 2010-2024 backtest without heavy imputation, the primary universe should use the nine long-history Select Sector SPDR ETFs:

$$
\mathcal{A} =
\{\text{XLB}, \text{XLE}, \text{XLF}, \text{XLI}, \text{XLK}, \text{XLP}, \text{XLU}, \text{XLV}, \text{XLY}\}.
$$

The full eleven-sector version can be implemented as a robustness check over the shorter period in which all eleven ETFs are available. The nine-ETF universe is the main recommendation because it is liquid, interpretable, and small enough for stable covariance estimation.

Daily adjusted close prices and volume should be collected through `yfinance`. VIX and high-yield credit spreads should be collected from FRED:

$$
\text{VIXCLS}_t = \text{CBOE Volatility Index},
$$

$$
\text{BAMLH0A0HYM2}_t = \text{ICE BofA US High Yield Index Option-Adjusted Spread}.
$$

The baseline sample should run from January 2010 to December 2024, conditional on data availability. All features at date \(t\) must be computable using information available no later than the close of date \(t\). Portfolio returns used for evaluation should start at \(t+1\).

## 2. Daily Regime Feature Set

Let \(P_{i,t}\) be the adjusted close of asset \(i\), \(V_{i,t}\) its share volume, and

$$
r_{i,t} = \log P_{i,t} - \log P_{i,t-1}.
$$

Let \(P_{m,t}\) denote a market proxy, preferably SPY or an equal-weighted sector ETF index. The regime model should use a compact feature vector \(X_t \in \mathbb{R}^d\). The following features are recommended.

### 2.1 Market Return and Trend

Short- and medium-horizon trend features summarize directional market conditions:

$$
\text{MOM}_{t}^{(L)} = \log P_{m,t} - \log P_{m,t-L},
\quad L \in \{21, 63\}.
$$

A rolling drawdown feature captures distance from recent highs:

$$
\text{DD}_{t}^{(L)} =
\frac{P_{m,t}}{\max_{0 \le j \le L} P_{m,t-j}} - 1,
\quad L = 63.
$$

These features distinguish calm uptrends from persistent drawdowns without requiring intraday data.

### 2.2 Realized Volatility

Rolling realized volatility is computed from daily market returns:

$$
\text{RV}_{t}^{(L)} =
\sqrt{\frac{252}{L-1}
\sum_{j=0}^{L-1}
\left(r_{m,t-j} - \bar{r}_{m,t}^{(L)}\right)^2},
\quad L \in \{21, 63\}.
$$

The 21-day measure reacts quickly to stress, while the 63-day measure captures persistent volatility regimes.

### 2.3 Amihud Illiquidity

For each ETF, daily dollar volume is

$$
\text{DollarVolume}_{i,t} = P_{i,t} V_{i,t}.
$$

The daily Amihud illiquidity proxy is

$$
\text{ILLIQ}_{i,t}
= \frac{|r_{i,t}|}
{\max(\text{DollarVolume}_{i,t}, \epsilon)},
$$

where \(\epsilon > 0\) prevents division by zero. The cross-sectional market liquidity feature is

$$
\text{ILLIQ}_{t}^{m}=
\log\left(
1 +
\frac{1}{N}\sum_{i=1}^{N} \text{ILLIQ}_{i,t}
\right).
$$

This feature is important because the proposed optimizer raises trading-cost penalties when liquidity conditions deteriorate.

### 2.4 VIX and Credit Spread Features

VIX captures option-implied equity volatility. The high-yield option-adjusted spread captures credit-market stress. The regime model should include levels and short changes:

$$
\text{VIXLevel}_t = \log(\text{VIXCLS}_t),
\quad
\Delta \text{VIX}_{t}^{(21)} =
\log(\text{VIXCLS}_t) - \log(\text{VIXCLS}_{t-21}),
$$

$$
\text{HYSpread}_t = \text{BAMLH0A0HYM2}_t,
\quad
\Delta \text{HYSpread}_{t}^{(21)} =
\text{HYSpread}_t - \text{HYSpread}_{t-21}.
$$

Missing FRED observations should be forward-filled only across market holidays or isolated missing dates. Longer gaps should be dropped from the regime estimation window.

### 2.5 Optional Close Location Value

An optional daily order-pressure proxy is the close location value:

$$
\text{CLV}_{i,t} =
\frac{2C_{i,t} - H_{i,t} - L_{i,t}}
{\max(H_{i,t} - L_{i,t}, \epsilon)}.
$$

It lies approximately in \([-1,1]\) and measures whether the close is near the high or low of the daily range. Because it is noisier than VIX, realized volatility, or credit spreads, it should be treated as optional.

## 3. Feature Standardization and Look-Ahead Control

All features should be standardized using only historical information. For feature \(j\), define an expanding or rolling mean and standard deviation:

$$
Z_{j,t}=
\frac{X_{j,t} - \hat{m}_{j,t}^{\text{train}}}
{\hat{s}_{j,t}^{\text{train}}}.
$$

At date \(t\), \(\hat{m}_{j,t}^{\text{train}}\) and \(\hat{s}_{j,t}^{\text{train}}\) must be estimated from observations no later than \(t\). A practical implementation can use an initial training period of at least 756 trading days and then refit the scaler and HMM monthly using an expanding window. This reduces computation while respecting the time ordering of information.

The final recommended feature vector is

$$
X_t =
\left[
\text{MOM}^{(21)}_t,\,
\text{MOM}^{(63)}_t,\,
\text{DD}^{(63)}_t,\,
\text{RV}^{(21)}_t,\,
\text{RV}^{(63)}_t,\,
\text{ILLIQ}^{m}_t,\,
\text{VIXLevel}_t,\,
\Delta \text{VIX}^{(21)}_t,\,
\text{HYSpread}_t,\,
\Delta \text{HYSpread}^{(21)}_t
\right]^\top.
$$

This vector is intentionally low dimensional. It captures trend, volatility, liquidity, and credit stress without making the HMM fragile.

## 4. Hidden Markov Model Specification

Let \(s_t \in \{1,\dots,K\}\) denote the latent market regime at date \(t\). The HMM assumes

$$
\Pr(s_t = j \mid s_{t-1}=i) = P_{ij},
$$

and

$$
X_t \mid s_t=k \sim \mathcal{N}(m_k, \Omega_k),
$$

where \(P\) is a \(K \times K\) transition matrix, \(m_k\) is the state-specific feature mean, and \(\Omega_k\) is the state-specific feature covariance matrix. The parameters can be estimated by maximum likelihood using the expectation-maximization algorithm implemented in `hmmlearn`.

The default number of states should be \(K=3\). A two-state model may be too coarse because it usually collapses all non-calm periods into one stress state. A four-state model may be useful as a robustness check, but it increases parameter uncertainty. The empirical section should compare \(K \in \{2,3,4\}\) using BIC, state persistence, and out-of-sample portfolio performance; however, the primary specification should be fixed before final performance reporting to avoid data mining.

## 5. Regime Labeling

HMM state labels are arbitrary. After each fit, states should be sorted by a stress score:

$$
\text{StressScore}_k=
\mathbb{E}\left[
Z(\text{RV}^{(63)}_t)+ Z(\text{ILLIQ}^{m}_t)+ Z(\text{VIXLevel}_t)+ Z(\text{HYSpread}_t)- Z(\text{MOM}^{(63)}_t)
\mid s_t=k
\right].
$$

The sorted labels are:

1. Calm or growth regime: low volatility, low spreads, low illiquidity, and positive trend.
2. Transitional regime: intermediate volatility or weakening trend.
3. Stress or illiquid regime: high volatility, wide spreads, high illiquidity, and negative trend.

These labels are descriptive rather than structural. The HMM is not assumed to discover true economic states; it is used to summarize recurring patterns that are useful for conditional portfolio inputs.

## 6. Regime Probability Forecasts

Let

$$
q_{t,k} = \Pr(s_t=k \mid X_1,\dots,X_t)
$$

be the filtered posterior probability after observing date \(t\). The \(h\)-step-ahead regime distribution is

$$
q_{t+h \mid t} = q_t P^h,
\quad h=1,\dots,H.
$$

These probabilities feed the MPC optimizer. The use of probabilities is preferable to hard labels because it produces smoother forecasts of return, risk, and trading costs.

## 7. Regime-Conditioned Return and Risk Estimates

Let \(R_{u+1} \in \mathbb{R}^N\) denote next-day asset returns. To avoid look-ahead bias, the state probability at date \(u\) should be paired with \(R_{u+1}\). For each regime \(k\), define the effective sample size

$$
n_{k,t}^{\text{eff}} = \sum_{u \le t-1} q_{u,k}.
$$

The raw regime-conditioned mean is

$$
\mu_{k,t}^{\text{raw}}=
\frac{
\sum_{u \le t-1} q_{u,k} R_{u+1}
}{
\sum_{u \le t-1} q_{u,k}
}.
$$

The raw regime-conditioned covariance is

$$
S_{k,t}^{\text{raw}}=
\frac{
\sum_{u \le t-1} q_{u,k}
\left(R_{u+1}-\mu_{k,t}^{\text{raw}}\right)
\left(R_{u+1}-\mu_{k,t}^{\text{raw}}\right)^\top
}{
\sum_{u \le t-1} q_{u,k}
}.
$$

Because regime samples can be small, the estimates should be shrunk toward unconditional rolling estimates:

$$
\hat{\mu}_{k,t}=
\alpha_{k,t}\mu_{k,t}^{\text{raw}}+ (1-\alpha_{k,t})\mu_{0,t},
\quad\alpha_{k,t}=\frac{n_{k,t}^{\text{eff}}}{n_{k,t}^{\text{eff}} + n_0},
$$

where \(\mu_{0,t}\) is an unconditional rolling mean and \(n_0\) is a stabilizing constant such as 252. Covariance matrices should be positive semidefinite. A practical shrinkage rule is

$$
\hat{\Sigma}_{k,t}=
(1-\lambda_{k,t})S_{k,t}^{\text{raw}}+ \lambda_{k,t}D_{0,t}+ \eta I,
$$

where \(D_{0,t}\) is the diagonal of an unconditional covariance estimate, \(\lambda_{k,t} \in [0,1]\), and \(\eta > 0\) is a small ridge term. Ledoit-Wolf shrinkage can also be used if implemented carefully.

For horizon \(h\), the regime-mixture expected return is

$$
\hat{\mu}_{t,h}=
\sum_{k=1}^{K} q_{t+h \mid t,k}\hat{\mu}_{k,t}.
$$

The mixture covariance is

$$
\hat{\Sigma}_{t,h}=
\sum_{k=1}^{K}
q_{t+h \mid t,k}
\left[
\hat{\Sigma}_{k,t}
+
\left(\hat{\mu}_{k,t}-\hat{\mu}_{t,h}\right)
\left(\hat{\mu}_{k,t}-\hat{\mu}_{t,h}\right)^\top
\right].
$$

This formula preserves positive semidefiniteness when the component covariances are positive semidefinite.

## 8. State-Dependent Trading-Cost Multipliers

The optimizer should use a linear turnover cost with a regime-dependent multiplier. Let \(a_i^{0}\) be a baseline per-dollar trading-cost coefficient for ETF \(i\). A simple implementation can set all \(a_i^{0}\) equal to 5 basis points:

$$
a_i^0 = 0.0005.
$$

Let \(m_{k,t}\) be the trading-cost multiplier for regime \(k\). A transparent specification is

$$
m_{k,t}=
\max\left[
1,\,
1+ \eta_v \bar{Z}_{k,t}(\text{VIXLevel})+ \eta_l \bar{Z}_{k,t}(\text{ILLIQ}^{m})+ \eta_c \bar{Z}_{k,t}(\text{HYSpread})
\right],
$$

where \(\bar{Z}_{k,t}(\cdot)\) is the posterior-weighted state mean of the standardized feature. The coefficients \(\eta_v,\eta_l,\eta_c\) should be fixed before backtesting. A simpler alternative is to sort regimes by stress score and set

$$
(m_{\text{calm}}, m_{\text{transition}}, m_{\text{stress}})=
(1.0, 1.5, 3.0).
$$

The horizon-specific cost vector passed to the optimizer is

$$
\hat{a}_{t,h,i}=
a_i^0
\sum_{k=1}^{K} q_{t+h \mid t,k}m_{k,t}.
$$

This design preserves convexity because \(\hat{a}_{t,h,i}\) is a nonnegative parameter, not a decision variable.

## 9. Diagnostics

The empirical section should report the following regime diagnostics before presenting portfolio results:

- The estimated transition matrix \(P\).
- Average regime duration implied by \(1/(1-P_{kk})\).
- Time series of posterior regime probabilities.
- Regime means of VIX, realized volatility, credit spread, Amihud illiquidity, and market return.
- Regime distribution during March 2020 and the 2022 inflation-tightening period.
- Sensitivity of results to \(K=2,3,4\).

These diagnostics are necessary because the portfolio results are only credible if the regimes are interpretable and not merely statistical artifacts.

## 10. Handoff to the Optimization Layer

For each rebalancing date \(t\), the regime agent should produce:

$$
\left\{
\hat{\mu}_{t,h},
\hat{\Sigma}_{t,h},
\hat{a}_{t,h}
\right\}_{h=1}^{H}.
$$

The optimization agent treats these arrays as fixed inputs. No HMM state variable appears inside the CVXPY decision problem. This separation is the central design choice that makes the project implementable: the statistical model can be improved or replaced without changing the convex optimization structure.

## References

Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56. https://doi.org/10.1016/S1386-4181(01)00024-6

Ang, A., and Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313-337. https://doi.org/10.1146/annurev-financial-110311-101808

Federal Reserve Bank of St. Louis. CBOE Volatility Index: VIXCLS. FRED. https://fred.stlouisfed.org/series/VIXCLS

Federal Reserve Bank of St. Louis. ICE BofA US High Yield Index Option-Adjusted Spread: BAMLH0A0HYM2. FRED. https://fred.stlouisfed.org/series/BAMLH0A0HYM2

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384. https://doi.org/10.2307/1912559

Ledoit, O., and Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4), 110-119. https://doi.org/10.3905/jpm.2004.110

Nystrup, P., Madsen, H., and Lindstrom, E. (2018). Dynamic portfolio optimization across hidden market regimes. *Quantitative Finance*, 18(1), 83-95. https://doi.org/10.1080/14697688.2017.1342857

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286. https://doi.org/10.1109/5.18626
