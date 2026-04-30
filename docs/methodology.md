\# Methodology



\## Overview



This project studies a regime-aware multi-period portfolio optimization strategy with transaction costs.



The pipeline has five main steps:



1\. Clean and align financial and macro-financial data.

2\. Build market-regime features.

3\. Estimate market regimes using a Hidden Markov Model.

4\. Construct regime-dependent expected returns, covariances, and transaction costs.

5\. Run portfolio optimization and backtesting.



\## Feature engineering



The regime model uses features designed to capture market trend, volatility, liquidity, and macro stress.



The main features include:



\- 21-day momentum

\- 63-day momentum

\- 63-day drawdown

\- 21-day realized volatility

\- 63-day realized volatility

\- Amihud illiquidity

\- log VIX

\- 21-day VIX change

\- high-yield spread

\- 21-day high-yield spread change



The relevant function is `regime\_mpc.features.build\_regime\_features`.



\## Regime modeling



The project uses a Gaussian Hidden Markov Model with three states.



The three states are interpreted as:



\- calm

\- transition

\- stress



After fitting the HMM, states are ordered using a stress score based on volatility, illiquidity, VIX, credit spread, and momentum.



The relevant function is `regime\_mpc.regimes.fit\_hmm\_regime\_inputs`.



\## Portfolio optimization



The project includes two main optimization routines.



\## Markowitz baseline



The Markowitz baseline solves a long-only mean-variance optimization problem with a maximum weight constraint.



The relevant function is `regime\_mpc.optimization.solve\_markowitz`.



\## Regime-aware MPC



The proposed strategy uses regime-dependent expected returns, covariances, and transaction costs in a multi-period optimization problem.



The optimizer includes:



\- long-only constraints

\- full-investment constraint

\- maximum asset weight constraint

\- turnover constraint

\- linear transaction costs

\- quadratic trade penalty



The relevant function is `regime\_mpc.optimization.solve\_mpc`.



\## Backtesting



The backtesting engine simulates daily portfolio returns with periodic rebalancing.



It tracks:



\- daily net returns

\- daily turnover

\- daily transaction costs

\- portfolio weights



The relevant function is `regime\_mpc.backtest.run\_backtest`.



\## Performance metrics



The project reports:



\- annualized return

\- annualized volatility

\- Sharpe ratio

\- maximum drawdown

\- final wealth

\- average turnover

\- total transaction cost



The relevant function is `regime\_mpc.metrics.performance\_table`.

