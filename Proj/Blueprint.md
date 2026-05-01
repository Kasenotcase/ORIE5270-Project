# Project Blueprint: Regime-Aware Multi-Period Portfolio Optimization

This document describes the design blueprint for the ORIE5270 final project. The project builds a data-driven portfolio optimization pipeline that combines market regime detection, rolling estimation, and constrained convex optimization.

The goal is to demonstrate how financial time-series data can be transformed into a reproducible decision-making system. The project is not only a portfolio backtest, but also an applied data analytics workflow involving data preparation, statistical learning, optimization modeling, and empirical evaluation.

## 1. Project Motivation

Financial markets are not stationary. Expected returns, volatility, and cross-asset correlations can change substantially across different market environments. A static allocation rule may perform well in one period but become unstable or inefficient when the market regime changes.

This project studies whether a regime-aware optimization framework can improve sector ETF allocation. Instead of using one fixed estimate of expected return and covariance, the strategy first estimates latent market regimes and then uses regime-conditioned information in a multi-period portfolio optimization problem.

The project is designed around the following research question:

> Can market regime information improve the stability and out-of-sample performance of a constrained portfolio optimization strategy?

## 2. ORIE5270 Project Framing

This project was developed as an ORIE5270 applied analytics project. It emphasizes the full data-to-decision pipeline:

1. collecting and organizing financial market data;
2. cleaning and transforming price data into return data;
3. estimating latent market states using statistical learning methods;
4. building a constrained optimization model;
5. running rolling out-of-sample experiments;
6. comparing the proposed method against transparent baselines;
7. documenting the methodology and empirical findings.

The project connects naturally to ORIE5270 because it requires practical data processing, reproducible implementation, model validation, and careful communication of results.

## 3. Portfolio Universe and Data

The empirical study focuses on a universe of U.S. sector exchange-traded funds. Sector ETFs provide a useful testbed because they are liquid, economically interpretable, and exposed to different parts of the market.

The portfolio universe contains nine sector ETFs. Daily historical prices are converted into daily returns and aligned into a common time index. The empirical period covers data from 2010 to 2024, with out-of-sample evaluation beginning in 2013.

The data preparation stage includes:

- loading and aligning ETF price data;
- computing daily returns;
- handling missing values;
- separating training and testing windows;
- preparing inputs for regime estimation and optimization.

## 4. Baseline Strategies

The project compares the proposed regime-aware strategy with standard benchmark strategies.

### 4.1 Equal-Weight Portfolio

The equal-weight portfolio assigns the same weight to each ETF. It is simple, transparent, and does not depend on estimated expected returns or covariance matrices.

This benchmark is useful because many sophisticated portfolio strategies fail to outperform equal weighting after estimation error and transaction costs are considered.

### 4.2 Mean-Variance / Markowitz Portfolio

The Markowitz baseline uses historical estimates of expected returns and covariance to solve a mean-variance optimization problem. This strategy provides a classical optimization benchmark.

However, mean-variance optimization can be sensitive to noisy return estimates. This makes it an important comparison point for evaluating whether regime-aware inputs and stronger regularization improve portfolio behavior.

### 4.3 Original HMM-MPC Strategy

The original HMM-MPC implementation combines hidden Markov model regime detection with model predictive control style portfolio optimization. It serves as the first version of the proposed regime-aware framework.

This version is useful for identifying practical issues such as unstable regime estimates, excessive turnover, and sensitivity to noisy short-window forecasts.

### 4.4 V3 Rolling Regime-MPC Strategy

The V3 strategy is the improved implementation. It uses rolling estimation, regime-conditioned inputs, stronger portfolio constraints, and transaction-cost-aware evaluation.

The V3 version is intended to produce more stable allocations and more realistic out-of-sample performance.

## 5. Market Regime Modeling

The project uses a hidden Markov model to estimate latent market regimes from historical return data. Each regime represents a different market environment, such as a lower-volatility growth state, a higher-volatility stress state, or an intermediate transition state.

The regime model is used to estimate:

- the probability of the current market regime;
- regime-dependent expected returns;
- regime-dependent volatility and covariance behavior;
- changes in allocation signals across market environments.

The purpose of the regime model is not to perfectly label the market, but to provide a structured way to adapt the optimization inputs over time.

## 6. Optimization Framework

The core decision problem is a constrained portfolio optimization model. At each rebalance date, the strategy chooses portfolio weights across the ETF universe.

The optimization problem is designed to balance several objectives:

- expected return;
- portfolio risk;
- transaction costs;
- turnover control;
- diversification;
- allocation stability.

The main constraints include:

- long-only portfolio weights;
- full investment;
- upper bounds on individual ETF weights;
- turnover limits or turnover penalties;
- sector diversification controls.

The optimization layer is implemented using CVXPY, which allows the model to be expressed clearly as a convex optimization problem.

## 7. Rolling Backtest Design

The project evaluates the strategies using a rolling out-of-sample backtest. At each rebalance date, the model uses only information available up to that date.

The rolling design is important because it helps avoid look-ahead bias. It also reflects a more realistic investment workflow:

1. use historical data available at the rebalance date;
2. estimate market regimes and model inputs;
3. solve the portfolio optimization problem;
4. hold the portfolio over the next evaluation period;
5. record realized return, turnover, and transaction costs;
6. move the window forward and repeat.

This structure allows the project to evaluate whether the strategy performs well outside the data used for estimation.

## 8. Evaluation Metrics

The empirical analysis evaluates each strategy using both performance and implementation metrics.

The main performance metrics are:

- cumulative return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- maximum drawdown.

The main implementation metrics are:

- turnover;
- transaction-cost-adjusted return;
- allocation stability;
- regime sensitivity;
- interpretability of portfolio weights.

This combination is important because a strategy with high raw return may be unattractive if it requires unrealistic trading or produces unstable allocations.

## 9. Documentation Plan

The written documentation is organized into several parts.

### Literature Review

The literature review discusses portfolio optimization, mean-variance analysis, regime switching models, hidden Markov models, and multi-period portfolio control.

### Theoretical Model

The theoretical model document presents the mathematical formulation of the regime-aware optimization problem, including objective terms and constraints.

### Market Regime Design

The market regime design document explains how latent regimes are estimated, interpreted, and incorporated into the optimization pipeline.

### Empirical Section

The empirical section reports the backtest setup, benchmark comparison, performance metrics, and interpretation of results.

## 10. Implementation Plan

The implementation is organized as a reproducible Python pipeline.

The main stages are:

1. prepare return data from sector ETF prices;
2. implement benchmark strategies;
3. estimate market regimes;
4. construct regime-conditioned optimization inputs;
5. solve portfolio optimization problems with CVXPY;
6. run rolling backtests;
7. save performance tables and figures;
8. summarize results in the project documentation.

The repository separates scripts, outputs, documentation, and improved model versions so that each part of the project can be inspected independently.

## 11. Expected Contribution

The project contributes an applied framework for combining statistical regime detection with constrained portfolio optimization. Its main value is not a claim of guaranteed trading performance, but a reproducible demonstration of how big data technologies and optimization methods can be used together in financial decision-making.

The project highlights several practical lessons:

- portfolio optimization is sensitive to noisy inputs;
- regime information can be useful when combined with regularization;
- transaction costs and turnover control are essential for realistic evaluation;
- rolling out-of-sample testing is necessary for credible empirical analysis;
- transparent baselines are important for interpreting model performance.

## 12. Final Project Positioning

This blueprint frames the project as an ORIE5270 final project from the beginning: a data-driven, reproducible, and empirically evaluated analytics pipeline for financial portfolio optimization.

The final deliverable combines code, data processing, statistical modeling, convex optimization, backtesting, and written analysis. The project is therefore best understood as an applied big-data decision system rather than a purely theoretical portfolio model.

