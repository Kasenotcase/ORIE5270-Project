# ORIE5270 Project: Regime-Aware Multi-Period Portfolio Optimization

This repository contains the final project for **ORIE5270: Big Data Technologies**. The project studies how data-driven market regime detection can be integrated with convex portfolio optimization to improve sector ETF allocation over time.

The central idea is to combine historical market data processing, hidden Markov model based regime estimation, convex optimization with turnover and risk controls, rolling-window backtesting, and empirical comparison against standard portfolio baselines.

The project is designed as an applied ORIE5270 implementation exercise: it turns financial time-series data into a reproducible optimization pipeline and evaluates the resulting strategy on out-of-sample market data.

## Project Overview

The portfolio universe consists of nine U.S. sector ETFs. The empirical study uses historical daily return data from 2010 to 2024, with out-of-sample evaluation beginning in 2013.

The main strategy is a **regime-aware multi-period model predictive control portfolio optimizer**. At each rebalance date, the pipeline estimates the current market regime, forecasts regime-dependent return and risk inputs, and solves a constrained convex optimization problem for the next portfolio allocation.

The project compares the proposed strategy with several benchmarks:

- Equal-weight portfolio
- Mean-variance / Markowitz portfolio
- Original HMM-MPC implementation
- V3 Rolling Regime-MPC implementation

The final implementation emphasizes realistic portfolio constraints, including long-only weights, full investment, turnover control, sector diversification, and transaction-cost-aware evaluation.

## Repository Structure

```text
.
├── README.md
└── Proj/
    ├── AGENTS.md
    ├── Blueprint.md
    ├── data/
    ├── docs/
    ├── improvements/
    ├── outputs/
    └── scripts/
```

### Key directories

`Proj/data/` contains the historical market data used by the empirical pipeline.

`Proj/scripts/` contains data processing, baseline strategy, diagnostic, and testing scripts.

`Proj/improvements/` contains the improved V3 rolling regime-MPC implementation and related experimental outputs.

`Proj/outputs/` contains generated backtest results, diagnostic summaries, tables, and figures.

`Proj/docs/` contains the written project documentation, including theory, literature review, regime design, and empirical analysis.

## Methodology

The project follows a modular pipeline.

### 1. Data preparation

Daily sector ETF prices are converted into return series. The pipeline aligns the ETF universe, handles missing data, and prepares rolling training and testing windows for out-of-sample evaluation.

### 2. Market regime modeling

A hidden Markov model is used to identify latent market regimes from historical return behavior. These regimes are interpreted as different market environments with distinct return, volatility, and correlation characteristics.

### 3. Multi-period portfolio optimization

Given regime-conditioned estimates, the optimizer solves a convex portfolio allocation problem. The formulation includes expected return, risk, transaction cost, and turnover considerations.

At a high level, the objective balances return seeking, risk control, transaction-cost reduction, portfolio stability, and diversification.

The optimization layer is implemented with CVXPY.

### 4. Rolling backtest

The strategy is evaluated in a rolling out-of-sample setting. At each rebalance date, only information available up to that date is used. This design is intended to avoid look-ahead bias and to approximate a realistic investment workflow.

### 5. Benchmark comparison

The final analysis compares the proposed regime-aware approach against simpler and commonly used baselines. Performance is evaluated using cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, turnover, and transaction-cost-adjusted results.

## Main Results

The project finds that the improved V3 rolling regime-MPC strategy produces a more stable and economically meaningful allocation process than the initial implementation. The V3 version is designed to address practical issues such as excessive turnover, unstable regime estimates, and sensitivity to noisy short-window return forecasts.
Proj/outputs/
Proj/improvements/
```

## How to Run

The project scripts are organized around the pipeline in `Proj/scripts/` and `Proj/improvements/`.

A typical workflow is:

```bash
python scripts/baseline_strategy.py

# Run the improved rolling regime-MPC experiment
python improvements/v3_rolling_regime_mpc.py
The main Python dependencies are:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `cvxpy`
- `matplotlib`
- `seaborn`
- `yfinance` or another data source package, if data is refreshed

A typical installation command is:

```bash
pip install numpy pandas scipy scikit-learn cvxpy matplotlib seaborn yfinance
```

## Documentation

The project documentation is organized under `Proj/docs/`.

Recommended reading order:

1. Project overview
2. Literature review
3. Theoretical model
4. Market regime design
5. Empirical analysis

These documents explain the motivation, mathematical formulation, regime modeling design, and final empirical findings.

## ORIE5270 Connection

This project was developed for ORIE5270 as a hands-on application of big data technologies to financial decision-making. It emphasizes the full data-to-decision pipeline: collecting and organizing financial time-series data, extracting latent market structure, building reproducible analytical code, solving constrained optimization problems, evaluating results with out-of-sample experiments, and communicating findings through technical documentation.

The goal is not only to build a portfolio strategy, but also to demonstrate how data engineering, statistical learning, optimization, and empirical validation can be combined in a coherent analytics project.

## Notes

This repository is intended for academic and educational use. The strategies and results are not financial advice. Backtest performance does not guarantee future investment performance.
## Dependencies

```

Depending on the local Python environment and data location, some script paths may need to be adjusted. The project was developed as a reproducible course project, so outputs are also included in the repository for reference.
cd Proj

# Run baseline experiments and diagnostics

```text
Proj/docs/

