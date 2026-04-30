# Regime-Aware MPC Portfolio Optimization

This project studies a regime-aware multi-period portfolio optimization strategy with transaction costs. The goal is to evaluate whether using market-regime information can improve portfolio performance compared with simpler baseline strategies.

The project is adapted from a previous ORIE 5370 finance optimization project and has been reorganized as a reproducible Python package for ORIE 5270.

---

## Project purpose

Portfolio optimization methods often perform poorly in practice when they ignore changing market conditions and transaction costs. This project investigates a regime-aware portfolio construction pipeline:

1. Use real financial and macro-financial data.
2. Build market-regime features from returns, volatility, liquidity, VIX, and credit-spread indicators.
3. Estimate calm, transition, and stress regimes using a Hidden Markov Model.
4. Use regime-dependent expected returns, covariances, and transaction costs as inputs to a multi-period convex optimization problem.
5. Compare the proposed strategy against baseline portfolio strategies.

The main computational focus is on building a clean, reproducible optimization and backtesting pipeline.

---

## Dataset

The full empirical project uses real financial market data.

### Asset data

The asset universe consists of sector ETF data from Yahoo Finance, including adjusted close prices, close prices, and trading volume.

The default asset universe is:

```text
XLE, XLF, XLK, XLY, XLI, XLV, XLP, XLU, XLB