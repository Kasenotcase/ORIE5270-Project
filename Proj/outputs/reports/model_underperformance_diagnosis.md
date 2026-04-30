# Diagnosis of HMM-MPC Underperformance

## Executive Diagnosis

The proposed HMM-MPC strategy is not a bad result, but it does not dominate the equal-weight benchmark. The evidence points to four main reasons.

First, the data set is clean enough for this stage. The underperformance is not explained by missing prices, missing macro data, zero volume, or obvious data corruption. The only absolute daily return above 20% is documented in `diagnostic_large_return_outliers.csv`.

Second, the asset universe itself is unusually friendly to equal weighting. From 2013 to 2024, several growth and defensive sector ETFs had strong realized performance, and a simple diversified 1/N allocation captured that without estimation error. The HMM-MPC model beat equal weight in 5 out of 12 calendar years, while it beat single-period Markowitz in 8 out of 12 years.

Third, the current optimizer still relies on rolling historical mean estimates. The cross-sectional return-forecast diagnostics show weak predictive content. When expected-return estimates are noisy, a constrained optimizer can easily underweight assets that later perform well. This is a classic mean-variance problem rather than only an HMM problem.

Fourth, the current HMM-MPC implementation is conservative and risk/cost-aware. It improves on single-period Markowitz by reducing turnover and transaction costs, but the same conservatism prevents it from fully participating in the strongest sectors during long bull periods.

## 1. Data Quality

The current evidence does not indicate a fatal data issue.


| date       | asset | daily_return |
| ---------- | ----- | ------------ |
| 2020-03-09 | XLE   | -0.201411    |


The strongest data-related caution is not missingness, but source reproducibility. `yfinance.download` failed in this environment, so the script used the Yahoo Finance chart endpoint fallback. The downloaded data are complete for the selected ETFs, but the final paper should disclose this fallback and keep the cached CSVs for reproducibility.

## 2. Historical Asset Performance

The sector ETF universe creates a difficult benchmark for any estimated optimizer. Equal weight automatically holds the long-run winners and keeps rebalancing into them without making forecasts. The strongest assets over the full sample include XLK, XLY, XLV, and XLI, while XLE had weak long-run performance and very large drawdowns.


| asset | annualized_return | annualized_volatility | sharpe   | max_drawdown | daily_skew  | daily_kurtosis | min_daily_return | max_daily_return |
| ----- | ----------------- | --------------------- | -------- | ------------ | ----------- | -------------- | ---------------- | ---------------- |
| XLB   | 0.0855409         | 0.208637              | 0.409999 | -0.372742    | -0.277873   | 6.84542        | -0.110084        | 0.117601         |
| XLE   | 0.0597582         | 0.274371              | 0.217801 | -0.712628    | -0.383452   | 13.0069        | -0.201411        | 0.160374         |
| XLF   | 0.117871          | 0.221379              | 0.532442 | -0.428625    | -0.223266   | 11.3896        | -0.137093        | 0.131566         |
| XLI   | 0.129534          | 0.194474              | 0.666071 | -0.423341    | -0.322824   | 10.4004        | -0.113441        | 0.126512         |
| XLK   | 0.182509          | 0.21289               | 0.857296 | -0.335591    | -0.221715   | 8.77198        | -0.13814         | 0.117319         |
| XLP   | 0.104201          | 0.137051              | 0.760309 | -0.245123    | -0.29783    | 13.8339        | -0.0939562       | 0.0851065        |
| XLU   | 0.0990361         | 0.175881              | 0.563087 | -0.360668    | -0.00195531 | 15.796         | -0.113577        | 0.127934         |
| XLV   | 0.122126          | 0.160941              | 0.758823 | -0.284043    | -0.269216   | 7.76417        | -0.09861         | 0.0770572        |
| XLY   | 0.158216          | 0.200557              | 0.788887 | -0.396696    | -0.553554   | 7.43654        | -0.126686        | 0.0937963        |


The HMM-MPC average weight deviations from equal weight are:


| asset | HMM-MPC average weight | difference from equal weight |
| ----- | ---------------------- | ---------------------------- |
| XLB   | 0.18%                  | -10.93%                      |
| XLI   | 1.39%                  | -9.72%                       |
| XLE   | 3.41%                  | -7.70%                       |
| XLF   | 3.51%                  | -7.60%                       |
| XLY   | 9.03%                  | -2.09%                       |
| XLK   | 13.11%                 | 1.98%                        |
| XLU   | 17.20%                 | 6.09%                        |
| XLV   | 22.67%                 | 11.56%                       |
| XLP   | 29.51%                 | 18.40%                       |


The most negative contribution differences versus equal weight were:


| asset | EqualWeight_weekly | HMM_MPC_CVXPY_weekly | Markowitz_CVXPY_weekly | hmm_minus_ew_contribution |
| ----- | ------------------ | -------------------- | ---------------------- | ------------------------- |
| XLF   | 0.0162377          | 0.00104845           | 0.00361823             | -0.0151892                |
| XLI   | 0.0154685          | 0.000624719          | -0.00136079            | -0.0148437                |
| XLB   | 0.0117462          | -0.000855467         | -0.000454435           | -0.0126017                |
| XLE   | 0.010316           | 0.000334954          | 0.00154194             | -0.00998108               |
| XLY   | 0.017772           | 0.013914             | 0.0132258              | -0.00385805               |


The subperiod pattern is also informative:


| period               | EqualWeight_weekly_ann_return | Markowitz_CVXPY_weekly_ann_return | HMM_MPC_CVXPY_weekly_ann_return | HMM_minus_EqualWeight_ann_return | HMM_minus_Markowitz_ann_return |
| -------------------- | ----------------------------- | --------------------------------- | ------------------------------- | -------------------------------- | ------------------------------ |
| 2013-2016            | 0.130444                      | 0.124048                          | 0.141474                        | 0.0110297                        | 0.0174263                      |
| 2017-2019            | 0.126555                      | 0.130744                          | 0.146926                        | 0.020371                         | 0.0161826                      |
| 2020_crisis_recovery | 0.100315                      | 0.125648                          | 0.105606                        | 0.00529032                       | -0.0200423                     |
| 2021                 | 0.296042                      | 0.213671                          | 0.202429                        | -0.0936133                       | -0.0112425                     |
| 2022_inflation_bear  | -0.051087                     | -0.0800355                        | -0.0793569                      | -0.0282698                       | 0.000678604                    |
| 2023-2024            | 0.149282                      | 0.0326523                         | 0.0388704                       | -0.110412                        | 0.00621808                     |


The proposed strategy was competitive before 2020 and improved on equal weight in the 2013-2016 and 2017-2019 subperiods. The overall gap is mainly created after 2021, especially in 2023-2024, when equal weight benefited from broad equity recovery and strong growth-sector performance while the optimizer remained heavily defensive.

## 3. Model Assumptions

The current model makes several strong assumptions:

- The HMM state inferred from market-level features is assumed to be useful for sector-level expected returns.
- State-conditional means are estimated from a relatively short rolling sample.
- Gaussian diagonal HMM emissions are assumed to summarize nonlinear market stress dynamics.
- The optimizer treats estimated means as reliable enough to tilt away from equal weight.
- The cost multiplier increases in stress states, but the strategy has no explicit cash or bond asset, so it can only rotate within equities.

These assumptions are individually reasonable for a first implementation, but together they create a conservative equity-sector rotation model with weak alpha forecasts.

## 4. Forecast Quality

The forecast-quality table evaluates whether rolling historical means predict future cross-sectional ETF returns. Low or unstable information coefficients imply that the optimizer's mean vector is mostly noise.


| rolling_window | future_horizon_days | mean_spearman_ic | sign_hit_rate | annualized_top_minus_bottom_realized |
| -------------- | ------------------- | ---------------- | ------------- | ------------------------------------ |
| 252            | 1                   | 0.0126313        | 0.523758      | 0.0971447                            |
| 504            | 1                   | 0.0180447        | 0.528884      | 0.0321503                            |
| 756            | 1                   | 0.0143865        | 0.536332      | 0.010571                             |
| 252            | 5                   | 0.0143297        | 0.544203      | 0.0888479                            |
| 504            | 5                   | 0.0207137        | 0.558612      | 0.0527129                            |
| 756            | 5                   | 0.00888002       | 0.571907      | -0.00234019                          |
| 252            | 21                  | 0.00292153       | 0.567946      | 0.0836864                            |
| 504            | 21                  | 0.00167202       | 0.594034      | 0.0490127                            |
| 756            | 21                  | 0.00904762       | 0.623165      | -0.0169462                           |


This is probably the most important empirical finding. The optimizer is convex and numerically stable, but the expected-return input is not strong enough to beat an equal-weight benchmark over this asset universe.

## 5. Regime-Conditioned Performance


| regime     | observations | EqualWeight_weekly_ann_return | EqualWeight_weekly_vol | EqualWeight_weekly_avg_turnover | EqualWeight_weekly_total_cost | Markowitz_CVXPY_weekly_ann_return | Markowitz_CVXPY_weekly_vol | Markowitz_CVXPY_weekly_avg_turnover | Markowitz_CVXPY_weekly_total_cost | HMM_MPC_CVXPY_weekly_ann_return | HMM_MPC_CVXPY_weekly_vol | HMM_MPC_CVXPY_weekly_avg_turnover | HMM_MPC_CVXPY_weekly_total_cost | HMM_minus_EqualWeight_ann_return | HMM_minus_Markowitz_ann_return |
| ---------- | ------------ | ----------------------------- | ---------------------- | ------------------------------- | ----------------------------- | --------------------------------- | -------------------------- | ----------------------------------- | --------------------------------- | ------------------------------- | ------------------------ | --------------------------------- | ------------------------------- | -------------------------------- | ------------------------------ |
| calm       | 1576         | 0.127641                      | 0.0955541              | 0.00103714                      | 0.00163453                    | 0.094789                          | 0.0876225                  | 0.00812354                          | 0.0128027                         | 0.0945515                       | 0.0893929                | 0.00363125                        | 0.0058291                       | -0.03309                         | -0.000237526                   |
| stress     | 508          | 0.200761                      | 0.280063               | 0.00144141                      | 0.000732235                   | 0.134646                          | 0.253168                   | 0.0118107                           | 0.00599981                        | 0.177958                        | 0.26523                  | 0.00422515                        | 0.00631506                      | -0.0228033                       | 0.0433118                      |
| transition | 929          | 0.0842757                     | 0.165074               | 0.00134218                      | 0.00124688                    | 0.0844923                         | 0.143651                   | 0.00714419                          | 0.00663695                        | 0.0889139                       | 0.144559                 | 0.00379517                        | 0.00526851                      | 0.00463819                       | 0.00442159                     |


The regime-aware model is most defensible when compared with single-period Markowitz. It lowers turnover and costs, and its performance is less fragile than an optimizer that repeatedly jumps to a one-period target. However, the regime model does not automatically create excess return versus equal weight.

## 6. Implementation Issues

The implementation is broadly consistent with the theoretical design: the HMM is fit using only historical features, regime probabilities are external inputs, and the MPC problem is solved with CVXPY. There are nevertheless implementation choices that may suppress performance:

- The same risk-aversion parameter is used for Markowitz and HMM-MPC; this may make the proposed model too conservative.
- The strategy is very close to single-period Markowitz in realized returns. The return correlation between HMM-MPC and Markowitz is 0.989, while the average weight distance from equal weight is about 0.549. This indicates that the current HMM-MPC behaves mainly like a stabilized defensive mean-variance optimizer, not yet like a distinct regime-alpha model.
- The trading-cost multiplier is deliberately high in stress states, which reduces turnover but can also slow recovery-period reallocation.
- The HMM is refit at every rebalance, which is conservative but may produce unstable state labels even after sorting by stress score.
- The model has no explicit shrinkage of expected returns toward economically motivated priors such as equal weight, momentum, or volatility timing.
- The objective uses daily arithmetic return and covariance estimates. This is mathematically valid, but the practical aggressiveness depends heavily on gamma and cost calibration. With the current gamma, the optimizer strongly prefers low-volatility defensive sectors.

## 7. Most Promising Improvement Directions

The most promising directions are:

1. Replace raw rolling mean forecasts with more robust expected-return inputs. Candidate signals include 12-1 momentum, short-term reversal control, volatility-adjusted momentum, or shrinkage toward zero/equal-weight implied returns.
2. Add a no-alpha or weak-alpha version of MPC. In this version, expected returns are set equal across assets or heavily shrunk, so the model mainly optimizes risk and trading costs. This directly tests whether the current underperformance comes from noisy means.
3. Tune risk aversion and trading-cost multipliers on a validation period only. The current parameters are defensible but not calibrated. A small grid over gamma, turnover cap, and stress-cost multipliers could improve the risk-return trade-off without data mining the final test set.
4. Add a defensive asset or cash proxy. A sector-only long-only universe cannot truly de-risk in stress regimes; it can only rotate among risky equity sectors. Adding SHY, IEF, or BIL would make the market-regime signal more actionable.
5. Use regime probabilities mainly for risk and cost, not expected returns. The current regime-conditioned mean estimates are noisy. A more robust design may let HMM drive covariance and transaction-cost penalties while expected returns come from simpler momentum or remain shrunk.
6. Report the equal-weight result honestly. Equal weight is a strong benchmark in this data set. The final paper should frame the proposed model as an improvement over single-period Markowitz in turnover/cost control, while acknowledging that it does not outperform 1/N in total return over 2013-2024.

## Generated Diagnostic Tables

- `outputs/tables/diagnostic_yearly_strategy_returns.csv`
- `outputs/tables/diagnostic_subperiod_performance.csv`
- `outputs/tables/diagnostic_average_weights.csv`
- `outputs/tables/diagnostic_asset_contributions.csv`
- `outputs/tables/diagnostic_hmm_vs_ew_contribution_delta.csv`
- `outputs/tables/diagnostic_regime_performance.csv`
- `outputs/tables/diagnostic_forecast_quality.csv`
- `outputs/tables/diagnostic_large_return_outliers.csv`
- `outputs/tables/diagnostic_strategy_similarity.csv`
- `outputs/tables/diagnostic_strategy_return_correlation.csv`

