# Quantitative Approaches to Portfolio Optimisation

This project develops and evaluates systematic portfolio construction strategies by integrating different expected return estimators, covariance estimation techniques, and optimization objectives.

We combine:

- Return estimators: Mean Historical Return, Exponentially Weighted Moving Average (EWMA), and CAPM-based returns.
- Covariance estimators: Sample Covariance, Exponentially Weighted Covariance, and Ledoit–Wolf Shrinkage.
- Optimization objectives: Maximum Sharpe Ratio and Global Minimum Variance (GMV).

Backtests are conducted on NIFTY 500 stocks using a rolling window framework with weekly rebalancing, incorporating transaction costs. Performance is compared against an equally weighted portfolio, the NIFTY 500 index, and a mutual fund benchmark.


## Tools and Packages

- Python: NumPy, Pandas, Matplotlib, Seaborn
- Finance Libraries: yfinance, PyPortfolioOpt, CVXPY
- Version Control: Git & GitHub
