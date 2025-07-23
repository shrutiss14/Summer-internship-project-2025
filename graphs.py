import matplotlib.pyplot as plt
import pandas as pd

portfolio_df_max_sharpe=pd.read_csv("Max_sharpe_ratio_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_GMV=pd.read_csv("Global_min_variance_portfolio.csv",index_col="Date",parse_dates=True)
# portfolio_df_cov_shrinkage=pd.read_csv("Covariance Shrinkage Portfolio.csv",index_col="Date",parse_dates=True)
'''equally_weighted_returns=pd.read_csv("equally_weighted_portfolio_returns.csv",index_col="Date",parse_dates=True)
equally_weighted_returns=equally_weighted_returns.loc[portfolio_df_Mean.index]
equally_weighted_returns.loc['2012-12-26', 'Weekly Portfolio Return']=None
equally_weighted_returns['Growth of ₹1'] = (1 + equally_weighted_returns['Weekly Portfolio Return']).cumprod().fillna(1)'''


plt.figure(figsize=(12,6))
plt.plot(portfolio_df_GMV.index, portfolio_df_GMV['Portfolio Value'], label='Global Minimum Variance Portfolio')
plt.plot(portfolio_df_max_sharpe.index, portfolio_df_max_sharpe['Portfolio Value'], label='Max Sharpe ratio Portfolio')
# plt.plot(portfolio_df_cov_shrinkage.index, portfolio_df_cov_shrinkage['Portfolio Value'], label='Covariance Shrinkage Portfolio')
# plt.plot(equally_weighted_returns.index, equally_weighted_returns['Growth of ₹1'], label='Equally weighted portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()