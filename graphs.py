import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# return models

'''portfolio_df_mean_historical_return=pd.read_csv("Mean_shrinkage_Sharpe_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_EWMA=pd.read_csv("EWMA_shrinkage_Sharpe_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_CAPM=pd.read_csv("CAPM_shrinkage_Sharpe_portfolio.csv",index_col="Date",parse_dates=True)


portfolio_df_mean_historical_return['portfolio_returns'] = portfolio_df_mean_historical_return['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_EWMA['portfolio_returns'] = portfolio_df_EWMA['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_CAPM['portfolio_returns'] = portfolio_df_CAPM['Portfolio Value'].pct_change(fill_method=None)

portfolio_df_mean_historical_return.dropna(inplace=True)
portfolio_df_EWMA.dropna(inplace=True)
portfolio_df_CAPM.dropna(inplace=True)

vol1=portfolio_df_mean_historical_return['portfolio_returns'].std()*np.sqrt(52)  
vol2=portfolio_df_EWMA['portfolio_returns'].std()*np.sqrt(52)  
vol3=portfolio_df_CAPM['portfolio_returns'].std()*np.sqrt(52)  

target_volatility=0.2
portfolio_df_mean_historical_return['scaled_returns'] = portfolio_df_mean_historical_return['portfolio_returns'] * (target_volatility / vol1)
portfolio_df_EWMA['scaled_returns'] = portfolio_df_EWMA['portfolio_returns'] * (target_volatility / vol2)
portfolio_df_CAPM['scaled_returns'] = portfolio_df_CAPM['portfolio_returns'] * (target_volatility / vol3)

portfolio_df_mean_historical_return['scaled portfolio value'] = (1 + portfolio_df_mean_historical_return['scaled_returns']).cumprod()
portfolio_df_EWMA['scaled portfolio value'] = (1 + portfolio_df_EWMA['scaled_returns']).cumprod()
portfolio_df_CAPM['scaled portfolio value'] = (1 + portfolio_df_CAPM['scaled_returns']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(portfolio_df_mean_historical_return.index, portfolio_df_mean_historical_return['scaled portfolio value'], label='Mean historical return',color="#1f78b4")
plt.plot(portfolio_df_EWMA.index, portfolio_df_EWMA['scaled portfolio value'], label='EWMA historical return',color="#33a02c")
plt.plot(portfolio_df_CAPM.index, portfolio_df_CAPM['scaled portfolio value'], label='CAPM historical return',color='#fb9a99')

plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

# covariance models

'''portfolio_df_sample=pd.read_csv("CAPM_sample_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_EWMA=pd.read_csv("CAPM_EWMA_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_shrinkage=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)


portfolio_df_sample['portfolio_returns'] = portfolio_df_sample['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_EWMA['portfolio_returns'] = portfolio_df_EWMA['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_shrinkage['portfolio_returns'] = portfolio_df_shrinkage['Portfolio Value'].pct_change(fill_method=None)

portfolio_df_sample.dropna(inplace=True)
portfolio_df_EWMA.dropna(inplace=True)
portfolio_df_shrinkage.dropna(inplace=True)

vol1=portfolio_df_sample['portfolio_returns'].std()*np.sqrt(52)  
vol2=portfolio_df_EWMA['portfolio_returns'].std()*np.sqrt(52)  
vol3=portfolio_df_shrinkage['portfolio_returns'].std()*np.sqrt(52)  

target_volatility=0.2
portfolio_df_sample['scaled_returns'] = portfolio_df_sample['portfolio_returns'] * (target_volatility / vol1)
portfolio_df_EWMA['scaled_returns'] = portfolio_df_EWMA['portfolio_returns'] * (target_volatility / vol2)
portfolio_df_shrinkage['scaled_returns'] = portfolio_df_shrinkage['portfolio_returns'] * (target_volatility / vol3)

portfolio_df_sample['scaled portfolio value'] = (1 + portfolio_df_sample['scaled_returns']).cumprod()
portfolio_df_EWMA['scaled portfolio value'] = (1 + portfolio_df_EWMA['scaled_returns']).cumprod()
portfolio_df_shrinkage['scaled portfolio value'] = (1 + portfolio_df_shrinkage['scaled_returns']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(portfolio_df_sample.index, portfolio_df_sample['scaled portfolio value'], label='Sample covariance',color="#1f78b4")
plt.plot(portfolio_df_EWMA.index, portfolio_df_EWMA['scaled portfolio value'], label='EW covariance',color="#33a02c")
plt.plot(portfolio_df_shrinkage.index, portfolio_df_shrinkage['scaled portfolio value'], label='Covariance Shrinkage',color='#fb9a99')

plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

# obj functions

'''portfolio_df_sharpe=pd.read_csv("CAPM_shrinkage_sharpe_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_GMV=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)


portfolio_df_sharpe['portfolio_returns'] = portfolio_df_sharpe['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_GMV['portfolio_returns'] = portfolio_df_GMV['Portfolio Value'].pct_change(fill_method=None)

portfolio_df_sharpe.dropna(inplace=True)
portfolio_df_GMV.dropna(inplace=True)

vol1=portfolio_df_sharpe['portfolio_returns'].std()*np.sqrt(52)  
vol2=portfolio_df_GMV['portfolio_returns'].std()*np.sqrt(52)  

target_volatility=0.2
portfolio_df_sharpe['scaled_returns'] = portfolio_df_sharpe['portfolio_returns'] * (target_volatility / vol1)
portfolio_df_GMV['scaled_returns'] = portfolio_df_GMV['portfolio_returns'] * (target_volatility / vol2)

portfolio_df_sharpe['scaled portfolio value'] = (1 + portfolio_df_sharpe['scaled_returns']).cumprod()
portfolio_df_GMV['scaled portfolio value'] = (1 + portfolio_df_GMV['scaled_returns']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(portfolio_df_sharpe.index, portfolio_df_sharpe['scaled portfolio value'], label='Max Sharpe',color="#1f78b4")
plt.plot(portfolio_df_GMV.index, portfolio_df_GMV['scaled portfolio value'], label='Global Minimum Variance',color="#33a02c")

plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''


# benchmark comparisons
portfolio_df_sharpe=pd.read_csv("CAPM_shrinkage_sharpe_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_GMV=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_nifty_500=pd.read_csv("Nifty_500_index.csv",index_col="Date",parse_dates=True)
portfolio_df_mutual_fund=pd.read_csv("Motilal_Oswal.csv",index_col="Date",parse_dates=True)


portfolio_df_sharpe['portfolio_returns'] = portfolio_df_sharpe['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_GMV['portfolio_returns'] = portfolio_df_GMV['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_nifty_500['portfolio_returns'] = portfolio_df_nifty_500['Portfolio Value'].pct_change(fill_method=None)
portfolio_df_mutual_fund['portfolio_returns'] = portfolio_df_mutual_fund['Portfolio Value'].pct_change(fill_method=None)

portfolio_df_sharpe.dropna(inplace=True)
portfolio_df_GMV.dropna(inplace=True)
portfolio_df_nifty_500.dropna(inplace=True)
portfolio_df_mutual_fund.dropna(inplace=True)

vol1=portfolio_df_sharpe['portfolio_returns'].std()*np.sqrt(52)  
vol2=portfolio_df_GMV['portfolio_returns'].std()*np.sqrt(52)  
vol3=portfolio_df_nifty_500['portfolio_returns'].std()*np.sqrt(52)  
vol5=portfolio_df_mutual_fund['portfolio_returns'].std()*np.sqrt(52)  

target_volatility=0.2

portfolio_df_sharpe['scaled_returns'] = portfolio_df_sharpe['portfolio_returns'] * (target_volatility / vol1)
portfolio_df_GMV['scaled_returns'] = portfolio_df_GMV['portfolio_returns'] * (target_volatility / vol2)
portfolio_df_nifty_500['scaled_returns'] = portfolio_df_nifty_500['portfolio_returns'] * (target_volatility / vol3)
portfolio_df_mutual_fund['scaled_returns'] = portfolio_df_mutual_fund['portfolio_returns'] * (target_volatility / vol5)

portfolio_df_sharpe['scaled portfolio value'] = (1 + portfolio_df_sharpe['scaled_returns']).cumprod()
portfolio_df_GMV['scaled portfolio value'] = (1 + portfolio_df_GMV['scaled_returns']).cumprod()
portfolio_df_nifty_500['scaled portfolio value'] = (1 + portfolio_df_nifty_500['scaled_returns']).cumprod()
portfolio_df_mutual_fund['scaled portfolio value'] = (1 + portfolio_df_mutual_fund['scaled_returns']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(portfolio_df_sharpe.index, portfolio_df_sharpe['scaled portfolio value'], label='Max Sharpe',color="#1f78b4")
plt.plot(portfolio_df_GMV.index, portfolio_df_GMV['scaled portfolio value'], label='Global Minimum Variance',color="#33a02c")
plt.plot(portfolio_df_nifty_500.index, portfolio_df_nifty_500['scaled portfolio value'], label='Nifty 500 index',color="#fdbf6f")
plt.plot(portfolio_df_mutual_fund.index, portfolio_df_mutual_fund['scaled portfolio value'], label='Motilal Oswal Mutual Fund',color="#cab2d6")

plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




'''
portfolio_df_sample_covar=pd.read_csv("CAPM_sample_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_EWMA_covar=pd.read_csv("CAPM_EWMA_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_shrinkage_covar=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)


portfolio_df_max_sharpe=pd.read_csv("CAPM_shrinkage_sharpe_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_GMV=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_Motilal_Oswal=pd.read_csv("Motilal_Oswal.csv",index_col="Date",parse_dates=True)

portfolio_df_Nifty_500_index=pd.read_csv("Nifty_500_index.csv",index_col="Date",parse_dates=True)


equally_weighted_returns=pd.read_csv("equally_weighted_portfolio_returns.csv",index_col="Date",parse_dates=True)
equally_weighted_returns=equally_weighted_returns.loc[portfolio_df_max_sharpe.index]
equally_weighted_returns = equally_weighted_returns[equally_weighted_returns.index >= '2014-01-01']
equally_weighted_returns.loc['2014-01-01', 'Weekly Portfolio Return']=None
equally_weighted_returns['Portfolio Value'] = (1 + equally_weighted_returns['Weekly Portfolio Return']).cumprod().fillna(1.0)


plt.figure(figsize=(12,6))
plt.plot(portfolio_df_linear.index, portfolio_df_linear['Portfolio Value'], label='Linear',color="#1f78b4")
plt.plot(portfolio_df_lasso.index, portfolio_df_lasso['Portfolio Value'], label='Lasso',color="#33a02c")
plt.plot(portfolio_df_ridge.index, portfolio_df_ridge['Portfolio Value'], label='Ridge',color="#6a3d9a")

plt.plot(portfolio_df_DTree.index, portfolio_df_DTree['Portfolio Value'], label='Decision Tree',color="#e31a1c")
plt.plot(portfolio_df_RF.index, portfolio_df_RF['Portfolio Value'], label='Random Forest',color="#b15928")
plt.plot(portfolio_df_xgb.index, portfolio_df_xgb['Portfolio Value'], label='XG boost',color="#a6cee3")

# plt.plot(portfolio_df_Nifty_500_index.index, portfolio_df_Nifty_500_index['Portfolio Value'], label='Nifty 500 index',color="#ff7f00")
# plt.plot(equally_weighted_returns.index, equally_weighted_returns['Portfolio Value'], label='Equally weighted portfolio',color="#e31a1c")
plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''