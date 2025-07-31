import matplotlib.pyplot as plt
import pandas as pd

'''portfolio_df_mean_historical_return=pd.read_csv("Mean_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_EWMA=pd.read_csv("EWMA_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_CAPM=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)'''
'''portfolio_df_sample_covar=pd.read_csv("CAPM_sample_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_EWMA_covar=pd.read_csv("CAPM_EWMA_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_shrinkage_covar=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)'''
'''portfolio_df_max_sharpe=pd.read_csv("CAPM_shrinkage_sharpe_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_GMV=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)
portfolio_df_Motilal_Oswal=pd.read_csv("Motilal_Oswal.csv",index_col="Date",parse_dates=True)'''
# portfolio_df_Nifty_500_index=pd.read_csv("Nifty_500_index.csv",index_col="Date",parse_dates=True)


'''portfolio_df_linear=pd.read_csv("linear_portfolio_value.csv",index_col="Date",parse_dates=True)
portfolio_df_lasso=pd.read_csv("lasso_portfolio_value.csv",index_col="Date",parse_dates=True)
portfolio_df_ridge=pd.read_csv("ridge_portfolio_value.csv",index_col="Date",parse_dates=True)'''
portfolio_df_elasticnet=pd.read_csv("elasticnet_portfolio_value.csv",index_col="Date",parse_dates=True)
portfolio_df_GMV=pd.read_csv("CAPM_shrinkage_GMV_portfolio.csv",index_col="Date",parse_dates=True)
'''portfolio_df_DTree=pd.read_csv("dtree_portfolio_value.csv",index_col="Date",parse_dates=True)
portfolio_df_RF=pd.read_csv("rf_portfolio_value.csv",index_col="Date",parse_dates=True)
portfolio_df_xgb=pd.read_csv("xgb_portfolio_value.csv",index_col="Date",parse_dates=True)'''


'''equally_weighted_returns=pd.read_csv("equally_weighted_portfolio_returns.csv",index_col="Date",parse_dates=True)
equally_weighted_returns=equally_weighted_returns.loc[portfolio_df_max_sharpe.index]
equally_weighted_returns = equally_weighted_returns[equally_weighted_returns.index >= '2014-01-01']
equally_weighted_returns.loc['2014-01-01', 'Weekly Portfolio Return']=None
equally_weighted_returns['Portfolio Value'] = (1 + equally_weighted_returns['Weekly Portfolio Return']).cumprod().fillna(1.0)'''


plt.figure(figsize=(12,6))
'''plt.plot(portfolio_df_linear.index, portfolio_df_linear['Portfolio Value'], label='Linear',color="#1f78b4")
plt.plot(portfolio_df_lasso.index, portfolio_df_lasso['Portfolio Value'], label='Lasso',color="#33a02c")
plt.plot(portfolio_df_ridge.index, portfolio_df_ridge['Portfolio Value'], label='Ridge',color="#6a3d9a")'''
plt.plot(portfolio_df_elasticnet.index, portfolio_df_elasticnet['Portfolio Value'], label='Elastic net',color="#ff7f00")

plt.plot(portfolio_df_GMV.index, portfolio_df_GMV['Portfolio Value'], label='Global Minimum Variance',color="#33a02c")
'''plt.plot(portfolio_df_DTree.index, portfolio_df_DTree['Portfolio Value'], label='Decision Tree',color="#e31a1c")
plt.plot(portfolio_df_RF.index, portfolio_df_RF['Portfolio Value'], label='Random Forest',color="#b15928")
plt.plot(portfolio_df_xgb.index, portfolio_df_xgb['Portfolio Value'], label='XG boost',color="#a6cee3")'''

# plt.plot(portfolio_df_Nifty_500_index.index, portfolio_df_Nifty_500_index['Portfolio Value'], label='Nifty 500 index',color="#ff7f00")
# plt.plot(equally_weighted_returns.index, equally_weighted_returns['Portfolio Value'], label='Equally weighted portfolio',color="#e31a1c")
plt.xlabel('Date')
plt.ylabel('Cumulative Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()