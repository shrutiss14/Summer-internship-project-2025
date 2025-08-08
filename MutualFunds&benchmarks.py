import yfinance as yf
import pandas as pd
import numpy as np
from comparison_metric import sharpe_ratio,compute_cagr_from_weekly_returns,annualised_volatility,max_drawdown_by_vol,sortino_ratio,treynor_ratio,information_ratio

# MOTILALOFS.NS - motilal oswal mutual funds



# ^CRSLDX - nifty 500 index
df=yf.download("MOTILALOFS.NS",start="2014-01-01",end="2025-01-01",auto_adjust=False,progress=True)['Adj Close']

df = df.ffill() # forward fill missing values
weekly_data = df.resample('W-WED').last() #resampling to weekly data
weekly_returns=weekly_data.pct_change(fill_method=None)

weekly_returns = weekly_returns.rename(columns={"MOTILALOFS.NS": "Weekly Return"})

weekly_returns['Portfolio Value'] = (1 + weekly_returns['Weekly Return']).cumprod().fillna(1.0)

weekly_returns.to_csv("Motilal_Oswal.csv")

#_________________________________________________________________________________________________________
portfolio_returns=weekly_returns['Weekly Return']
portfolio_df= pd.read_csv("Motilal_Oswal.csv", usecols=lambda col: col != 'Weekly Return')


market_ret=pd.read_csv("equally_weighted_portfolio_returns.csv",index_col="Date",parse_dates=True)
market_ret=market_ret.squeeze()
market_ret=market_ret.loc[portfolio_returns.index]

cagr = compute_cagr_from_weekly_returns(portfolio_returns)
print(f"CAGR: {cagr:.2%}")

ann_vol=annualised_volatility(portfolio_returns)
print("Annualised volatility",ann_vol)

max_draw=max_drawdown_by_vol(portfolio_df,portfolio_returns)
print("Max Drawdown",max_draw)

sharpe=sharpe_ratio(portfolio_returns)
print("Sharpe:",sharpe)

sortino=sortino_ratio(portfolio_returns)
print("Sortino:",sortino)

treynor=treynor_ratio(portfolio_returns,market_ret)
print("Treynor:",treynor)

inf_rat=information_ratio(portfolio_returns,market_ret)
print("Information Ratio",inf_rat)