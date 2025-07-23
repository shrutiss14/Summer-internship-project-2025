import pandas as pd
import numpy as np
import yfinance as yf
from comparison_metric import compute_cagr_from_weekly_returns,sharpe_ratio

#_________________________________________________________________________________________________

url="https://en.wikipedia.org/wiki/NIFTY_500"
tables = pd.read_html(url)
tickers_table = tables[4]
tickers_table.columns = tickers_table.iloc[0]
tickers_table = tickers_table.drop(index=0).reset_index(drop=True)
tickers = [ticker + '.NS' for ticker in tickers_table['Symbol'].dropna()]
data=yf.download(tickers,start="2003-01-01",end="2025-01-01",auto_adjust=False,progress=True)['Adj Close']

summary = []
for ticker in data.columns:
    series=data[ticker].dropna()
    start=series.index.min()
    end=series.index.max()
    n_years = (end - start).days / 365.25
    summary.append({
                "Ticker": ticker,
                "Start Date": start.date(),
                "End Date": end.date(),
                "Years of Data": round(n_years, 2)
            })
summary_df = pd.DataFrame(summary)
tickers_to_keep = summary_df[summary_df["Years of Data"] >15]["Ticker"].tolist()
data=data[tickers_to_keep]

#______________________________________________________________________________________________________

data = data.ffill() # forward fill missing values
weekly_data = data.resample('W-WED').last() #resampling to weekly data
weekly_returns=weekly_data.pct_change(fill_method=None).dropna(how="all")

#_______________________________________________________________________________________________________

portfolio_values=[1]
dates=weekly_returns.index
prev_weights_dict = None
prev_valid_assets = None

window=520
import time
start_time = time.time()

for i in range(0, len(weekly_returns)):

    if i>=520:    
        available_assets = weekly_returns.iloc[i].dropna().index.tolist()
        train_data = weekly_returns[available_assets].iloc[i-window:i]
        train_data = train_data.dropna(axis=1,how='any')
        valid_assets = train_data.columns.tolist()
        print(f"number of valid assets in week {i}",len(valid_assets))  
        n_assets=len(valid_assets)
    else:
        available_assets=weekly_returns.iloc[520].dropna().index.tolist()
        train_data = weekly_returns[available_assets].iloc[0:520]
        train_data = train_data.dropna(axis=1,how='any')
        valid_assets = train_data.columns.tolist()
        print(f"number of valid assets in week {i}",len(valid_assets))  
        n_assets=len(valid_assets)      

    c_h = np.ones(len(valid_assets)) * 0.01   


    if prev_weights_dict is None:
        w_prev = None
    else:
        prev_weights_series = pd.Series(prev_weights_dict)
        aligned = prev_weights_series.reindex(valid_assets).fillna(0)
        w_prev = aligned.values

    
    cleaned_weights= {ticker: 1/n_assets for ticker in valid_assets}

    if prev_weights_dict is not None:
        returns=weekly_returns.iloc[i][prev_valid_assets]
        portfolio_return = np.dot(prev_weights_series.values, returns.values)

        drifted_values = prev_weights_series * (1 + returns)
        if drifted_values.sum() != 0:
            drifted_weights_series = drifted_values / drifted_values.sum()
        else: 
            drifted_weights_series = pd.Series(0, index=prev_valid_assets)

        target_weights_series = pd.Series(cleaned_weights, index=valid_assets)

        all_assets = drifted_weights_series.index.union(target_weights_series.index)
        w_drifted = drifted_weights_series.reindex(all_assets).fillna(0)
        w_target = target_weights_series.reindex(all_assets).fillna(0)

        weight_diff = np.abs(w_target.values - w_drifted.values)
        per_ticker_costs = weight_diff * c_h
        transaction_cost = np.sum(per_ticker_costs)

        net_portfolio_return = portfolio_return - transaction_cost

        portfolio_values.append(portfolio_values[-1] * (1 + net_portfolio_return))

    prev_weights_dict=cleaned_weights
    prev_valid_assets=valid_assets


end_time = time.time()
print(f"Total runtime for solver: {end_time - start_time:.2f} seconds")


#__________________________________________________________________________________________________________________________
portfolio_df = pd.DataFrame({
    'Portfolio Value': portfolio_values
}, index=dates[0:])

print(portfolio_df)

portfolio_df.to_csv("equally_weighted_portfolio_prices.csv")

portfolio_returns = portfolio_df.pct_change(fill_method=None).dropna()
portfolio_returns.columns = ['Weekly Portfolio Return']
portfolio_returns=portfolio_returns.squeeze()
print(portfolio_returns)

portfolio_returns.to_csv("equally_weighted_portfolio_returns.csv", header=True)