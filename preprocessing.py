import pandas as pd
import numpy as np
import yfinance as yf

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

data = data.ffill() # forward fill missing values
weekly_data = data.resample('W-WED').last() #resampling to weekly data
weekly_returns=weekly_data.pct_change(fill_method=None).dropna(how="all")

weekly_returns.to_csv("Final_weekly_returns_data.csv")