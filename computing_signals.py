import numpy as np
import pandas as pd

def compute_max_factor(group):
    group['daily_returns']=group['Adj Close'].pct_change(fill_method=None).dropna(how="all")
    group['max_return'] = group['daily_returns'].rolling(window=5).max() # max in 5 days lookback period

    return group


def compute_signals(group):
    group['weekly_returns']=group['Adj Close'].pct_change(fill_method=None).dropna(how="all")


    group['momentum_returns']=group['Adj Close'].pct_change(periods=52)
    group['momentum_returns']=group['momentum_returns'].replace([np.inf, -np.inf], 0)

    group['shifted_data']=group['Adj Close'].shift(periods=4)  
    group['12-1 momentum']=group['shifted_data'].pct_change(periods=48)
    group['12-1 momentum']=group['12-1 momentum'].replace([np.inf, -np.inf], 0)

    group['volatility']=group['weekly_returns'].rolling(window=52).std()
    group['annualised_volatility']=group['volatility']*np.sqrt(52)

    return group

def compute_seasonality_feature(df, y=3):
    df = df.copy()
    df['seasonality_score'] = np.nan  # initialize empty column

    dates = df.index.get_level_values('Date').unique()
    tickers = df.index.get_level_values('Ticker').unique()

    for date in dates:
        for ticker in tickers:
            signals = []
            for i in range(1, y + 1):
                past_date = date - pd.DateOffset(weeks=52 * i)
 
                try:
                    ret = df.loc[(past_date, ticker), 'weekly_returns']
                except KeyError:
                    ret = np.nan

                if pd.isna(ret):
                    signal = 0
                elif ret > 0:
                    signal = 1
                elif ret < 0:
                    signal = -1

                signals.append(signal)

            score = sum(signals)
            if (date, ticker) in df.index:
                df.loc[(date, ticker), 'seasonality_score'] = score

    return df
