import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return,ema_historical_return,capm_return
from pypfopt.risk_models import sample_cov,exp_cov,CovarianceShrinkage
from new_optimisation_models import global_min_var_cvxpy,max_sharpe_modified_cvxpy
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


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

market_ret=pd.read_csv("equally_weighted_portfolio_returns.csv",index_col="Date",parse_dates=True)

#____________________________________________________________________________________________________

portfolio_values=[1]
dates=weekly_returns.index
prev_weights_dict = None
prev_valid_assets = None

return_rows=[]

window=520
start_index=573

import time
start_time = time.time()

for i in range(start_index, len(weekly_returns)):

    available_assets = weekly_returns.iloc[i].dropna().index.tolist()
    train_data = weekly_returns[available_assets].iloc[i-window:i]
    train_data = train_data.dropna(axis=1,how='any')
    valid_assets = train_data.columns.tolist()
    print(f"number of valid assets in week {i}",len(valid_assets))

    mu = mean_historical_return(train_data,returns_data=True,frequency=1,compounding=True,log_returns=False)
    # mu = ema_historical_return(train_data,returns_data=True,frequency=1,compounding=True,log_returns=False,span=500)
    # mu = capm_return(train_data,market_prices=market_ret,returns_data=True,frequency=1,compounding=True,log_returns=False,risk_free_rate=0.00135) # 7% annualised risk free rate
     
    
    # S = sample_cov(train_data,returns_data=True,frequency=52,log_returns=False)
    # S = exp_cov(train_data,returns_data=True,frequency=52,log_returns=False,span=180)
    S = CovarianceShrinkage(train_data,returns_data=True,frequency=1,log_returns=False).ledoit_wolf()
    

    c_h = np.ones(len(valid_assets)) * 0.01
    lamb = 0.01
    kappa = 2
    allow_short = True
    b_h = 0.0013

    if prev_weights_dict is None:
        w_prev = None
    else:
        prev_weights_series = pd.Series(prev_weights_dict)
        aligned = prev_weights_series.reindex(valid_assets).fillna(0)
        w_prev = aligned.values

    cleaned_weights = global_min_var_cvxpy(mu, S, w_prev=w_prev, b_h=b_h, k=kappa, lamb=lamb,c_h=c_h,allow_short=allow_short, tickers=valid_assets)
    # cleaned_weights = max_sharpe_modified_cvxpy(mu, S, k=kappa, allow_short=allow_short, w_prev=w_prev, lamb=lamb, c_h=c_h, tickers=valid_assets)
    
    
    print(f"got optimal weights for week {i}")

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

    if i + 1 < len(weekly_returns): 
        predicted_mu = mu  # predicted returns for week i+1 

        for stock in prev_valid_assets:
            if stock in predicted_mu:
                predicted = predicted_mu[stock]  # prediction for week i+1
                actual = weekly_returns.iloc[i + 1][stock]  # actual return in week i+1

                return_rows.append({
                    "Date": weekly_returns.index[i + 1],  # date of realized return
                    "Ticker": stock,
                    "predicted_return": predicted,
                    "weekly_returns": actual
                })


end_time = time.time()
print(f"Total runtime for solver: {end_time - start_time:.2f} seconds")
#________________________________________________________________________________________________

pred_vs_actual_df = pd.DataFrame(return_rows)

# pred_vs_actual_df.to_csv("predicted_vs_actual_returns.csv", index=False)
print("\n--- Summary Statistics ---")
print("Predicted returns (μ̂):")
print(pred_vs_actual_df["predicted_return"].describe())

print("\nActual weekly returns (r):")
print(pred_vs_actual_df["weekly_returns"].describe())


actual=pred_vs_actual_df['weekly_returns']
predicted=pred_vs_actual_df['predicted_return']
r2 = r2_score(actual, predicted)

print("R² score:",r2)



plt.figure(figsize=(10, 5))
sns.histplot(pred_vs_actual_df["predicted_return"], color="blue", label="Predicted", kde=True, stat="density")
sns.histplot(pred_vs_actual_df["weekly_returns"], color="orange", label="Actual", kde=True, stat="density")
plt.legend()
plt.title("Distribution of Predicted vs Actual Weekly Returns")
plt.xlabel("Return")
plt.grid(True)
plt.tight_layout()
plt.show()

#_____________________________________________________________________________________________________


# Scatter plot with a regression line
plt.figure(figsize=(8, 6))
sns.regplot(
    x=pred_vs_actual_df['predicted_return'],
    y=pred_vs_actual_df['weekly_returns'],
    line_kws={"color": "red"},
    scatter_kws={"alpha": 0.6}
)
plt.xlabel("Predicted Returns (μ̂)")
plt.ylabel("Actual Returns (r)")
plt.title(f"Predicted vs Actual Weekly Returns (R² = {r2:.2f})")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


#__________________________________________________________________________________________________________________________
portfolio_df = pd.DataFrame({
    'Portfolio Value': portfolio_values
}, index=dates[start_index:])

print(portfolio_df)
# portfolio_df.to_csv("Mean_shrinkage_GMV_portfolio.csv")
