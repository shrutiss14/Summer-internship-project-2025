import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt.risk_models import sample_cov,exp_cov,CovarianceShrinkage
from new_optimisation_models import global_min_var_cvxpy,max_sharpe_modified_cvxpy
from computing_signals import compute_max_factor,compute_seasonality_feature,compute_signals
from sklearn.model_selection import GridSearchCV
from models import model_configs
from comparison_metric import compute_cagr_from_weekly_returns,sharpe_ratio
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

url="https://en.wikipedia.org/wiki/NIFTY_500"
tables = pd.read_html(url)
tickers_table = tables[4]
tickers_table.columns = tickers_table.iloc[0]
tickers_table = tickers_table.drop(index=0).reset_index(drop=True)
tickers = [ticker + '.NS' for ticker in tickers_table['Symbol'].dropna()]
data=yf.download(tickers,start="2003-01-01",end="2025-01-01",auto_adjust=False,progress=True)['Adj Close']
print("data downloaded")

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

long_df=data.reset_index().melt(id_vars=['Date'],var_name="Ticker",value_name="Adj Close")
long_df = long_df.set_index(['Date','Ticker']).sort_index()

long_df=long_df.groupby(level='Ticker',group_keys=False).apply(compute_max_factor)

print("midway computing signals")

weekly_data = (
    long_df.groupby(level='Ticker')
        .resample('W-WED', level='Date')
        .last()
)

weekly_data = weekly_data.swaplevel('Ticker', 'Date').sort_index()
weekly_data=weekly_data.groupby(level='Ticker',group_keys=False).apply(compute_signals)

weekly_data = compute_seasonality_feature(weekly_data, y=5)

print("signals computed")

for ticker, group in weekly_data.groupby('Ticker'):
    weekly_data.loc[group.index, 'momentum_returns(lag_1)'] = group['momentum_returns'].shift(1)
    weekly_data.loc[group.index, '12-1 momentum(lag_1)'] = group['12-1 momentum'].shift(1)
    weekly_data.loc[group.index, 'annualised_volatility(lag_1)'] = group['annualised_volatility'].shift(1)
    weekly_data.loc[group.index, 'max_return(lag_1)'] = group['max_return'].shift(1)
    weekly_data.loc[group.index, 'seasonality_score(lag_1)'] = group['seasonality_score'].shift(1)


final_df = weekly_data.drop(columns=['Adj Close','daily_returns','momentum_returns','shifted_data', '12-1 momentum','volatility', 'annualised_volatility','max_return', 'seasonality_score',])
final_df.dropna(inplace=True)

print("final df ready")
#_____________________________________________________________________

def get_valid_assets(df, current_date, min_weeks=468):
    """
    Returns list of tickers that have at least `min_weeks` of historical returns 
    before the given current_date (continuous data assumption).
    """
    current_date = pd.Timestamp(current_date)
    valid_assets = []

    all_tickers = df.index.get_level_values('Ticker').unique()

    for ticker in all_tickers:
        ticker_data = df.xs(ticker, level='Ticker')
        ticker_data = ticker_data[ticker_data.index < current_date]

        # Check if enough past data exists
        if ticker_data['weekly_returns'].count() >= min_weeks:
            # Verify continuous time span
            if ticker_data.index.max() - ticker_data.index.min() >= pd.Timedelta(weeks=min_weeks):
                valid_assets.append(ticker)

    return valid_assets


def walk_forward_predictions_dynamic(df, features, target, start_date, end_date, min_weeks=520,model=None,param_grid=None):
    """
    Walk-forward predictions with dynamic asset eligibility based on 10-year history.
    """
    dates = df.index.get_level_values('Date').unique()
    dates = dates[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]

    all_preds = []

    for current_date in dates:
        valid_assets = get_valid_assets(df, current_date, min_weeks=min_weeks)
        current_date = pd.to_datetime(current_date) 
        train_start = current_date - pd.Timedelta(weeks=min_weeks)
        
        train_df = df.loc[(df.index.get_level_values('Date') >= train_start) &
                          (df.index.get_level_values('Date') < current_date) &
                          (df.index.get_level_values('Ticker').isin(valid_assets))]

        test_df = df.loc[(df.index.get_level_values('Date') == current_date) &
                         (df.index.get_level_values('Ticker').isin(valid_assets))]

        X_train = train_df[features].values
        y_train = train_df[target].values
        X_test = test_df[features].values

        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        
        preds = best_model.predict(X_test)

        print(f"predicting for {current_date}")

        temp = test_df.copy()
        temp['predicted_return'] = preds
        all_preds.append(temp[['predicted_return']])

    return pd.concat(all_preds)

selected_model_key = "lasso"
model = model_configs[selected_model_key]["model"]
param_grid = model_configs[selected_model_key]["param_grid"]


pred_df = walk_forward_predictions_dynamic(
    df=final_df,
    features=['momentum_returns(lag_1)', '12-1 momentum(lag_1)', 'annualised_volatility(lag_1)', 'max_return(lag_1)', 'seasonality_score(lag_1)'],
    target='weekly_returns',
    start_date='2014-01-01',
    end_date='2025-01-01',
    min_weeks=520 , model=model,
    param_grid=param_grid  
)

print("prediction complete")


actual_returns = final_df[['weekly_returns']]
merged_df = pred_df.join(actual_returns, how='left')

r2=r2_score(merged_df['weekly_returns'], merged_df['predicted_return'])
print("R² Score:", r2)
'''merged_df.to_csv("predvsactualTEST.csv")


plt.figure(figsize=(8, 6))
sns.regplot(
    x=merged_df['predicted_return'],
    y=merged_df['weekly_returns'],
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
plt.show()'''
'''
#_____________________________________________________________________________________________________
dates = merged_df.index.get_level_values('Date').unique()
portfolio_values = [1]
prev_weights_dict = None
prev_valid_assets = None

window = 520  # 10-year window

import time
start_time = time.time()

for current_date in dates:
    mu_series = merged_df.loc[current_date]['predicted_return'].dropna()
    valid_assets = mu_series.index.tolist()
    hist_start = pd.to_datetime(current_date) - pd.Timedelta(weeks=520)
    mask = (final_df.index.get_level_values("Date") >= hist_start) & \
       (final_df.index.get_level_values("Date") < current_date) & \
       (final_df.index.get_level_values("Ticker").isin(valid_assets))

    hist_data = final_df.loc[mask, 'weekly_returns'].unstack('Ticker')
    S = CovarianceShrinkage(hist_data, returns_data=True, frequency=52,log_returns=False).ledoit_wolf()
    
    print(f"covariance for date{current_date} computed" )

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

    cleaned_weights = global_min_var_cvxpy(
        mu_series, S,
        w_prev=w_prev,
        b_h=b_h,
        k=kappa,
        lamb=lamb,
        c_h=c_h,
        allow_short=allow_short,
        tickers=valid_assets
    )
    print(f"got optimal weights for date{current_date}")

    if prev_weights_dict is not None:
        
        returns = final_df.loc[(current_date, prev_valid_assets), 'weekly_returns']
        returns.index = returns.index.get_level_values('Ticker')
        returns = returns.reindex(prev_valid_assets)

        portfolio_return = np.dot(prev_weights_series.values, returns.values)

        # Drifted weights due to asset returns
        drifted_values = prev_weights_series * (1 + returns)
        if drifted_values.sum() != 0:
            drifted_weights_series = drifted_values / drifted_values.sum()
        else:
            drifted_weights_series = pd.Series(0, index=prev_valid_assets)

        # Target weights
        target_weights_series = pd.Series(cleaned_weights, index=valid_assets)

        # Align both sets of weights
        all_assets = drifted_weights_series.index.union(target_weights_series.index)
        w_drifted = drifted_weights_series.reindex(all_assets).fillna(0)
        w_target = target_weights_series.reindex(all_assets).fillna(0)

        # Transaction cost
        weight_diff = np.abs(w_target.values - w_drifted.values)
        per_ticker_costs = weight_diff * c_h
        transaction_cost = np.sum(per_ticker_costs)

        net_portfolio_return = portfolio_return - transaction_cost
        print(f"the portfolio return for date {current_date} is ", net_portfolio_return)
        portfolio_values.append(portfolio_values[-1] * (1 + net_portfolio_return))

    prev_weights_dict = cleaned_weights
    prev_valid_assets = valid_assets

end_time = time.time()
print(f"Total runtime for solver: {end_time - start_time:.2f} seconds")

# ____________________________________________________________________________

portfolio_df = pd.DataFrame({
    'Portfolio Value': portfolio_values
}, index=dates[len(dates)-len(portfolio_values):])


print(portfolio_df)
portfolio_df.to_csv("rf_portfolio_value.csv")

portfolio_returns = portfolio_df.pct_change(fill_method=None).dropna()
portfolio_returns=portfolio_returns.squeeze()

cagr = compute_cagr_from_weekly_returns(portfolio_returns)
print(f"CAGR: {cagr:.2%}")

sharpe=sharpe_ratio(portfolio_returns)
print("Sharpe:",sharpe)
print("R² Score:", r2)
'''