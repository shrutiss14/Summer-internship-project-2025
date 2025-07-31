from comparison_metric import compute_cagr_from_weekly_returns,sharpe_ratio,annualised_volatility,max_drawdown_by_vol,sortino_ratio,treynor_ratio,information_ratio
import pandas as pd


portfolio_df=pd.read_csv("rf_portfolio_value.csv",index_col="Date",parse_dates=True)
portfolio_returns = portfolio_df.pct_change(fill_method=None).dropna()
portfolio_returns=portfolio_returns.squeeze()

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

#___________________________________________________________________________________________________________________________

# Collect all metrics in a dictionary
metrics = {
    "CAGR": [cagr],
    "Annualised Volatility": [ann_vol],
    "Max Drawdown/vol": [max_draw],
    "Sharpe": [sharpe],
    "Sortino": [sortino],
    "Treynor": [treynor]
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)

# Save to CSV
metrics_df.to_csv("rf_portfolio_value_metrics.csv", index=False)

#__________________________________________________________________________________________________

