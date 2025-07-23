import numpy as np
import pandas as pd


def compute_cagr_from_weekly_returns(weekly_returns):
    n_weeks = len(weekly_returns)
    if n_weeks == 0:
        raise ValueError("weekly_returns cannot be empty.")
    n_years = n_weeks / 52
    growth_factor = 1 + weekly_returns
    total_growth_factor = np.prod(growth_factor)
    return total_growth_factor ** (1 / n_years) - 1


def annualised_volatility(portfolio_returns):
    weekly_std_dev=portfolio_returns.std()
    annual_volatility=weekly_std_dev*np.sqrt(52)
    return annual_volatility


def max_drawdown(portfolio_df):
    df=portfolio_df.copy()
    df['Running Peak']=df['Portfolio Value'].cummax()
    df['Drawdown']=(df['Running Peak'] - df['Portfolio Value'])/df['Running Peak']
    max_dd=df['Drawdown'].max()
    return max_dd

def sharpe_ratio(portfolio_returns):
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()

    sharpe_ratio_annual = (mean_return*52) / (std_dev*np.sqrt(52))
    return sharpe_ratio_annual

def sortino_ratio(portfolio_returns):
    mean_return=portfolio_returns.mean()
    downside_returns=portfolio_returns[portfolio_returns<0]
    downside_std_dev=np.sqrt((downside_returns ** 2).mean())
    if downside_std_dev == 0:
        return np.inf if mean_return > 0 else -np.inf
    
    sortino_ratio_weekly = mean_return / downside_std_dev
    sortino_ratio_annual = sortino_ratio_weekly * np.sqrt(52)
    return sortino_ratio_annual

def treynor_ratio(portfolio_returns,market_returns):
    combined=pd.concat([portfolio_returns,market_returns],axis=1).dropna()
    combined.columns=['Portfolio Return', 'Market Return']

    cov=np.cov(combined['Portfolio Return'],combined['Market Return'])[0,1]
    var_market=np.var(combined['Market Return'])

    port_beta=cov/var_market

    port_mean_return=combined['Portfolio Return'].mean()
    treynor=port_mean_return/port_beta
    annualised_treynor=treynor * 52

    return annualised_treynor

def information_ratio(portfolio_returns,market_returns):
    combined=pd.concat([portfolio_returns,market_returns],axis=1).dropna()
    combined.columns=['Portfolio Return', 'Market Return']
    active_returns=combined['Portfolio Return']-combined['Market Return']
    avg_active_return=active_returns.mean()
    track_err=active_returns.std()
    if track_err == 0:
        return np.inf if avg_active_return > 0 else -np.inf
    infor_ratio_annual=(avg_active_return*52)/(track_err* np.sqrt(52))
    return infor_ratio_annual
    





    



