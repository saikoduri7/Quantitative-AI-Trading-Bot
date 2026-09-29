import numpy as np


def total_return(portfolio, initial_capital=10000):
    return portfolio.iloc[-1] / initial_capital - 1


def sharpe_ratio(returns, periods_per_year=252):
    std = returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (returns.mean() / std) * np.sqrt(periods_per_year)


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def drawdown_series(portfolio):
    running_max = portfolio.cummax()
    return (portfolio - running_max) / running_max


def number_of_trades(backtest):
    return int((backtest["Trade"] > 0).sum())
