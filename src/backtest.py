import numpy as np
import pandas as pd


def run_backtest(actual_returns, signals, initial_capital=10000, transaction_cost=0.0005, index=None):
    """Backtest daily positions with turnover-based transaction costs."""
    actual_returns = np.asarray(actual_returns).reshape(-1)
    signals = np.asarray(signals).reshape(-1)
    if len(actual_returns) != len(signals):
        raise ValueError("actual_returns and signals must have the same length")

    backtest = pd.DataFrame({"Actual_Return": actual_returns, "Signal": signals}, index=index)
    backtest["Strategy_Return"] = backtest["Signal"] * backtest["Actual_Return"]
    backtest["Trade"] = backtest["Signal"].diff().abs().fillna(backtest["Signal"].abs())
    backtest["Strategy_Return_Net"] = (
        backtest["Strategy_Return"] - backtest["Trade"] * transaction_cost
    )
    backtest["Portfolio"] = initial_capital * (1 + backtest["Strategy_Return_Net"]).cumprod()
    backtest["Buy_Hold"] = initial_capital * (1 + backtest["Actual_Return"]).cumprod()
    return backtest
