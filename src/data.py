import pandas as pd
import yfinance as yf


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted daily OHLCV data for one ticker."""
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    # yfinance may return a 2-level column index even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.get_level_values(-1).unique()) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            raise ValueError("Expected data for a single ticker.")

    return df.sort_index()
