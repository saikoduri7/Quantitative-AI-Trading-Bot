import pandas as pd

FEATURES = [
    "Return",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "Momentum_5",
    "Momentum_20",
    "Volatility_10",
    "Volatility_20",
    "Volume_Change",
    "High_Low_Range",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features used in the exploration notebook."""
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["SMA_5"] = out["Close"].rolling(5).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["Momentum_5"] = out["Close"].pct_change(5)
    out["Momentum_20"] = out["Close"].pct_change(20)
    out["Volatility_10"] = out["Return"].rolling(10).std()
    out["Volatility_20"] = out["Return"].rolling(20).std()
    out["Volume_Change"] = out["Volume"].pct_change()
    out["High_Low_Range"] = (out["High"] - out["Low"]) / out["Close"]
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-day return and direction targets, then remove incomplete rows."""
    out = df.copy()
    out["Target"] = out["Return"].shift(-1)
    out = out.dropna().copy()
    out["Direction_Target"] = (out["Target"] > 0).astype(int)
    return out
