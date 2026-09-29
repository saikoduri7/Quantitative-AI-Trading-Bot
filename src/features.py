def create_features(df):

  df = df.copy()
  
  df["Return"] = df["Close"].pct_change()

  # moving averages
  df["SMA_5"] = df["Close"].rolling(5).mean()
  df["SMA_20"] = df["Close"].rolling(20).mean()
  df["SMA_50"] = df["Close"].rolling(50).mean()
  
  # momentum
  df["Momentum_5"] = df["Close"].pct_change(5)
  df["Momentum_20"] = df["Close"].pct_change(20)
  
  # volatility
  df["Volatility_10"] = df["Return"].rolling(10).std()
  df["Volatility_20"] = df["Return"].rolling(20).std()
  
  # volume change
  df["Volume_Change"] = df["Volume"].pct_change()
  
  # intraday range
  df["High_Low_Range"] = (
      (df["High"] - df["Low"]) / df["Close"]
  )

  return df
