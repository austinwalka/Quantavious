# src/intraday_fetch.py
import pandas as pd
import yfinance as yf

def get_price_data(ticker: str, mode: str = "intraday", days: int = 5, interval: str = "1m") -> pd.DataFrame:
    """
    mode: 'intraday' or 'daily'
    intraday: use interval (1m/5m/15m). Yahoo allows ~7 days for 1m.
    daily: last `days`*1 business days.
    """
    if mode == "intraday":
        # yfinance uses 'period' for intraday, e.g., '5d'
        period = f"{max(1, min(days, 7))}d"
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    else:
        # daily last N business days + some buffer
        df = yf.download(ticker, period=f"{max(5, days+5)}d", interval="1d", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df
