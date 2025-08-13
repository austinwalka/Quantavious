# src/features.py
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

def _add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Momentum
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()

    # Trend
    sma_20 = SMAIndicator(close, window=20).sma_indicator()
    sma_50 = SMAIndicator(close, window=50).sma_indicator()

    # Volatility
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_h = bb.bollinger_hband()
    bb_l = bb.bollinger_lband()
    bb_w = (bb_h - bb_l) / close

    out = df.copy()
    out["RSI_14"] = rsi
    out["MACD"] = macd_line
    out["MACD_signal"] = macd_signal
    out["SMA_20"] = sma_20
    out["SMA_50"] = sma_50
    out["BB_width"] = bb_w

    # Basic returns/vol
    out["ret_1d"] = close.pct_change()
    out["ret_5d"] = close.pct_change(5)
    out["vol_10d"] = out["ret_1d"].rolling(10).std()

    return out

def _fetch_macro_proxies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch light macro proxies aligned to index:
      - ^VIX  (fear/vol)
      - ^TNX  (10Y yield)
    """
    start = index.min() - pd.Timedelta(days=7)
    end = index.max() + pd.Timedelta(days=1)
    frames = []
    for sym, col in [("^VIX", "VIX"), ("^TNX", "TNX")]:
        try:
            d = yf.download(sym, start=start, end=end, progress=False)
            d = d[["Close"]].rename(columns={"Close": col})
            frames.append(d)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(index=index)
    macro = pd.concat(frames, axis=1).reindex(index).ffill()
    # pct changes to make stationary-ish signals
    if "VIX" in macro:
        macro["VIX_chg"] = macro["VIX"].pct_change()
    if "TNX" in macro:
        macro["TNX_chg"] = macro["TNX"].pct_change()
    return macro

def create_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a feature-augmented dataframe aligned with price_df.
    Assumes price_df has columns: Open, High, Low, Close, Volume and Date index.
    """
    df = price_df.copy()
    df = _add_ta_features(df)
    macro = _fetch_macro_proxies(df.index)
    df = df.join(macro, how="left")
    df = df.ffill().bfill()
    return df
