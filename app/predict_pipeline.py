import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
import os
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_forecast(ticker):
    """Load forecast CSV for a ticker"""
    filepath = os.path.join(DATA_DIR, f"{ticker}_forecast.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return None

def download_last_90_days(ticker):
    """Download last 90 days and compute technical indicators"""
    df = yf.download(ticker, period="90d", interval="1d")
    if df.empty:
        return None
    df["SMA20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    return df
