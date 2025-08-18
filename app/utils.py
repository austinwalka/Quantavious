import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_stock_list():
    """Return available tickers (folders inside data)."""
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def load_forecast(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, ticker, "forecast_30d.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_crash(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, ticker, "crash_30d.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_indicators(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, ticker, "indicators.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    return pd.DataFrame()

def load_backtest(ticker: str) -> dict:
    path = os.path.join(DATA_DIR, ticker, "backtest.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def load_meta(ticker: str) -> dict:
    path = os.path.join(DATA_DIR, ticker, "meta.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}
