# src/features.py
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add some common technical indicators to df with 'Close' column.
    """
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    df = df.dropna()
    return df
