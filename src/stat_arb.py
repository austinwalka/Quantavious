# src/stat_arb.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from itertools import combinations

def hedge_ratio(y: pd.Series, x: pd.Series):
    x = sm.add_constant(x)
    model = sm.OLS(y, x, missing="drop").fit()
    return model.params.iloc[1]

def pick_pairs(tickers: list[str], lookback_days: int = 120, top_n: int = 3):
    prices = {}
    for t in tickers:
        d = yf.download(t, period=f"{lookback_days+10}d", interval="1d", progress=False)[["Close"]]
        if not d.empty:
            prices[t] = d["Close"]
    if len(prices) < 2:
        return []
    df = pd.DataFrame(prices).dropna()
    results = []
    for a, b in combinations(df.columns, 2):
        hr = hedge_ratio(df[a], df[b])
        spread = df[a] - hr*df[b]
        z = (spread - spread.mean())/spread.std(ddof=1)
        score = -np.abs(z.iloc[-1])  # closer to 0 is "stable"
        results.append((score, a, b, hr, z.iloc[-1]))
    results.sort(key=lambda r: r[0], reverse=True)
    return results[:top_n]

def zscore_series(spread: pd.Series, win=60):
    mu = spread.rolling(win).mean()
    sd = spread.rolling(win).std(ddof=1)
    return (spread - mu)/sd
