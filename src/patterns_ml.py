# src/patterns_ml.py
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

def micro_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["Close"].pct_change()
    out["ret5"] = out["Close"].pct_change(5)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["vwap_proxy"] = (out["High"] + out["Low"] + 2*out["Close"]) / 4
    out["RSI_14"] = RSIIndicator(out["Close"], 14).rsi()
    macd = MACD(out["Close"])
    out["MACD"] = macd.macd()
    out["MACD_sig"] = macd.macd_signal()
    return out.dropna()

def template_match(close: pd.Series, lookback=60, k=5):
    """
    Find k nearest historical motifs to the last `lookback` window (Euclidean).
    Return their next-step returns as a small ensemble.
    """
    x = close.values
    if len(x) <= lookback+1:
        return np.array([0.0])
    ref = (x[-lookback:] / x[-lookback]) - 1.0
    scores = []
    for i in range(lookback, len(x)-1):
        seg = (x[i-lookback:i] / x[i-lookback]) - 1.0
        d = np.linalg.norm(ref - seg)
        nxt = (x[i+1] / x[i]) - 1.0
        scores.append((d, nxt))
    scores.sort(key=lambda z: z[0])
    top = [n for _, n in scores[:k]]
    return np.array(top)

def short_horizon_ml(df: pd.DataFrame, horizon_steps: int = 5):
    """
    Fast regressors to predict next price (1 step) and roll forward to horizon.
    """
    f = micro_features(df)
    y = f["Close"].shift(-1).dropna()
    X = f.loc[y.index].drop(columns=["Close","Open","High","Low","Volume"])
    split = int(len(X)*0.8)
    if split < 50:
        # not enough data
        last = df["Close"].iloc[-1]
        return np.array([last]*horizon_steps)

    # blend Ridge + GBDT
    ridge = Ridge(alpha=1.0)
    gbr = GradientBoostingRegressor()
    ridge.fit(X.iloc[:split], y.iloc[:split])
    gbr.fit(X.iloc[:split], y.iloc[:split])

    # roll forward predictions
    cur_row = X.iloc[-1:].copy()
    preds = []
    last_price = df["Close"].iloc[-1]
    for _ in range(horizon_steps):
        p = 0.5*ridge.predict(cur_row)[0] + 0.5*gbr.predict(cur_row)[0]
        preds.append(p)
        # update row with new "Close"
        # (simple: move price; keep features mostly static for ultra-short horizon)
        last_price = p
    return np.array(preds)
