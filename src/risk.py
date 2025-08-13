# src/risk.py
import numpy as np
import pandas as pd

def volatility_target(weights: np.ndarray, cov: np.ndarray, target_vol_annual=0.15):
    cur_vol = np.sqrt(np.dot(weights, cov).dot(weights.T)) * np.sqrt(252)
    if cur_vol <= 1e-8:
        return weights
    scale = target_vol_annual / cur_vol
    return np.clip(weights*scale, -1.5, 1.5)

def kelly_fractional(edge: float, odds: float = 1.0, frac: float = 0.25):
    """Return fraction of capital to risk (capped) using fractional Kelly."""
    if odds <= 0:
        return 0.0
    f = (edge/odds)
    return float(np.clip(frac * f, -0.5, 0.5))

def apply_stops(equity: pd.Series, stop=-0.03, take=0.06):
    ret = equity.pct_change().fillna(0)
    cum = (1+ret).cumprod()
    dd = (cum / cum.cummax()) - 1.0
    signals = pd.Series(1.0, index=equity.index)
    signals[(ret <= stop) | (ret >= take) | (dd < -0.10)] = 0.0
    return signals
