import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor  # (kept for easy future extension)

# -----------------------------
# Horizon helpers
# -----------------------------
_HORIZON_TO_DAYS = {
    "1d": 1,
    "5d (1w)": 5,
    "1mo": 21,    # trading days approx
    "3mo": 63,
    "6mo": 126,
}

def _get_days(horizon: str) -> int:
    return _HORIZON_TO_DAYS.get(horizon, 21)

def _years_from_days(days: int) -> float:
    # trading year ~ 252 days
    return days / 252.0

# -----------------------------
# Data fetch
# -----------------------------
def fetch_history(ticker: str, lookback_days: int = 400) -> pd.Series:
    """
    Download adjusted close for ~lookback_days. Padding to cover longer horizons.
    """
    period_map = {
        120: "6mo",
        250: "1y",
        500: "2y",
        750: "3y",
    }
    # pick a period big enough
    period = "2y"
    for d, p in period_map.items():
        if lookback_days <= d:
            period = p
            break

    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    closes = df["Close"].dropna()
    return closes

# -----------------------------
# Drift & volatility
# -----------------------------
def estimate_mu_sigma(closes: pd.Series) -> Tuple[float, float]:
    rets = np.log(closes / closes.shift(1)).dropna()
    if len(rets) < 2:
        return 0.0, 0.0001
    mu = rets.mean() * 252.0
    sigma = rets.std(ddof=1) * np.sqrt(252.0)
    # avoid zero sigma
    sigma = float(max(sigma, 1e-6))
    return float(mu), float(sigma)

# -----------------------------
# GBM model
# -----------------------------
def gbm_simulate(S0: float, mu: float, sigma: float, days: int, n_paths: int = 2000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    dt = 1.0 / 252.0
    N = days
    dW = np.random.normal(0.0, np.sqrt(dt), size=(n_paths, N))
    W = np.cumsum(dW, axis=1)
    t = np.arange(1, N + 1) * dt
    paths = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return paths  # shape (n_paths, days)

def gbm_point_forecast(S0: float, mu: float, sigma: float, days: int) -> float:
    T = _years_from_days(days)
    return float(S0 * np.exp(mu * T))

# -----------------------------
# OU (mean reversion) on log price
# -----------------------------
def fit_ou_params(closes: pd.Series) -> Tuple[float, float, float]:
    """
    Discrete-time OU on log-price: dx_t = kappa*(mu - x_t)*dt + sigma*dW
    OLS on daily steps to estimate kappa, mu, sigma.
    """
    x = np.log(closes).dropna().values
    if len(x) < 30:
        # fallback
        return 0.2, float(np.mean(x)), 0.2

    dt = 1.0
    x_t = x[:-1]
    dx = x[1:] - x[:-1]

    # Regress dx on (mu - x_t) ~ a + b*x_t
    # dx = kappa*mu*dt - kappa*x_t*dt + noise
    # => dx = A + B*x_t; with A=kappa*mu*dt, B = -kappa*dt
    X = np.vstack([np.ones_like(x_t), x_t]).T
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    A, B = beta[0], beta[1]

    kappa = -B / dt
    mu = A / (kappa * dt) if kappa != 0 else np.mean(x)
    # residual std
    resid = dx - (A + B * x_t)
    sigma = np.std(resid, ddof=1)  # daily
    sigma = float(max(sigma, 1e-6))
    # scale sigma to annualized on log-scale ~ sqrt(252) if needed; but we forecast in log-space over days -> daily sigma ok
    return float(max(kappa, 1e-6)), float(mu), float(sigma)

def ou_point_forecast(closes: pd.Series, days: int) -> float:
    S0 = float(closes.iloc[-1])
    kappa, mu_log, sigma = fit_ou_params(closes)
    x0 = float(np.log(S0))
    # mean of OU at horizon
    x_T = mu_log + (x0 - mu_log) * np.exp(-kappa * days)
    return float(np.exp(x_T))

# -----------------------------
# ARIMA (short horizon)
# -----------------------------
def arima_point_forecast(closes: pd.Series, days: int) -> Optional[float]:
    """
    Simple ARIMA(1,1,1) on close for short-term horizons.
    """
    if len(closes) < 40:
        return None
    try:
        model = ARIMA(closes, order=(1, 1, 1))
        fit = model.fit(method_kwargs={"warn_convergence": False})
        fc = fit.forecast(steps=days)
        return float(fc.iloc[-1])
    except Exception:
        return None

# -----------------------------
# Schrödinger-inspired (FFT cycle)
# -----------------------------
def schrodinger_cycle_forecast(closes: pd.Series, days: int) -> float:
    """
    Use FFT to detect dominant cycle in detrended returns, then project a sinusoid on top of last price.
    This is a light-weight 'probability amplitude' proxy capturing cyclical structure, not a true SE solver.
    """
    S0 = float(closes.iloc[-1])
    rets = np.log(closes / closes.shift(1)).dropna().values
    if len(rets) < 30:
        return S0
    # remove mean
    r = rets - np.mean(rets)
    fft = np.fft.rfft(r)
    mags = np.abs(fft)
    # ignore DC
    if len(mags) <= 2:
        return S0
    k = np.argmax(mags[1:]) + 1
    # estimated cycle length in days
    cycle_len = max(int(len(r) / max(k, 1)), 5)
    # amplitude scaled by std of returns
    amp = 0.5 * np.std(r) * cycle_len
    phase = 0.0
    signal = amp * np.sin(2 * np.pi * (days % cycle_len) / cycle_len + phase)
    return float(S0 * np.exp(signal))  # map sinusoid in log-return space

# -----------------------------
# Risk: VaR & CVaR from simulations
# -----------------------------
def var_cvar_from_paths(paths: np.ndarray, alpha: float = 0.95, relative: bool = True) -> Tuple[float, float]:
    """
    Compute VaR_alpha and CVaR_alpha on horizon returns from simulated price paths.
    Negative values -> losses. Returned as percentages if relative=True.
    """
    # pick horizon values (last column)
    horizon_vals = paths[:, -1]
    start_vals = paths[:, 0] if paths.shape[1] > 1 else horizon_vals * 0 + horizon_vals.mean()
    rets = (horizon_vals - start_vals) / start_vals  # simple return
    # losses = -rets
    losses = -rets
    # quantile at alpha
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean() if np.any(losses >= var) else var
    if relative:
        return float(var * 100.0), float(cvar * 100.0)
    else:
        # convert back to price terms relative to start mean
        S0 = float(start_vals.mean())
        return float(var * S0), float(cvar * S0)

# -----------------------------
# Meta-blend (adaptive by horizon)
# -----------------------------
def blend_predictions(horizon_days: int,
                      gbm_pred: float,
                      ou_pred: float,
                      arima_pred: Optional[float],
                      sch_pred: float) -> Tuple[float, Dict[str, float]]:
    """
    Adaptive weights:
      - 1d: ARIMA 0.45, OU 0.35, GBM 0.15, SCH 0.05
      - 5d: ARIMA 0.35, OU 0.30, GBM 0.25, SCH 0.10
      - 1mo: ARIMA 0.20, OU 0.25, GBM 0.40, SCH 0.15
      - 3mo: ARIMA 0.10, OU 0.20, GBM 0.50, SCH 0.20
      - 6mo: ARIMA 0.05, OU 0.15, GBM 0.60, SCH 0.20
    """
    if horizon_days <= 1:
        w = {"ARIMA": 0.45, "OU": 0.35, "GBM": 0.15, "SCH": 0.05}
    elif horizon_days <= 5:
        w = {"ARIMA": 0.35, "OU": 0.30, "GBM": 0.25, "SCH": 0.10}
    elif horizon_days <= 21:
        w = {"ARIMA": 0.20, "OU": 0.25, "GBM": 0.40, "SCH": 0.15}
    elif horizon_days <= 63:
        w = {"ARIMA": 0.10, "OU": 0.20, "GBM": 0.50, "SCH": 0.20}
    else:
        w = {"ARIMA": 0.05, "OU": 0.15, "GBM": 0.60, "SCH": 0.20}

    # if ARIMA is None, redistribute its weight proportionally
    if arima_pred is None:
        remove = w["ARIMA"]
        w["ARIMA"] = 0.0
        s = w["OU"] + w["GBM"] + w["SCH"]
        if s > 0:
            w["OU"] += remove * (w["OU"] / s)
            w["GBM"] += remove * (w["GBM"] / s)
            w["SCH"] += remove * (w["SCH"] / s)

    p_arima = arima_pred if arima_pred is not None else 0.0
    blended = (w["ARIMA"] * p_arima +
               w["OU"] * ou_pred +
               w["GBM"] * gbm_pred +
               w["SCH"] * sch_pred)

    return float(blended), w

# -----------------------------
# Public API
# -----------------------------
def predict_stock(ticker: str,
                  retrain_if_missing: bool = False,
                  horizon: str = "1mo") -> Optional[Dict]:
    """
    Returns a dict with:
      Ticker, Horizon, Last Close, Predicted Price, GBM, OU, ARIMA, Schrodinger,
      VaR_95_pct, CVaR_95_pct, BlendWeights (string), Prediction Date
    """
    days = _get_days(horizon)

    # Get enough history: a bit more than horizon for stability
    lookback = max(180, days * 4)
    closes = fetch_history(ticker, lookback_days=lookback)
    if closes.empty or len(closes) < 30:
        return None

    S0 = float(closes.iloc[-1])
    mu, sigma = estimate_mu_sigma(closes)

    # --- individual model point forecasts ---
    gbm_pred = gbm_point_forecast(S0, mu, sigma, days)
    ou_pred = ou_point_forecast(closes, days)
    arima_pred = arima_point_forecast(closes, days)  # may be None
    sch_pred = schrodinger_cycle_forecast(closes, days)

    # --- risk via Monte-Carlo (simulate GBM; OU is slower to simulate robustly) ---
    gbm_paths = gbm_simulate(S0, mu, sigma, days, n_paths=3000, seed=123)
    var95, cvar95 = var_cvar_from_paths(gbm_paths, alpha=0.95, relative=True)

    # --- meta blend ---
    blended, weights = blend_predictions(days, gbm_pred, ou_pred, arima_pred, sch_pred)

    result = {
        "Ticker": ticker,
        "Horizon": horizon,
        "Last Close": round(S0, 4),
        "Predicted Price": round(blended, 4),
        "GBM": round(gbm_pred, 4),
        "OU": round(ou_pred, 4),
        "ARIMA": None if arima_pred is None else round(arima_pred, 4),
        "Schrodinger": round(sch_pred, 4),
        "VaR_95_pct": round(var95, 2),
        "CVaR_95_pct": round(cvar95, 2),
        "BlendWeights": f"ARIMA={weights['ARIMA']:.2f}, OU={weights['OU']:.2f}, GBM={weights['GBM']:.2f}, SCH={weights['SCH']:.2f}",
        "Prediction Date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    return result


# For quick local testing
if __name__ == "__main__":
    out = predict_stock("AAPL", horizon="5d (1w)")
    print(out)
