# src/predict_pipeline.py
"""
Full predict pipeline (LightGBM, XGBoost, Prophet, LSTM, GBM, OU, Schrodinger)
- Defensive imports: heavy libraries are optional and will fallback gracefully.
- Streamlit caching: use @st.cache_resource and @st.cache_data if run under Streamlit.
- Returns: forecast_df, rmse_scores, blend_weights
"""

from __future__ import annotations
import os
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import yfinance as yf

# Optional heavy imports
_HAS_XGBOOST = False
_HAS_LIGHTGBM = False
_HAS_PROPHET = False
_HAS_TRANSFORMERS = False

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except Exception:
    _HAS_LIGHTGBM = False

try:
    # Prophet import (optional)
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# Streamlit caching helpers (use if running under Streamlit)
try:
    import streamlit as st
    CACHE_RESOURCE = st.cache_resource
    CACHE_DATA = st.cache_data
except Exception:
    # dummy decorators if streamlit is not available
    def CACHE_RESOURCE(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

    def CACHE_DATA(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

# PyTorch for LSTM (required)
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    # We'll raise later if LSTM requested but torch not installed

# FFT for Schrödinger
from scipy.fftpack import fft, ifft

# -------------------------
# Utilities / data fetch
# -------------------------
def fetch_price_data(ticker: str, mode: str = "intraday", days: int = 5, interval: str = "1m") -> pd.DataFrame:
    """
    Fetch OHLCV via yfinance. mode: 'intraday' or 'daily'.
    For intraday use interval '1m','5m','15m'. yfinance limits apply.
    """
    if mode == "intraday":
        period = f"{max(1, min(days, 7))}d"
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    else:
        df = yf.download(ticker, period=f"{max(10, days+10)}d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

# -------------------------
# Math models (GBM, OU, Schrodinger)
# -------------------------
def simulate_gbm(S0: float, mu: float, sigma: float, steps: int, dt: float, n_paths: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, steps)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt
    paths = np.empty((n_paths, steps+1))
    paths[:,0] = S0
    for t in range(steps):
        paths[:, t+1] = paths[:, t] * np.exp(drift + sigma * shocks[:, t])
    return paths

@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float

def estimate_ou_params(returns: np.ndarray, dt: float = 1.0) -> OUParams:
    if len(returns) < 3:
        return OUParams(theta=1.0, mu=float(np.mean(returns) if len(returns)>0 else 0.0), sigma=float(np.std(returns) if len(returns)>1 else 1e-6))
    rt = returns[1:]
    rlag = returns[:-1]
    X = np.vstack([np.ones_like(rlag), rlag]).T
    beta, *_ = np.linalg.lstsq(X, rt, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    b = float(np.clip(b, -0.99, 0.9999))
    theta = -np.log(max(1e-8, b)) / dt if abs(b) < 0.9999 else 1e-6
    mu = a / (1 - b) if abs(1-b) > 1e-9 else 0.0
    eps = rt - (a + b * rlag)
    sigma = float(np.std(eps) if len(eps)>1 else 1e-6)
    return OUParams(theta=theta, mu=mu, sigma=abs(sigma))

def simulate_ou(r0: float, params: OUParams, steps: int, dt: float, n_paths: int = 1000, seed: int = 7):
    rng = np.random.default_rng(seed)
    r = np.zeros((n_paths, steps+1))
    r[:,0] = r0
    for t in range(steps):
        mean = r[:,t] + params.theta * (params.mu - r[:,t]) * dt
        std = params.sigma * np.sqrt(dt)
        r[:, t+1] = mean + std * rng.standard_normal(n_paths)
    return r

# Schrödinger split-step implementation (full)
class Schrodinger:
    def __init__(self, x, psi_x0, V_x, k0=None, hbar=1.0, m=1.0, t0=0.0):
        self.x = np.asarray(x)
        self.N = len(self.x)
        self.dx = self.x[1] - self.x[0]
        self.hbar = hbar
        self.m = m
        self.V_x = np.asarray(V_x)
        self.psi_x = np.asarray(psi_x0).astype(np.complex128)
        self.t = t0
        self.dk = 2*np.pi / (self.N * self.dx)
        if k0 is None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            self.k0 = k0
        self.k = self.k0 + self.dk * np.arange(self.N)

    def time_step(self, dt, steps=1):
        x_evolve_half = np.exp(-0.5j * self.V_x / self.hbar * dt)
        x_evolve = x_evolve_half**2
        k_evolve = np.exp(-0.5j * self.hbar * (self.k**2) / self.m * dt)
        for _ in range(steps):
            self.psi_x *= x_evolve_half
            psi_k = fft(self.psi_x)
            psi_k *= k_evolve
            self.psi_x = ifft(psi_k)
            self.psi_x *= x_evolve
        self.psi_x *= x_evolve_half
        self.t += dt * steps

def schrodinger_terminal_pdf(S0: float, mu: float, sigma: float, horizon_days: float, N_x=1024):
    x_min = np.log(S0) - 5 * sigma * np.sqrt(max(horizon_days,1e-9))
    x_max = np.log(S0) + 5 * sigma * np.sqrt(max(horizon_days,1e-9))
    x = np.linspace(x_min, x_max, N_x)
    a = 0.05
    k0 = (mu - sigma**2/2) / max(sigma, 1e-9)
    psi0 = ((a/np.sqrt(np.pi))**0.5) * np.exp(-0.5 * ((x-np.log(S0))/a)**2 + 1j * k0 * x)
    V_x = np.zeros_like(x)
    se = Schrodinger(x, psi0, V_x, hbar=1.0, m=1.0/(2*sigma**2 if sigma>0 else 1.0))
    T = horizon_days / 252.0
    dt = T / 400.0
    steps = max(1, 400)
    se.time_step(dt, steps)
    p_x = np.abs(se.psi_x)**2
    dx = x[1]-x[0]
    p_x /= p_x.sum() * dx
    S_grid = np.exp(x)
    return S_grid, p_x

# -------------------------
# LSTM (PyTorch) implementation
# -------------------------
if _HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
else:
    LSTMModel = None

@CACHE_RESOURCE
def train_lstm_cached(series: np.ndarray, lookback=60, epochs=30, lr=1e-3, device: str = "cpu"):
    """Train small LSTM and return (model, scaler). Cached via Streamlit when available."""
    if not _HAS_TORCH:
        return None, None
    device = torch.device(device)
    scaler = MinMaxScaler(feature_range=(0,1))
    arr = series.reshape(-1,1)
    scaled = scaler.fit_transform(arr).astype(np.float32)
    if len(scaled) < lookback + 5:
        return None, scaler
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1,1)
    split = max(1, int(len(X)*0.8))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        bx = torch.from_numpy(X_train).float().to(device)
        by = torch.from_numpy(y_train).float().to(device)
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        # optional validate
    return model, scaler

def lstm_forecast_from_model(model, scaler, series: np.ndarray, lookback: int, horizon: int, device: str = "cpu"):
    if model is None or scaler is None:
        return None
    if not _HAS_TORCH:
        return None
    device = torch.device(device)
    arr = series.reshape(-1,1)
    scaled = scaler.transform(arr).astype(np.float32).flatten()
    seq = scaled[-lookback:].copy()
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.from_numpy(seq.reshape(1, lookback, 1)).float().to(device)
            out = model(x).cpu().numpy().flatten()[0]
            preds.append(out)
            seq = np.roll(seq, -1)
            seq[-1] = out
    preds = np.array(preds).reshape(-1,1)
    inv = scaler.inverse_transform(preds).flatten()
    S0 = float(series.flatten()[-1])
    return np.concatenate(([S0], inv))

# -------------------------
# LightGBM / XGBoost wrappers (optional)
# -------------------------
def fit_lightgbm(X, y):
    if not _HAS_LIGHTGBM:
        # fallback to sklearn gbdt
        m = GradientBoostingRegressor(n_estimators=200)
        m.fit(X, y)
        return m
    model = lgb.LGBMRegressor(n_estimators=200)
    model.fit(X, y)
    return model

def fit_xgboost(X, y):
    if not _HAS_XGBOOST:
        m = GradientBoostingRegressor(n_estimators=200)
        m.fit(X, y)
        return m
    model = xgb.XGBRegressor(n_estimators=200, verbosity=0, eval_metric="rmse", use_label_encoder=False)
    model.fit(X, y)
    return model

# -------------------------
# Prophet wrapper (optional)
# -------------------------
def fit_prophet(df_close: pd.Series, periods: int = 5):
    if not _HAS_PROPHET:
        return None
    df = pd.DataFrame({"ds": df_close.index, "y": df_close.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    # return predictions for future tail
    return fc[["ds","yhat"]].set_index("ds")["yhat"].iloc[-periods:]

# -------------------------
# finBERT optional
# -------------------------
@CACHE_RESOURCE
def load_finbert_cached(model_name: str = "ProsusAI/finbert"):
    if not _HAS_TRANSFORMERS:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True)
        return clf
    except Exception:
        return None

def finbert_score_from_headlines(headlines: List[str], pipe=None) -> float:
    if not headlines:
        return 0.0
    if pipe is None:
        # simple heuristic fallback
        vals = []
        for h in headlines:
            t = (h or "").lower()
            pos = sum(t.count(w) for w in ("beat","surge","gain","upgrade","outperform"))
            neg = sum(t.count(w) for w in ("miss","drop","downgrade","loss","weak"))
            vals.append(np.tanh((pos-neg)/5.0))
        return float(np.clip(np.mean(vals) if vals else 0.0, -1.0, 1.0))
    all_scores = []
    for i in range(0, len(headlines), 8):
        chunk = headlines[i:i+8]
        try:
            preds = pipe(chunk, truncation=True)
            for p in preds:
                mapping = {d["label"].lower(): d["score"] for d in p}
                score = mapping.get("positive",0.0) - mapping.get("negative",0.0)
                all_scores.append(score)
        except Exception:
            continue
    if not all_scores:
        return 0.0
    return float(np.clip(np.mean(all_scores), -1.0, 1.0))

def fetch_headlines_newsapi(ticker: str, api_key: Optional[str], days: int = 3, limit: int = 12):
    if not api_key:
        return []
    import requests
    from datetime import datetime, timedelta
    frm = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    q = f"{ticker} stock"
    url = ("https://newsapi.org/v2/everything?"
           f"q={q}&from={frm}&sortBy=publishedAt&language=en&pageSize={limit}")
    headers = {"X-Api-Key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
        return [a.get("title","") for a in arts if a.get("title")]
    except Exception:
        return []

# -------------------------
# Meta-blender helpers
# -------------------------
def compute_skills(preds: Dict[str, np.ndarray], actual: Optional[np.ndarray]):
    skills = {}
    if actual is None:
        for k in preds.keys():
            skills[k] = np.nan
        return skills
    for k,v in preds.items():
        try:
            arr = np.asarray(v).flatten()[:len(actual)]
            skills[k] = float(sqrt(mean_squared_error(actual[:len(arr)], arr)))
        except Exception:
            skills[k] = float("nan")
    return skills

def weights_from_skills(skills: Dict[str,float], eps: float = 1e-8):
    inv = {}
    for k,v in skills.items():
        if np.isnan(v) or v <= 0 or not np.isfinite(v):
            inv[k] = 0.0
        else:
            inv[k] = 1.0 / (v + eps)
    s = sum(inv.values())
    if s <= 0:
        n = len(skills)
        return {k:1.0/n for k in skills}
    return {k: float(inv[k]/s) for k in skills}

def tilt_weights_by_sentiment(base: Dict[str,float], sentiment: float, trend_models=("GBM","ML"), meanrev_models=("OU",), tilt_strength=0.5):
    s = float(np.clip(sentiment, -1.0, 1.0))
    tilt = s * float(np.clip(tilt_strength, 0.0, 1.0))
    if abs(tilt) < 1e-9:
        return base
    w = base.copy()
    trend_set = set(trend_models)
    mean_set = set(meanrev_models)
    donors = mean_set if tilt > 0 else trend_set
    recipients = trend_set if tilt > 0 else mean_set
    donor_mass = sum(w.get(d,0.0) for d in donors)
    recip_mass = sum(w.get(r,0.0) for r in recipients)
    if donor_mass <= 0 or recip_mass <= 0:
        return w
    shift = donor_mass * abs(tilt)
    for d in donors:
        if d in w:
            w[d] = max(0.0, w[d] - (w[d]/donor_mass)*shift)
    for r in recipients:
        if r in w:
            w[r] = w[r] + (w[r]/recip_mass)*shift
    ssum = sum(w.values())
    if ssum <= 0:
        n = len(w)
        return {k:1.0/n for k in w}
    return {k: float(v/ssum) for k,v in w.items()}

def blend_paths(preds: Dict[str,np.ndarray], weights: Dict[str,float]):
    arrays = {k: np.asarray(v).flatten() for k,v in preds.items()}
    T = min(len(a) for a in arrays.values())
    total = np.zeros(T)
    for k,a in arrays.items():
        w = weights.get(k,0.0)
        total += w * a[:T]
    return total

# -------------------------
# Top-level pipeline
# -------------------------
@CACHE_DATA
def predict_stock(
    ticker: str,
    mode: str = "intraday",
    interval: str = "1m",
    lookback_days: int = 5,
    forecast_steps: int = 60,
    n_paths: int = 800,
    news_api_key: Optional[str] = None,
    tilt_strength: float = 0.5,
    use_prophet: bool = False,
    use_xgb: bool = False,
    use_lgb: bool = False,
    lstm_lookback: int = 60,
    lstm_epochs: int = 25,
    device: str = "cpu"
) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,float]]:
    """
    Returns (forecast_df, rmse_scores, blend_weights)
    forecast_df columns: GBM, OU, ML (LSTM), Best_Guess, Sentiment, Schrodinger_mean
    """
    # 1) data
    price_df = fetch_price_data(ticker, mode=mode, days=lookback_days, interval=interval)
    if price_df.empty:
        raise ValueError(f"No price data for {ticker}")
    S0 = float(price_df["Close"].iloc[-1])

    # dt and steps
    if mode == "intraday":
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            dt = minutes / 390.0
        else:
            dt = 1.0/252.0
        steps = forecast_steps
    else:
        dt = 1.0/252.0
        steps = min(forecast_steps, 5)

    # empirical drift/vol
    logret = np.log(price_df["Close"]).diff().dropna().values
    if len(logret) < 2:
        mu = 0.0
        sigma = 1e-6
    else:
        mu = float(np.mean(logret) / dt)
        sigma = float(np.std(logret) / np.sqrt(max(dt,1e-9)))

    # GBM
    try:
        gbm_paths = simulate_gbm(S0, mu, sigma, steps=steps, dt=dt, n_paths=n_paths)
        gbm_path = gbm_paths.mean(axis=0)
    except Exception:
        gbm_path = np.repeat(S0, steps+1)

    # OU
    if len(logret) > 5:
        oup = estimate_ou_params(logret, dt=dt)
        try:
            ou_r = simulate_ou(0.0, oup, steps=steps, dt=dt, n_paths=n_paths)
            ou_mean = ou_r.mean(axis=0)
            ou_price = S0 * np.exp(np.cumsum(ou_mean))
        except Exception:
            ou_price = np.repeat(S0, steps+1)
    else:
        ou_price = np.repeat(S0, steps+1)

    # Schrödinger terminal mean
    try:
        S_grid, sch_pdf = schrodinger_terminal_pdf(S0, mu, sigma, horizon_days=steps*dt, N_x=1024)
        sch_mean = float(np.trapz(S_grid * sch_pdf, S_grid))
    except Exception:
        sch_mean = S0

    # LSTM ML (train cached)
    ml_path = np.repeat(S0, steps+1)
    try:
        if _HAS_TORCH and len(price_df) > lstm_lookback + 5:
            series = price_df["Close"].values.astype(float)
            lstm_model, lstm_scaler = train_lstm_cached(series, lookback=lstm_lookback, epochs=lstm_epochs, device=device)
            if lstm_model is not None and lstm_scaler is not None:
                out = lstm_forecast_from_model(lstm_model, lstm_scaler, series, lookback=lstm_lookback, horizon=steps, device=device)
                if out is not None:
                    ml_path = out
    except Exception:
        ml_path = np.repeat(S0, steps+1)

    # Optionally: LightGBM/XGBoost/Prophet for short horizon (train on features)
    # Build a simple feature frame (lagged returns)
    feat = pd.DataFrame(index=price_df.index)
    feat["close"] = price_df["Close"]
    feat["r1"] = price_df["Close"].pct_change().fillna(0)
    feat["r2"] = price_df["Close"].pct_change(2).fillna(0)
    feat["vol5"] = feat["r1"].rolling(5).std().fillna(method="bfill")
    feat = feat.dropna()
    ml_tree_path = None
    if len(feat) > 30:
        y = feat["close"].shift(-1).dropna()
        X = feat.loc[y.index].drop(columns=["close"], errors="ignore")
        split = int(len(X)*0.75)
        if split >= 10:
            X_train, y_train = X.iloc[:split], y.iloc[:split]
            X_test = X.iloc[split:]
            try:
                # prefer LightGBM if requested/available
                if use_lgb and _HAS_LIGHTGBM:
                    model_tree = fit_lightgbm(X_train, y_train)
                elif use_xgb and _HAS_XGBOOST:
                    model_tree = fit_xgboost(X_train, y_train)
                else:
                    model_tree = GradientBoostingRegressor(n_estimators=200)
                    model_tree.fit(X_train, y_train)
                # predict next 'steps' by iterating last row (fast approximation)
                last_row = X.iloc[-1:].copy()
                preds = []
                for _ in range(steps):
                    p = model_tree.predict(last_row)[0]
                    preds.append(p)
                ml_tree_path = np.concatenate(([S0], np.array(preds)))
            except Exception:
                ml_tree_path = None

    # prefer LSTM ML path; fallback to tree path if available
    if ml_path is None or len(ml_path)==0:
        if ml_tree_path is not None:
            ml_path = ml_tree_path
        else:
            ml_path = np.repeat(S0, steps+1)

    # Prepare holdout actuals (last available T points)
    hold_len = min(len(price_df), steps)
    hold_actual = price_df["Close"].values[-hold_len:] if hold_len>0 else None

    # Align arrays (GBM, OU, ML) -> trim to same T
    arrays = {"GBM": np.asarray(gbm_path).flatten(), "OU": np.asarray(ou_price).flatten(), "ML": np.asarray(ml_path).flatten()}
    T = min(len(a) for a in arrays.values())
    for k in arrays:
        arrays[k] = arrays[k][:T]

    # Compute RMSE skills if actual available (use last T points of history as proxy)
    rmse_scores = {}
    if hold_actual is not None and len(hold_actual) >= 1:
        actual_trim = np.asarray(hold_actual)[-T:]
        for k, arr in arrays.items():
            try:
                rmse_scores[k] = float(sqrt(mean_squared_error(actual_trim, arr[:len(actual_trim)])))
            except Exception:
                rmse_scores[k] = float("nan")
    else:
        for k in arrays.keys():
            rmse_scores[k] = float("nan")

    # Sentiment
    headlines = fetch_headlines_newsapi(ticker, api_key=news_api_key, days=3, limit=12) if news_api_key else []
    if _HAS_TRANSFORMERS:
        pipe = load_finbert_cached()
        sent = finbert_score_from_headlines(headlines, pipe)
    else:
        sent = finbert_score_from_headlines(headlines, None)

    # Base weights from skills -> tilt by sentiment -> blend
    skills = compute_skills(arrays, actual=np.asarray(hold_actual) if hold_actual is not None else None)
    base_weights = weights_from_skills(skills)
    final_weights = tilt_weights_by_sentiment(base_weights, sent, trend_models=("GBM","ML"), meanrev_models=("OU",), tilt_strength=tilt_strength)
    best = blend_paths(arrays, final_weights)

    # Build forecast DataFrame (index: steps)
    if mode == "intraday":
        idx = pd.RangeIndex(0, T, name="step")
    else:
        last_dt = price_df.index[-1]
        idx = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=T)

    forecast_df = pd.DataFrame(index=idx)
    forecast_df["GBM"] = arrays["GBM"][:T]
    forecast_df["OU"] = arrays["OU"][:T]
    forecast_df["ML"] = arrays["ML"][:T]
    forecast_df["Best_Guess"] = best[:T]
    forecast_df["Sentiment"] = float(sent)
    forecast_df["Schrodinger_mean"] = float(sch_mean)

    return forecast_df, rmse_scores, final_weights
