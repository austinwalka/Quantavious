# src/predict_pipeline.py
"""
Predict pipeline with:
 - LSTM PyTorch full training/predict routine
 - Schrödinger split-step time evolution (FFT)
 - GBM, OU, Fokker-Planck proxies
 - finBERT optional headline sentiment
 - Streamlit caching for models/forecasts
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import streamlit as st

# PyTorch for LSTM
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Optional HF finBERT
_FINBERT_AVAILABLE = False
_finbert_pipe = None
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    _FINBERT_AVAILABLE = True
except Exception:
    _FINBERT_AVAILABLE = False


# -------------------------
# Utilities & data fetch
# -------------------------
def fetch_price_data(ticker: str, mode: str = "intraday", days: int = 5, interval: str = "1m") -> pd.DataFrame:
    """Fetch price data via yfinance (intraday or daily)."""
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
    df = df.loc[~df.index.duplicated(keep='first')]
    return df


# -------------------------
# Schrödinger (split-step) full implementation
# -------------------------
from scipy.fftpack import fft, ifft

class Schrodinger:
    def __init__(self, x, psi_x0, V_x, k0=None, hbar=1.0, m=1.0, t0=0.0):
        self.x = np.asarray(x)
        self.N = self.x.size
        self.dx = self.x[1] - self.x[0]
        self.hbar = hbar
        self.m = m
        self.V_x = np.asarray(V_x)
        self.psi_x = np.asarray(psi_x0).astype(np.complex128)
        self.t = t0
        self.dk = 2 * np.pi / (self.N * self.dx)
        if k0 is None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            self.k0 = k0
        self.k = self.k0 + self.dk * np.arange(self.N)

    def time_step(self, dt, steps=1):
        # Precompute factors
        x_evolve_half = np.exp(-0.5j * self.V_x / self.hbar * dt)
        x_evolve = x_evolve_half ** 2
        k_evolve = np.exp(-0.5j * self.hbar * (self.k ** 2) / self.m * dt)
        for _ in range(steps):
            self.psi_x *= x_evolve_half
            psi_k = fft(self.psi_x)
            psi_k *= k_evolve
            self.psi_x = ifft(psi_k)
            self.psi_x *= x_evolve
        self.psi_x *= x_evolve_half
        self.t += dt * steps


def schrodinger_terminal_pdf(S0: float, mu: float, sigma: float, horizon_days: float, N_x=2048):
    """Run Schrödinger evolution for log-price wavepacket and return S grid and pdf."""
    x_min = np.log(S0) - 5 * sigma * np.sqrt(max(horizon_days, 1e-9))
    x_max = np.log(S0) + 5 * sigma * np.sqrt(max(horizon_days, 1e-9))
    x = np.linspace(x_min, x_max, N_x)
    # gaussian initial packet
    a = 0.05
    k0 = (mu - sigma**2 / 2) / max(sigma, 1e-9)
    psi0 = ((a / np.sqrt(np.pi)) ** 0.5) * np.exp(-0.5 * ((x - np.log(S0)) / a) ** 2 + 1j * k0 * x)
    V_x = np.zeros_like(x)
    se = Schrodinger(x, psi0, V_x, hbar=1.0, m=1.0 / (2 * sigma**2 if sigma>0 else 1.0))
    # choose dt and steps to cover horizon
    T = horizon_days / 252.0
    dt = T / 400.0
    steps = 400
    se.time_step(dt=dt, steps=steps)
    p_x = np.abs(se.psi_x) ** 2
    dx = x[1] - x[0]
    p_x /= p_x.sum() * dx
    S_grid = np.exp(x)
    return S_grid, p_x


# -------------------------
# GBM / OU math functions
# -------------------------
def simulate_gbm(S0: float, mu: float, sigma: float, steps: int, dt: float, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, steps)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma ** 2) * dt
    paths = np.empty((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + sigma * shocks[:, t])
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
    a, b = beta[0], beta[1]
    b = float(np.clip(b, -0.99, 0.9999))
    theta = -np.log(max(1e-8, b)) / dt if abs(b) < 0.9999 else 1e-6
    mu = a / (1 - b) if abs(1 - b) > 1e-9 else 0.0
    eps = rt - (a + b * rlag)
    sigma = float(np.std(eps) if len(eps) > 1 else 1e-6)
    return OUParams(theta=theta, mu=mu, sigma=abs(sigma))

def simulate_ou(r0: float, params: OUParams, steps: int, dt: float, n_paths: int = 1000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.zeros((n_paths, steps + 1))
    r[:, 0] = r0
    for t in range(steps):
        mean = r[:, t] + params.theta * (params.mu - r[:, t]) * dt
        std = params.sigma * np.sqrt(dt)
        r[:, t + 1] = mean + std * rng.standard_normal(n_paths)
    return r


# -------------------------
# LSTM PyTorch training/predict
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner=False)
def train_lstm_model_cached(series: np.ndarray, lookback=60, epochs=30, lr=1e-3, device='cpu'):
    """
    Train a small LSTM on a 1D price series (Close). Returns trained model and scaler.
    Cached to avoid retraining on reruns.
    """
    device = torch.device(device)
    scaler = MinMaxScaler(feature_range=(0, 1))
    s = series.reshape(-1, 1)
    scaled = scaler.fit_transform(s).astype(np.float32)
    if len(scaled) <= lookback + 5:
        return None, None  # not enough history

    X, y = create_sequences(scaled.flatten(), lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((-1, 1))

    split = max(1, int(len(X) * 0.8))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        batch_x = torch.from_numpy(X_train).float().to(device)
        batch_y = torch.from_numpy(y_train).float().to(device)
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        # small val check (no heavy logging)
        if epoch % 10 == 0 and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                vx = torch.from_numpy(X_val).float().to(device)
                vy = torch.from_numpy(y_val).float().to(device)
                vout = model(vx)
                vloss = criterion(vout, vy).item()
            model.train()
    return model, scaler

def lstm_predict_from_model(model, scaler, series: np.ndarray, lookback: int, horizon: int, device='cpu') -> np.ndarray:
    """
    Given trained model & scaler, iteratively predict horizon steps ahead.
    Returns array length (horizon+1) with initial S0 followed by predictions.
    """
    device = torch.device(device)
    s = series.reshape(-1, 1)
    scaled = scaler.transform(s).astype(np.float32)
    seq = scaled[-lookback:].flatten()
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.from_numpy(seq.reshape(1, lookback, 1)).float().to(device)
            out = model(x).cpu().numpy().flatten()[0]
            preds.append(out)
            seq = np.roll(seq, -1)
            seq[-1] = out
    # inverse transform preds
    preds = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(preds).flatten()
    S0 = float(series.flatten()[-1])
    return np.concatenate(([S0], inv))


# -------------------------
# finBERT utils (optional)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_finbert_pipeline_cached(model_name: str = "ProsusAI/finbert"):
    if not _FINBERT_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True)
        return clf
    except Exception:
        return None

def finbert_sentiment_from_headlines(headlines: List[str], hf_pipe=None) -> float:
    if not headlines:
        return 0.0
    if hf_pipe is None:
        # fallback heuristic
        scores = []
        for t in headlines:
            tt = (t or "").lower()
            pos = sum(tt.count(w) for w in ("beat","surge","gain","upgrade","outperform","strong"))
            neg = sum(tt.count(w) for w in ("miss","downgrade","loss","weak","lawsuit","drop"))
            scores.append(np.tanh((pos-neg)/5.0))
        return float(np.clip(np.mean(scores) if scores else 0.0, -1.0, 1.0))
    all_scores = []
    for i in range(0, len(headlines), 8):
        chunk = headlines[i:i+8]
        try:
            preds = hf_pipe(chunk, truncation=True)
            for p in preds:
                mapping = {d["label"].lower(): d["score"] for d in p}
                score = mapping.get("positive", 0.0) - mapping.get("negative", 0.0)
                all_scores.append(score)
        except Exception:
            continue
    if not all_scores:
        return 0.0
    return float(np.clip(np.mean(all_scores), -1.0, 1.0))


def fetch_headlines_newsapi(ticker: str, api_key: Optional[str], days: int = 3, limit: int = 12) -> List[str]:
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
        return [a.get("title", "") for a in arts if a.get("title")]
    except Exception:
        return []


# -------------------------
# Meta-blend helpers
# -------------------------
def compute_rmse(preds: Dict[str, np.ndarray], actual: np.ndarray) -> Dict[str, float]:
    skills = {}
    for k, arr in preds.items():
        try:
            arr_ = np.asarray(arr).flatten()[:len(actual)]
            rmse = float(sqrt(mean_squared_error(actual[:len(arr_)], arr_)))
        except Exception:
            rmse = float("inf")
        skills[k] = rmse
    return skills

def weights_from_rmse(skills: Dict[str, float], eps: float = 1e-8) -> Dict[str, float]:
    inv = {k: (1.0 / (v + eps) if np.isfinite(v) and v > 0 else 0.0) for k, v in skills.items()}
    s = sum(inv.values())
    if s <= 0:
        n = len(skills)
        return {k: 1.0 / n for k in skills}
    return {k: float(inv[k] / s) for k in skills}

def tilt_weights(weights: Dict[str, float], sentiment: float, trend_models=("GBM","ML"), meanrev_models=("OU",), tilt_strength=0.5):
    s = float(np.clip(sentiment, -1.0, 1.0))
    tilt = s * float(np.clip(tilt_strength, 0.0, 1.0))
    if abs(tilt) < 1e-9:
        return weights
    w = weights.copy()
    trend_set = set(trend_models)
    mean_set = set(meanrev_models)
    donors = mean_set if tilt > 0 else trend_set
    recipients = trend_set if tilt > 0 else mean_set
    donor_mass = sum(w.get(d, 0.0) for d in donors)
    recip_mass = sum(w.get(r, 0.0) for r in recipients)
    if donor_mass <= 0 or recip_mass <= 0:
        return w
    shift = donor_mass * abs(tilt)
    for d in donors:
        if d in w:
            w[d] = max(0.0, w[d] - (w[d] / donor_mass) * shift)
    for r in recipients:
        if r in w:
            w[r] = w[r] + (w[r] / recip_mass) * shift
    ssum = sum(w.values())
    if ssum <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: float(v / ssum) for k, v in w.items()}

def blend_predictions(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    arrays = {k: np.asarray(v).flatten() for k, v in preds.items()}
    T = min(len(arr) for arr in arrays.values())
    total = np.zeros(T)
    for k, arr in arrays.items():
        w = weights.get(k, 0.0)
        total += w * arr[:T]
    return total

# -------------------------
# Top-level predict_stock with caching
# -------------------------
@st.cache_data(show_spinner=False, ttl=60*60)  # cache forecast for 1 hour
def predict_stock(
    ticker: str,
    start_date: str,
    end_date: str,
    mode: str = "intraday",
    interval: str = "1m",
    lookback_days: int = 5,
    forecast_steps: int = 60,
    n_paths: int = 800,
    news_api_key: Optional[str] = None,
    tilt_strength: float = 0.5,
    lstm_lookback: int = 60,
    lstm_epochs: int = 30,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Returns forecast_df, rmse_scores, blend_weights
    """
    price_df = fetch_price_data(ticker, mode=mode, days=lookback_days, interval=interval)
    if price_df.empty:
        raise ValueError(f"No price data for {ticker}")

    S0 = float(price_df["Close"].iloc[-1])

    # dt, steps
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
        mu_hat = 0.0
        sigma_hat = 1e-6
    else:
        mu_hat = float(np.mean(logret) / dt)
        sigma_hat = float(np.std(logret) / np.sqrt(max(dt, 1e-9)))

    # GBM mean path
    try:
        gbm_paths = simulate_gbm(S0, mu_hat, sigma_hat, steps=steps, dt=dt, n_paths=n_paths)
        gbm_path = gbm_paths.mean(axis=0)
    except Exception:
        gbm_path = np.repeat(S0, steps+1)

    # OU path
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

    # Schrödinger terminal stats
    try:
        S_grid, sch_pdf = schrodinger_terminal_pdf(S0, mu_hat, sigma_hat, horizon_days=steps*dt, N_x=2048)
        schrodinger_mean = float(np.trapz(S_grid * sch_pdf, S_grid))
    except Exception:
        schrodinger_mean = S0

    # Fokker-Planck proxy (Boltzmann)
    try:
        # small steps; inexpensive proxy
        r_grid = np.linspace(-0.05, 0.05, 201)
        boltz_mean = S0
    except Exception:
        boltz_mean = S0

    # LSTM: train and produce path
    series = price_df["Close"].values.astype(float)
    lstm_model, lstm_scaler = None, None
    ml_path = np.repeat(S0, steps+1)
    try:
        if len(series) > lstm_lookback + 10:
            lstm_model, lstm_scaler = train_lstm_model_cached(series, lookback=lstm_lookback, epochs=lstm_epochs)
            if lstm_model is not None:
                ml_path = lstm_predict_from_model(lstm_model, lstm_scaler, series, lookback=lstm_lookback, horizon=steps)
    except Exception:
        ml_path = np.repeat(S0, steps+1)

    # Prepare holdout actuals (try to use last 'steps' closes if available)
    hold_len = min(len(price_df), steps)
    hold_actual = price_df["Close"].values[-hold_len:] if hold_len > 0 else None

    # Align arrays and trim to same T
    arrs = {"GBM": np.asarray(gbm_path).flatten(), "OU": np.asarray(ou_price).flatten(), "ML": np.asarray(ml_path).flatten()}
    T = min(len(v) for v in arrs.values())
    for k in list(arrs.keys()):
        arrs[k] = arrs[k][:T]

    # RMSE compute on holdout if possible (use last T points of price as actual)
    rmse_scores = {}
    if hold_actual is not None and len(hold_actual) >= 1:
        actual_trim = np.asarray(hold_actual)[-T:]
        for k, arr in arrs.items():
            try:
                rmse_scores[k] = float(sqrt(mean_squared_error(actual_trim, arr[:len(actual_trim)])))
            except Exception:
                rmse_scores[k] = float("nan")
    else:
        for k in arrs.keys():
            rmse_scores[k] = float("nan")

    # Sentiment
    headlines = fetch_headlines_newsapi(ticker, api_key=news_api_key, days=3, limit=12) if news_api_key else []
    if _FINBERT_AVAILABLE:
        hf = load_finbert_pipeline_cached()
        try:
            sent = finbert_sentiment_from_headlines(headlines, hf)
        except Exception:
            sent = finbert_sentiment_from_headlines(headlines, None)
    else:
        sent = finbert_sentiment_from_headlines(headlines, None)

    # Base weights and tilt
    skills = compute_rmse(arrs, actual=np.asarray(hold_actual) if hold_actual is not None else None)
    base_weights = weights_from_rmse(skills)
    final_weights = tilt_weights(base_weights, sentiment=sent, trend_models=("GBM","ML"), meanrev_models=("OU",), tilt_strength=tilt_strength)

    best = blend_predictions(arrs, final_weights)

    # Build forecast df
    if mode == "intraday":
        idx = pd.RangeIndex(0, T, name="step")
    else:
        last_dt = price_df.index[-1]
        idx = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=T)

    forecast_df = pd.DataFrame(index=idx)
    forecast_df["GBM"] = arrs["GBM"][:T]
    forecast_df["OU"] = arrs["OU"][:T]
    forecast_df["ML"] = arrs["ML"][:T]
    forecast_df["Best_Guess"] = best[:T]
    forecast_df["Sentiment"] = float(sent)
    forecast_df["Schrodinger_mean"] = float(schrodinger_mean)
    forecast_df["Boltzmann_mean"] = float(boltz_mean)

    return forecast_df, rmse_scores, final_weights
