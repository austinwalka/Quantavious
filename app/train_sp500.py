# ============================================================
# Quantavious Colab Trainer: S&P 500 Forecasts + Crash Risk
# ============================================================
# Features:
# - S&P 500 training loop (batched)
# - Real historical prices via yfinance
# - Technical indicators (ta)
# - Math forecasts: GBM, OU (Langevin), Volatility-spread proxy
# - LSTM with 30-day generative (autoregressive) forecast
# - Crash risk using rolling EVT (GPD POT), optional GARCH(1,1)
# - Optional FinBERT daily sentiment via yfinance news
# - Backtesting RMSE for horizons [1, 5, 15, 30]
# - Saves outputs to Google Drive for Streamlit app
# ------------------------------------------------------------
# Expected Drive structure:
# /MyDrive/quant_results/<TICKER>/
#     - forecast_30d.csv        (price path: date, close, blended, gbm, ou, lstm, finbert_adjusted)
#     - crash_30d.csv           (date, p_crash)
#     - indicators.csv          (last 180 days with indicators)
#     - backtest.json           (RMSE, details)
#     - meta.json               (ticker, last_close, timestamp)
#     - sentiment_daily.csv     (if FinBERT enabled)
# ============================================================

#Only commented out for vscode! Uncomment for colab
#!pip -q install yfinance pandas numpy scipy scikit-learn ta arch plotly tqdm --upgrade
#!pip -q install tensorflow --upgrade
# Optional sentiment (FinBERT); costs time & memory. Set ENABLE_FINBERT=False to skip.
#!pip -q install transformers torch --upgrade

import os
import io
import json
from datetime import datetime, timedelta
import time
import math
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from scipy.stats import genpareto, norm
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

# --- Technical Indicators
import ta
# --- GARCH
from arch import arch_model

# --- Optional FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Mount Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# ========================
# Configuration
# ========================
SAVE_DIR = "/content/drive/MyDrive/quant_results"
os.makedirs(SAVE_DIR, exist_ok=True)

START_DATE = "1990-01-01"
LSTM_LOOKBACK = 1250
LSTM_EPOCHS = 200            # adjust for quality/runtime
LSTM_BATCH = 32
FORECAST_DAYS = 30
ROLL_WINDOW_EVT_DAYS = 756  # ~3 years for EVT threshold
ENABLE_GARCH = True         # set False if slow
ENABLE_FINBERT = True       # set False to skip sentiment

# S&P 500 batching controls (to avoid running all 500 at once)
MAX_TICKERS_PER_RUN = 30   # change to run more/less
TICKER_OFFSET = 0          # skip first N tickers to continue batch work

# Meta-blend weights
W_MATH = 0.20
W_LSTM = 0.60
W_FINBERT = 0.20

# Crash risk threshold definition: dynamic EVT 99.5th percentile of negative returns
CRASH_QUANTILE = 0.995

# ==============================================
# Ticker helpers (incl. BRK.B -> BRK-B mapping)
# ==============================================
def normalize_ticker(t):
    t = t.strip().upper()
    # handle class shares that break on Yahoo
    t = t.replace(".", "-")
    if t == "BRK-B":   # ensure common alias
        return "BRK-B"
    return t

def get_sp500_tickers():
    """
    Fetch S&P 500 tickers from Wikipedia. Fallback to yfinance Tickers if failed.
    """
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = table[0]
        tickers = df['Symbol'].tolist()
        #tickers = ["NVDA"]  # start with 1 ticker
        tickers = [normalize_ticker(t) for t in tickers]
        return tickers
    except Exception as e:
        print("Failed to fetch S&P 500 from Wikipedia, fallback to yfinance S&P namesets.")
        # fallback minimal set
        fallback = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "BRK-B", "XOM", "JPM", "V"]
        return [normalize_ticker(t) for t in fallback]

# ===========================
# Data downloading
# ===========================
def download_prices(ticker, start=START_DATE):
    t = normalize_ticker(ticker)
    df = yf.download(t, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df.dropna()
    return df

# ===========================
# Technical Indicators
# ===========================
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out['Close'].astype(float)

    # Trend indicators
    out['SMA20']  = price.rolling(20).mean()
    out['EMA20']  = price.ewm(span=20, adjust=False).mean()
    out['SMA50']  = price.rolling(50).mean()
    out['EMA50']  = price.ewm(span=50, adjust=False).mean()

    # Volatility indicators
    out['BB_MIDDLE'] = ta.volatility.BollingerBands(price, window=20).bollinger_mavg()
    out['BB_UPPER']  = ta.volatility.BollingerBands(price, window=20).bollinger_hband()
    out['BB_LOWER']  = ta.volatility.BollingerBands(price, window=20).bollinger_lband()

    # Momentum indicators
    out['RSI14'] = ta.momentum.RSIIndicator(price, window=14).rsi()

    macd = ta.trend.MACD(price)
    out['MACD']        = macd.macd()
    out['MACD_SIGNAL'] = macd.macd_signal()
    out['MACD_HIST']   = macd.macd_diff()

    out['RET'] = price.pct_change()
    out['LOGRET'] = np.log(price).diff()

    return out

# ===========================
# Math Models (MC forecasts)
# ===========================
def estimate_drift_vol(log_prices: np.ndarray, window=252):
    """
    Estimate drift (mu) and vol (sigma) from last 'window' trading days of log returns.
    """
    log_returns = np.diff(log_prices)
    if len(log_returns) < 2:
        return 0.0, 0.0
    use = log_returns[-window:] if len(log_returns) >= window else log_returns
    mu = np.mean(use)
    sigma = np.std(use)
    return mu, sigma

def simulate_gbm_path(S0, mu, sigma, days=30, n_paths=2000, dt=1/252):
    """
    Geometric Brownian Motion Monte Carlo. Returns expected path (mean).
    """
    if sigma <= 0:
        return np.array([S0]*(days))
    paths = np.zeros((n_paths, days))
    for i in range(n_paths):
        S = S0
        for t in range(days):
            z = np.random.normal()
            S = S * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            paths[i, t] = S
    return np.mean(paths, axis=0)

def simulate_ou_path(S0, long_mean, kappa, sigma, days=30, n_paths=2000, dt=1/252):
    """
    Ornstein-Uhlenbeck on price (mean-reverting proxy).
    """
    if sigma <= 0:
        return np.array([S0]*(days))
    paths = np.zeros((n_paths, days))
    for i in range(n_paths):
        x = S0
        for t in range(days):
            z = np.random.normal()
            x = x + kappa*(long_mean - x)*dt + sigma*np.sqrt(dt)*z
            paths[i, t] = x
    return np.mean(paths, axis=0)

def volatility_spread_proxy(S0, sigma, days=30, scale=1.0):
    """
    Simple 'Boltzmann-like' uncertainty envelope proxy:
    Expands around S0 using cumulative sigma * sqrt(t) heuristic.
    We'll produce a midline equal to S0 and slope it by drift ~0
    and return the midline as a 'path'.
    """
    if sigma <= 0:
        return np.array([S0]*(days))
    t = np.arange(1, days+1)
    spread = scale * sigma * np.sqrt(t/252.0) * S0
    # We'll return an expected path as S0 (flat) + small growth from sigma
    # But not too impactful; treat as small upward drift-free path:
    path = S0 + 0.10 * spread  # mild expansion
    return path

def math_forecasts(df: pd.DataFrame, days=30):
    """
    Produce math-based 30-day forecasts: GBM, OU, Vol Spreads.
    """
    price = df['Close'].astype(float).values
    logp  = np.log(price + 1e-9)
    S0 = float(price[-1])

    mu, sigma = estimate_drift_vol(logp, window=252)
    # OU parameters: estimate long mean as SMA(252), kappa mild
    long_mean = float(pd.Series(price).rolling(252).mean().dropna().iloc[-1]) if len(price) > 252 else float(price.mean())
    kappa = 1.5  # mild mean reversion speed

    gbm_path = simulate_gbm_path(S0, mu, sigma, days=days, n_paths=2000)
    ou_path  = simulate_ou_path(S0, long_mean, kappa, sigma, days=days, n_paths=2000)
    vol_path = volatility_spread_proxy(S0, sigma, days=days, scale=1.0)

    return gbm_path, ou_path, vol_path

# ===========================
# LSTM Model
# ===========================
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_lstm_model(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

def make_supervised_series(series, lookback=LSTM_LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def train_lstm_close(df: pd.DataFrame, epochs=LSTM_EPOCHS, lookback=LSTM_LOOKBACK):
    close = df['Close'].astype(float).values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close)

    # train/test split (last 252 days test)
    split = max(lookback+1, len(scaled)-252)
    train, test = scaled[:split], scaled[split:]

    X_train, y_train = make_supervised_series(train, lookback)
    if len(test) > lookback + 1:
        X_test, y_test = make_supervised_series(test, lookback)
    else:
        X_test, y_test = None, None

    model = build_lstm_model((lookback, 1))
    es = callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=LSTM_BATCH, verbose=0, callbacks=[es])

    return model, scaler, scaled

def predict_lstm_path(model, scaled_series, lookback=LSTM_LOOKBACK, days=30):
    """
    Autoregressive 30-day forecast: feed predictions back in.
    """
    window = scaled_series[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(days):
        nxt = model.predict(window, verbose=0)[0][0]
        preds.append(nxt)
        window = np.append(window[:,1:,:], [[[nxt]]], axis=1)
    return np.array(preds).reshape(-1,1)

# ===========================
# Crash Risk (EVT + optional GARCH)
# ===========================
def fit_garch_sigma(returns):
    # returns as percentage (e.g., 0.01), not log returns
    try:
        am = arch_model(returns*100.0, p=1, q=1, mean='Constant', vol='Garch', dist='normal')
        res = am.fit(disp='off')
        sigma_last = res.conditional_volatility.iloc[-1] / 100.0  # back to fraction
        return sigma_last
    except Exception as e:
        return np.std(returns[-252:])

def evt_dynamic_threshold(returns, quantile=0.995, roll_window=ROLL_WINDOW_EVT_DAYS):
    """
    Estimate dynamic threshold for 'crash' as the 99.5% worst return using POT.
    Returns Q < 0 (negative return threshold).
    """
    # Use last 'roll_window' returns if available
    use = returns[-roll_window:] if len(returns) >= roll_window else returns
    neg = use[use < 0]  # negative returns
    if len(neg) < 50:
        # fallback: empirical quantile
        return np.quantile(use, 1 - (1-quantile))
    # Work on magnitudes for tail fit
    magn = -neg
    # Threshold u at 90th percentile of magnitudes
    u = np.quantile(magn, 0.90)
    excess = magn[magn > u] - u
    if len(excess) < 25:
        return np.quantile(use, 1 - (1-quantile))
    # Fit GPD to excess
    c, loc, scale = genpareto.fit(excess, floc=0)
    # Compute VaR at target quantile for magnitudes
    # Tail prob for POT: p_tail = (1 - p_quant) / (1 - F(u))
    F_u = (magn <= u).mean()
    p = 1 - quantile
    p_tail = p / max(1 - F_u, 1e-6)
    VaR = u + (scale / max(c, 1e-8)) * ((p_tail ** (-c)) - 1) if c != 0 else u - scale * np.log(p_tail)
    Q = -VaR  # negative threshold in returns space
    return Q

def crash_probability_forecast(df, days=30, quantile=CRASH_QUANTILE, use_garch=True):
    """
    Produce p_crash per day for next 'days' using:
    - dynamic EVT threshold Q
    - volatility forecast sigma (GARCH last sigma or recent std)
    Assume mean ~0 for short horizon; Normal approx for next-day return.
    Sigma is kept flat or mildly increasing.
    """
    rets = df['Close'].pct_change().dropna().values
    if len(rets) < 60:
        return np.array([0.0]*days)

    Q = evt_dynamic_threshold(rets, quantile=quantile)
    sigma0 = fit_garch_sigma(rets) if use_garch else np.std(rets[-252:])
    sigma0 = max(sigma0, 1e-4)

    # increase sigma slightly over horizon to reflect compounding uncertainty
    p_list = []
    for d in range(1, days+1):
        sigma_d = sigma0 * math.sqrt(d/1.0)
        # p = P(R < Q) under Normal(0, sigma_d)
        z = (Q - 0.0) / sigma_d
        p = norm.cdf(z)
        p_list.append(float(np.clip(p, 0, 1)))
    return np.array(p_list)

# ===========================
# Optional: FinBERT sentiment (daily agg)
# ===========================
_FINBERT = None
_FINBERT_TOKENIZER = None

def init_finbert():
    global _FINBERT, _FINBERT_TOKENIZER
    if _FINBERT is not None:
        return
    try:
        model_name = "ProsusAI/finbert"
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _FINBERT = AutoModelForSequenceClassification.from_pretrained(model_name)
        _FINBERT.eval()
        print("FinBERT loaded.")
    except Exception as e:
        print("Failed to init FinBERT, disabling sentiment.", e)
        _FINBERT = None
        _FINBERT_TOKENIZER = None

@torch.no_grad()
def score_sentences_finbert(texts):
    """
    Returns probabilities for classes [negative, neutral, positive]
    """
    if _FINBERT is None or _FINBERT_TOKENIZER is None:
        return None
    enc = _FINBERT_TOKENIZER(texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
    out = _FINBERT(**enc)
    probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
    return probs

def fetch_yf_news_daily_sentiment(ticker, days_back=60):
    """
    Use yfinance news (recent) and FinBERT to build daily sentiment features.
    """
    t = normalize_ticker(ticker)
    tk = yf.Ticker(t)
    news = tk.news
    if not news:
        return pd.DataFrame(columns=["date","pos","neg","neu","count","pos_mean","neg_mean","neu_mean"])
    rows = []
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    texts = []
    meta = []
    for item in news:
        ts = item.get('providerPublishTime')
        if ts is None: 
            continue
        dt = datetime.utcfromtimestamp(int(ts))
        if dt < cutoff:
            continue
        title = item.get('title', '')
        if not title:
            continue
        texts.append(title)
        meta.append(dt.date())

    if len(texts) == 0:
        return pd.DataFrame(columns=["date","pos","neg","neu","count","pos_mean","neg_mean","neu_mean"])

    probs = score_sentences_finbert(texts)
    if probs is None:
        return pd.DataFrame(columns=["date","pos","neg","neu","count","pos_mean","neg_mean","neu_mean"])

    df = pd.DataFrame({
        "date": meta,
        "neg": probs[:,0],
        "neu": probs[:,1],
        "pos": probs[:,2],
    })
    agg = df.groupby("date").agg(
        pos_mean=("pos","mean"),
        neg_mean=("neg","mean"),
        neu_mean=("neu","mean"),
        count=("pos", "count"),
    ).reset_index()
    return agg

def sentiment_to_scalar(agg_df):
    """
    Turn daily FinBERT aggregates into a scalar [0..1] sentiment weight.
    For simplicity: score = sigmoid( pos_mean - neg_mean + 0.1*log(1+count) )
    """
    if agg_df is None or agg_df.empty:
        return 0.5
    last = agg_df.iloc[-1]
    raw = float(last['pos_mean'] - last['neg_mean']) + 0.1*math.log1p(float(last['count']))
    score = 1/(1+math.exp(-3*raw))  # strong squash
    return float(np.clip(score, 0.0, 1.0))

# ===========================
# Backtesting
# ===========================
def rmse(a, f):
    a, f = np.array(a), np.array(f)
    return sqrt(mean_squared_error(a, f))

def backtest_math_only(df, horizon_list=[1,5,15,30]):
    """
    A simple backtest using math forecasts without retraining (fast, indicative).
    For each day in last 252 days, estimate mu/sigma from prior 252 and
    compute 1-step ahead (or horizon) expected price using GBM mean.
    """
    price = df['Close'].astype(float).values
    logp  = np.log(price + 1e-9)
    N = len(price)
    H = max(horizon_list)
    if N < 252 + H + 5:
        return {str(h): None for h in horizon_list}

    preds = {h: [] for h in horizon_list}
    actuals = {h: [] for h in horizon_list}

    for t in range(252, N - H):
        mu, sigma = estimate_drift_vol(logp[:t], window=252)
        S0 = price[t-1]
        # Use GBM expected value over each horizon:
        for h in horizon_list:
            # E[S_{t+h}] approx S0 * exp(mu*h*dt)
            dt = 1/252
            expected = S0 * np.exp(mu * h * dt)  # ignoring var term
            preds[h].append(expected)
            actuals[h].append(price[t+h-1])

    scores = {}
    for h in horizon_list:
        if len(preds[h])>10:
            scores[str(h)] = rmse(actuals[h], preds[h])
        else:
            scores[str(h)] = None
    return scores

def backtest_lstm(df, lookback=LSTM_LOOKBACK):
    """
    Lightweight LSTM backtest: train once on historical data up to last 252 days,
    then predict next-day (1-step) iteratively for the last 252 days using autoregressive update.
    """
    close = df['Close'].astype(float).values.reshape(-1,1)
    if len(close) < lookback + 260:
        return None

    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(close)

    split = len(scaled) - 252
    train = scaled[:split]
    test  = scaled[split - lookback:]  # include lookback context

    X_train, y_train = make_supervised_series(train, lookback)
    model = build_lstm_model((lookback,1))
    es = callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[es])

    # rolling 1-step
    window = test[:lookback].reshape(1,lookback,1)
    preds, acts = [], []
    raw_test = scaler.inverse_transform(test)  # for actual closes
    for i in range(lookback, len(test)):
        p = model.predict(window, verbose=0)[0][0]
        preds.append(p)
        acts.append(test[i][0])
        # roll
        window = np.append(window[:,1:,:], [[[p]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()
    acts  = scaler.inverse_transform(np.array(acts).reshape(-1,1)).ravel()
    return rmse(acts, preds)

# ===========================
# Per-ticker pipeline
# ===========================
def run_ticker_pipeline(ticker):
    t = normalize_ticker(ticker)
    print(f"\n=== {t} ===")
    df = download_prices(t, start=START_DATE)
    if df is None or df.empty:
        print("No price data.")
        return

    # Indicators
    df = compute_technical_indicators(df)
    df = df.dropna()
    if df.empty:
        print("Insufficient data after indicators.")
        return

    last_close = float(df['Close'].iloc[-1])

    # Math forecasts (price paths)
    gbm_path, ou_path, vol_path = math_forecasts(df, days=FORECAST_DAYS)

    # LSTM train + 30-day generative forecast
    model, scaler, scaled = train_lstm_close(df, epochs=LSTM_EPOCHS, lookback=LSTM_LOOKBACK)
    lstm_scaled_preds = predict_lstm_path(model, scaled, lookback=LSTM_LOOKBACK, days=FORECAST_DAYS)
    lstm_preds_close = scaler.inverse_transform(lstm_scaled_preds).ravel()

    # Crash probabilities for next 30 days
    p_crash = crash_probability_forecast(df, days=FORECAST_DAYS, quantile=CRASH_QUANTILE, use_garch=ENABLE_GARCH)

    # Optional FinBERT
    sentiment_daily = None
    sent_scalar = 0.5
    if ENABLE_FINBERT:
        init_finbert()
        try:
            sentiment_daily = fetch_yf_news_daily_sentiment(t, days_back=60)
            sent_scalar = sentiment_to_scalar(sentiment_daily)
        except Exception as e:
            print("FinBERT news fetch/scoring failed", e)
            sentiment_daily = None
            sent_scalar = 0.5

    # Blend price paths:
    # We'll convert each path to a % return relative to last_close, then blend
    def to_ret(path): 
        return (np.array(path) - last_close) / max(last_close,1e-6)

    r_gbm = to_ret(gbm_path)
    r_ou  = to_ret(ou_path)
    r_vol = to_ret(vol_path)
    r_math = (r_gbm + r_ou + r_vol) / 3.0  # equally combine math trio

    r_lstm = to_ret(lstm_preds_close)

    # sentiment adjusts the blend slightly: if >0.5 increases weight to math trend; else tilt toward OU (mean reversion)
    alpha = sent_scalar  # [0..1]
    r_math_adj = alpha * r_gbm + (1-alpha) * r_ou  # simple tilt between GBM & OU
    r_math_final = 0.7 * r_math + 0.3 * r_math_adj

    # FinBERT contributes as weighting not a price path, so we compute blended returns:
    blended_returns = W_MATH * r_math_final + W_LSTM * r_lstm + W_FINBERT * (alpha - 0.5) * 0.10
    # the FINBERT term is a small tilt (+/- 5%) over 30 days

    blended_price = last_close * (1 + blended_returns)
    gbm_price     = last_close * (1 + r_gbm)
    ou_price      = last_close * (1 + r_ou)
    lstm_price    = last_close * (1 + r_lstm)
    finbert_adj   = last_close * (1 + (alpha - 0.5) * 0.10)  # constant tilt line (for reference)

    # Build results DataFrames
    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "last_close": last_close,
        "blended": blended_price,
        "gbm": gbm_price,
        "ou": ou_price,
        "lstm": lstm_price,
        "finbert_adjust_ref": finbert_adj,  # static line to visualize tilt
    })

    crash_df = pd.DataFrame({
        "date": future_dates,
        "p_crash": p_crash
    })

    # indicators (last 180 days)
    ind_df = df.tail(180).copy()
    ind_df.reset_index(inplace=True)
    ind_df.rename(columns={"index":"date"}, inplace=True)
    if "Date" in ind_df.columns:
        ind_df.rename(columns={"Date":"date"}, inplace=True)

    # Backtests
    bt_math = backtest_math_only(df, horizon_list=[1,5,15,30])
    bt_lstm = backtest_lstm(df)

    backtest_report = {
        "math_rmse": bt_math,
        "lstm_rmse_1d": bt_lstm,
        "note": "Math RMSE uses GBM expected value proxy; LSTM RMSE is 1-step walk-forward with fixed model."
    }

    # sentiment daily
    if sentiment_daily is not None and not sentiment_daily.empty:
        sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])

    # Save to Drive
    out_dir = os.path.join(SAVE_DIR, t)
    os.makedirs(out_dir, exist_ok=True)

    forecast_df.to_csv(os.path.join(out_dir, "forecast_30d.csv"), index=False)
    crash_df.to_csv(os.path.join(out_dir, "crash_30d.csv"), index=False)
    ind_df.to_csv(os.path.join(out_dir, "indicators.csv"), index=False)
    with open(os.path.join(out_dir, "backtest.json"), "w") as f:
        json.dump(backtest_report, f, indent=2)
    meta = {
        "ticker": t,
        "last_close": last_close,
        "sentiment_scalar": sent_scalar,
        "updated_utc": datetime.utcnow().isoformat()
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if sentiment_daily is not None and not sentiment_daily.empty:
        sentiment_daily.to_csv(os.path.join(out_dir, "sentiment_daily.csv"), index=False)

    print(f"Saved results to {out_dir}")

# ===========================
# MAIN LOOP: S&P 500 (batched)
# ===========================
tickers = get_sp500_tickers()
tickers = tickers[TICKER_OFFSET:TICKER_OFFSET+MAX_TICKERS_PER_RUN]
print(f"Processing {len(tickers)} tickers starting at offset {TICKER_OFFSET}:")

for tk in tqdm(tickers):
    try:
        run_ticker_pipeline(tk)
    except Exception as e:
        print(f"Error processing {tk}: {e}")
        continue

print("\nAll done. Results in:", SAVE_DIR)

