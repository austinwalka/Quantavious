# src/predict_pipeline.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

# Try to import TensorFlow/Keras (optional)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Try to import FinBERT (optional)
FINBERT_AVAILABLE = False
finbert = None
try:
    # try common name first
    from finbert_embedding import FinBertEmbedding as _FinBertEmbedding
    finbert = _FinBertEmbedding()
    FINBERT_AVAILABLE = True
except Exception:
    try:
        from finbert_embedding.embedding import FinbertEmbedding as _FinBertEmbedding2
        finbert = _FinBertEmbedding2()
        FINBERT_AVAILABLE = True
    except Exception:
        FINBERT_AVAILABLE = False
        finbert = None

# ---------------------
# Math-based Models
# ---------------------
def gbm_forecast(S0, mu, sigma, T=5, steps=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        dS = mu * prices[-1] * dt + sigma * prices[-1] * np.random.normal() * np.sqrt(dt)
        prices.append(prices[-1] + dS)
    return [float(x) for x in prices[1:]]

def langevin_forecast(S0, theta, mu, sigma, T=5, steps=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        dS = theta * (mu - prices[-1]) * dt + sigma * np.random.normal() * np.sqrt(dt)
        prices.append(prices[-1] + dS)
    return [float(x) for x in prices[1:]]

def boltzmann_forecast(S0, sigma, steps=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    prices = [S0]
    for _ in range(steps):
        dS = sigma * np.random.normal()
        prices.append(prices[-1] + dS)
    return [float(x) for x in prices[1:]]

def schrodinger_forecast(S0, steps=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sigma = 0.02 * S0
    prices = [S0]
    for _ in range(steps):
        prices.append(prices[-1] + np.random.normal(0, sigma))
    return [float(x) for x in prices[1:]]

# ---------------------
# Technical Indicators
# ---------------------
def compute_indicators(df):
    # expects df with column 'Close' and a DatetimeIndex or Date column
    df = df.copy()
    try:
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
    except Exception:
        # if ta fails, just forward fill basic missing columns
        for col in ['RSI','MACD','MACD_signal','SMA20','BB_upper','BB_lower']:
            df[col] = np.nan
    return df.fillna(method='bfill').fillna(method='ffill')

# ---------------------
# LSTM Forecast (optional)
# ---------------------
def build_and_fit_lstm(X, y, window=10, epochs=10):
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(32, input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.01), loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

def lstm_forecast(df, steps=5, feature_col='Close', window=10, epochs=10, ticker=None, pretrained_dir="models/lstm"):
    """
    If TF_AVAILABLE == True: trains a small LSTM on the provided df and returns `steps` forecasts.
    If TF_AVAILABLE == False but a pretrained model exists at models/lstm/{ticker}.h5 and TF available for loading,
    we attempt to load it (requires TF). Otherwise fall back to repeating last close.
    """
    # fallback simple behavior
    last_val = float(df[feature_col].iloc[-1])
    if not TF_AVAILABLE:
        # If no TF, try to see if a saved model exists and TF can be imported for loading
        model_path = os.path.join(pretrained_dir, f"{ticker}.h5") if ticker else None
        if model_path and os.path.exists(model_path):
            try:
                # Attempt to lazy import load_model if TF becomes available at runtime
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                data = df[[feature_col]].values
                last_window = data[-window:].reshape(1, window, 1)
                preds = []
                for _ in range(steps):
                    pred = model.predict(last_window, verbose=0)[0,0]
                    preds.append(float(pred))
                    last_window = np.roll(last_window, -1)
                    last_window[0,-1,0] = pred
                return preds
            except Exception:
                return [last_val] * steps
        else:
            return [last_val] * steps

    # If TF available, train quickly (small epochs by default)
    data = df[[feature_col]].values  # shape (n,1)
    if len(data) <= window:
        return [last_val] * steps

    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, 0])
        y.append(data[i+window, 0])
    X, y = np.array(X), np.array(y)
    model = build_and_fit_lstm(X, y, window=window, epochs=epochs)

    # Rolling predict
    last_window = data[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(steps):
        pred = model.predict(last_window, verbose=0)[0,0]
        preds.append(float(pred))
        last_window = np.roll(last_window, -1)
        last_window[0,-1,0] = pred

    # Optionally save model for reuse (if TF available)
    if ticker:
        try:
            os.makedirs(pretrained_dir, exist_ok=True)
            model.save(os.path.join(pretrained_dir, f"{ticker}.h5"))
        except Exception:
            pass

    return preds

# ---------------------
# Meta-blender
# ---------------------
def meta_blender(preds_dict, weights=None):
    """
    preds_dict: {model_name: [p1, p2, ...]}
    weights: optional dict of positive weights per model_name
    """
    # ensure preds are numpy arrays and same length
    lengths = {k: len(v) for k, v in preds_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("All prediction arrays must have same length")
    n = next(iter(lengths.values()))
    arrays = {k: np.asarray(v, dtype=float) for k, v in preds_dict.items()}

    if weights is None:
        weights = {k: 1.0 for k in preds_dict.keys()}
    # normalize weights
    total = float(sum(weights.values()))
    if total == 0:
        weights = {k: 1.0 for k in preds_dict.keys()}
        total = float(sum(weights.values()))
    blended = np.zeros(n, dtype=float)
    for k, arr in arrays.items():
        w = weights.get(k, 0.0) / total
        blended += arr * w
    return [float(x) for x in blended.tolist()]

# ---------------------
# FinBERT helper (optional)
# ---------------------
def get_sentiment_score(ticker, num_articles=5, use_finbert=True):
    """
    Returns a scalar sentiment score. If FinBERT isn't available or use_finbert=False, returns 0.
    """
    if not use_finbert or not FINBERT_AVAILABLE or finbert is None:
        return 0.0
    try:
        import requests
        from bs4 import BeautifulSoup
        url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Simple extraction of headlines; fallback if none found
        headlines = [t.get_text() for t in soup.find_all('h3')][:num_articles]
        if not headlines:
            return 0.0
        embeddings = finbert.sentence_vector(headlines)
        # embeddings may be list-of-arrays, take mean of means
        means = [np.mean(np.asarray(e)) for e in embeddings]
        return float(np.mean(means))
    except Exception:
        return 0.0

# ---------------------
# Helper: predict from preloaded df (used by backtest)
# ---------------------
def predict_stock_from_df(df, days=5, use_lstm=True, use_finbert=True, ticker=None):
    """
    Internal helper that mirrors predict_stock but accepts a preloaded dataframe (training data).
    df: DataFrame with 'Close' column and a DateTime index (or similar).
    """
    try:
        df = compute_indicators(df)
        last_price = float(df['Close'].iloc[-1])
        mu = float(np.log(df['Close']).diff().dropna().mean() if len(df) > 1 else 0.0)
        sigma = float(df['Close'].pct_change().dropna().std() if len(df) > 1 else 0.001)

        # Math models
        gbm = gbm_forecast(last_price, mu, sigma, T=days, steps=days)
        langevin = langevin_forecast(last_price, theta=0.1, mu=float(df['Close'].mean()), sigma=sigma, T=days, steps=days)
        boltz = boltzmann_forecast(last_price, sigma=0.02 * last_price, steps=days)
        schr = schrodinger_forecast(last_price, steps=days)

        # ML models
        if use_lstm:
            lstm_preds = lstm_forecast(df, steps=days, ticker=ticker)
        else:
            lstm_preds = [last_price] * days

        # Gradient Boosting / GBR fallback
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values.ravel()
        try:
            gbr = GradientBoostingRegressor(n_estimators=50).fit(X, y)
            gbr_preds = gbr.predict(np.arange(len(df), len(df) + days).reshape(-1, 1)).tolist()
        except Exception:
            gbr_preds = [last_price] * days

        preds_dict = {
            "GBM": gbm,
            "Langevin": langevin,
            "Boltzmann": boltz,
            "Schrödinger": schr,
            "LSTM": lstm_preds,
            "GBR": gbr_preds
        }

        # sentiment
        sentiment = get_sentiment_score(ticker if ticker else "", use_finbert=use_finbert)
        # convert sentiment scalar into small weight tilt: e.g., sentiment in [-1,1] -> weights 1+sentiment
        weights = {k: max(0.01, 1.0 + sentiment) for k in preds_dict.keys()}

        blended = meta_blender(preds_dict, weights=weights)
        return {"Predictions": blended, "Individual": preds_dict}
    except Exception as e:
        print(f"[predict_stock_from_df] error: {e}")
        return None

# ---------------------
# Public API: predict_stock
# ---------------------
def predict_stock(ticker, days=5, use_lstm=True, use_finbert=True, pretrained_lstm_dir="models/lstm"):
    """
    Main user-facing function.
      - ticker: stock symbol
      - days: integer number of trading-day steps to forecast
      - use_lstm: if False, skip LSTM and fallback to simple predictors
      - use_finbert: if False, skip FinBERT sentiment
      - pretrained_lstm_dir: directory where pre-trained LSTM models (ticker.h5) may be stored
    """
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"No data for {ticker}")
            return None

        df = compute_indicators(df)
        return predict_stock_from_df(df, days=days, use_lstm=use_lstm, use_finbert=use_finbert, ticker=ticker)
    except Exception as e:
        print(f"[predict_stock] error for {ticker}: {e}")
        return None

# ---------------------
# Walk-forward Backtesting
# ---------------------
def backtest_stock(ticker, forecast_days=5, train_window=180, use_lstm=False, use_finbert=False):
    """
    Walk-forward backtest that uses predict_stock_from_df to produce forecasts from sliding windows.
    Returns mean RMSE (or None if not enough windows).
    """
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = compute_indicators(df)
        errors = []
        N = len(df)
        if N < train_window + forecast_days + 1:
            return None

        for start in range(0, N - train_window - forecast_days):
            train_df = df.iloc[start:start + train_window].copy()
            actual = df['Close'].iloc[start + train_window:start + train_window + forecast_days].values
            if len(actual) < forecast_days:
                continue
            pred_result = predict_stock_from_df(train_df, days=forecast_days, use_lstm=use_lstm, use_finbert=use_finbert, ticker=ticker)
            if pred_result is None:
                continue
            pred = np.array(pred_result['Predictions'], dtype=float)
            errors.append(np.sqrt(np.mean((pred - actual) ** 2)))
        if not errors:
            return None
        return float(np.mean(errors))
    except Exception as e:
        print(f"[backtest_stock] error: {e}")
        return None
