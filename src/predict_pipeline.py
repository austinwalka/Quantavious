import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------
# Math-based models
# ----------------------------
def gbm_simulation(S0, mu, sigma, T=5/252, dt=1/252, n_paths=100):
    N = int(T/dt)
    t = np.linspace(0,T,N+1)
    dW = np.random.standard_normal(size=(n_paths,N))*np.sqrt(dt)
    W = np.cumsum(dW, axis=1)
    exp_term = (mu - 0.5*sigma**2)*t[1:]
    paths = S0*np.exp(np.hstack([np.zeros((n_paths,1)), exp_term[np.newaxis,:] + sigma*W]))
    return paths

def ou_simulation(S0, theta=0.1, mu=0, sigma=0.01, T=5/252, dt=1/252, n_paths=100):
    N = int(T/dt)
    paths = np.zeros((n_paths, N+1))
    paths[:,0] = S0
    for i in range(1,N+1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:,i] = paths[:,i-1] + theta*(mu - paths[:,i-1])*dt + sigma*dW
    return paths

def boltzmann_simulation(S0, kT=0.01, T=5/252, dt=1/252, n_paths=100):
    N = int(T/dt)
    paths = np.zeros((n_paths,N+1))
    paths[:,0] = S0
    for i in range(1,N+1):
        dE = np.random.normal(0, np.sqrt(kT), n_paths)
        paths[:,i] = paths[:,i-1] + dE
    return paths

def schrodinger_proxy(S0, T=5/252, dt=1/252, n_paths=100):
    # Simple random-walk proxy for quantum evolution
    N = int(T/dt)
    paths = np.zeros((n_paths,N+1))
    paths[:,0] = S0
    for i in range(1,N+1):
        paths[:,i] = paths[:,i-1] + np.random.normal(0, 0.01, n_paths)
    return paths

# ----------------------------
# Model directory
# ----------------------------
MODEL_DIR = "models"

def load_model(ticker, model_type):
    path = os.path.join(MODEL_DIR, ticker, f"{model_type}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# ----------------------------
# Meta-blender using optional finBERT weighting
# ----------------------------
def meta_blend(preds, sentiment_score=None):
    """
    preds: dict of model predictions (GBM, OU, Boltzmann, Schrödinger, LSTM, Prophet)
    sentiment_score: optional float, >0 positive, <0 negative, weight LSTM/Prophet
    """
    weights = {
        "GBM":1,
        "OU":1,
        "Boltzmann":1,
        "Schrödinger":1,
        "LSTM":2,
        "Prophet":2
    }
    if sentiment_score is not None:
        # tilt blend by sentiment
        weights["LSTM"] *= (1 + sentiment_score)
        weights["Prophet"] *= (1 + sentiment_score)

    total_weight = sum(weights.values())
    blended = sum(preds[m]*w for m,w in weights.items() if m in preds)/total_weight
    return blended

# ----------------------------
# Predict single stock
# ----------------------------
def predict_stock(ticker, forecast_days=5):
    # Download recent data
    data = yf.download(ticker, period="1y", interval="1d")
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    close = data["Close"].iloc[-1]
    sigma = data["Close"].pct_change().std()*np.sqrt(252)
    mu = data["Close"].pct_change().mean()*252

    # Load ML models
    lstm_model = load_model(ticker, "LSTM")
    prophet_model = load_model(ticker, "Prophet")

    # Math-based predictions
    preds = {}
    preds["GBM"] = np.mean(gbm_simulation(close, mu, sigma, T=forecast_days/252))
    preds["OU"] = np.mean(ou_simulation(close, mu=close, sigma=sigma, T=forecast_days/252))
    preds["Boltzmann"] = np.mean(boltzmann_simulation(close, T=forecast_days/252))
    preds["Schrödinger"] = np.mean(schrodinger_proxy(close, T=forecast_days/252))

    # LSTM prediction
    if lstm_model is not None:
        model, scaler = lstm_model
        last_20 = data["Close"].values[-20:].reshape(-1,1)
        scaled = scaler.transform(last_20)
        X = scaled.reshape(1,20,1)
        pred_scaled = model.predict(X)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
        preds["LSTM"] = pred
    else:
        preds["LSTM"] = close

    # Prophet placeholder
    preds["Prophet"] = close*1.01

    # Optional finBERT sentiment weighting
    sentiment_score = 0  # Replace with finBERT score if available

    best_guess = meta_blend(preds, sentiment_score)
    preds["Best_guess"] = best_guess

    return preds
