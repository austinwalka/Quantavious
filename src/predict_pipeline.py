import os
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Ornstein–Uhlenbeck fitting ---
def fit_ou_model(prices):
    log_prices = np.log(prices)
    dt = 1.0
    x = log_prices[:-1]
    dx = np.diff(log_prices)
    A = np.vstack([x, np.ones(len(x))]).T
    kappa, mu = np.linalg.lstsq(A, dx/dt, rcond=None)[0]
    sigma = np.std(dx - kappa * (mu - x) * dt) * np.sqrt(252)
    return {"kappa": kappa, "mu": mu, "sigma": sigma}

def predict_ou(params, last_price, days=5):
    kappa, mu, sigma = params["kappa"], params["mu"], params["sigma"]
    return last_price * np.exp((mu - last_price) * (1 - np.exp(-kappa*days)))

# --- Schrödinger-inspired (toy) ---
def schrodinger_wave_predict(prices, days=5):
    # FFT-based periodicity prediction
    fft_vals = np.fft.fft(prices - np.mean(prices))
    dominant_freq = np.argmax(np.abs(fft_vals[1:])) + 1
    prediction = prices[-1] + 0.1 * np.sin(2*np.pi*dominant_freq*days/len(prices))
    return prediction

# --- Gradient Boosting ---
def fit_gbm(X, y):
    gbm = GradientBoostingRegressor()
    gbm.fit(X, y)
    return gbm

# --- Main Predict Function ---
def predict_stock(ticker, retrain_if_missing=True):
    model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")

    # Try loading pre-trained model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    elif retrain_if_missing:
        # Download data
        data = yf.download(ticker, period="1y", interval="1d")
        if data.empty:
            return None

        # Prepare features
        data["Return"] = data["Close"].pct_change()
        data = data.dropna()

        X = data[["Open", "High", "Low", "Close", "Volume", "Return"]]
        y = data["Close"].shift(-1).dropna()
        X = X.iloc[:-1]

        # Train GBM
        gbm_model = fit_gbm(X, y)

        # Train OU
        ou_params = fit_ou_model(data["Close"].values)

        # Save meta model
        model = {
            "gbm": gbm_model,
            "ou": ou_params,
            "last_price": data["Close"].iloc[-1]
        }
        joblib.dump(model, model_path)
    else:
        return None

    # Predict with all models
    gbm_pred = model["gbm"].predict(
        np.array(model["last_price"]).reshape(1, -1)
        if np.isscalar(model["last_price"])
        else np.array([model["last_price"]])
    )[0]
    ou_pred = predict_ou(model["ou"], model["last_price"])
    schrod_pred = schrodinger_wave_predict(np.array([model["last_price"]] * 100))

    # Blend predictions
    blended = np.mean([gbm_pred, ou_pred, schrod_pred])

    return {
        "Predicted Price": round(float(blended), 2),
        "GBM": round(float(gbm_pred), 2),
        "OU": round(float(ou_pred), 2),
        "Schrodinger": round(float(schrod_pred), 2),
        "Last Close": round(float(model["last_price"]), 2),
        "Prediction Date": datetime.today().strftime('%Y-%m-%d')
    }
