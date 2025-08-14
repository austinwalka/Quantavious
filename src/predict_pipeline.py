"""
predict_pipeline.py
Updated for inference-only mode with pre-trained models.
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
from prophet import Prophet
from datetime import datetime, timedelta

# -------------------------
# LSTM Model class
# -------------------------
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# -------------------------
# Load pre-trained ML models
# -------------------------
def load_models(ticker, model_dir="models"):
    models = {}
    try:
        models["lgb"] = joblib.load(os.path.join(model_dir, f"{ticker}_lgb.pkl"))
    except:
        models["lgb"] = None
    try:
        models["xgb"] = joblib.load(os.path.join(model_dir, f"{ticker}_xgb.pkl"))
    except:
        models["xgb"] = None
    try:
        prophet_model = joblib.load(os.path.join(model_dir, f"{ticker}_prophet.pkl"))
        models["prophet"] = prophet_model
    except:
        models["prophet"] = None
    try:
        lstm_model = LSTMModel()
        lstm_model.load_state_dict(torch.load(os.path.join(model_dir, f"{ticker}_lstm.pt"), map_location=torch.device('cpu')))
        lstm_model.eval()
        models["lstm"] = lstm_model
    except:
        models["lstm"] = None
    return models

# -------------------------
# Math-based models
# -------------------------
def gbm_simulation(S0, mu, sigma, T, dt=1/252, n_paths=1000):
    N = int(T / dt)
    t = np.linspace(0, T, N+1)
    dW = np.random.standard_normal(size=(n_paths, N)) * np.sqrt(dt)
    W = np.cumsum(dW, axis=1)
    exp_term = (mu - 0.5*sigma**2) * t[1:]
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths,1)), exp_term[np.newaxis,:] + sigma * W]))
    return paths

def ou_process(S0, theta=0.1, mu=0, sigma=0.02, T=5, dt=1/252, n_paths=1000):
    N = int(T/dt)
    paths = np.zeros((n_paths, N+1))
    paths[:,0] = S0
    for t in range(1, N+1):
        dS = theta*(mu - paths[:,t-1])*dt + sigma*np.random.randn(n_paths)*np.sqrt(dt)
        paths[:,t] = paths[:,t-1] + dS
    return paths

# Placeholder for Schrödinger / Boltzmann etc. proxy
def quantum_proxy(S0, sigma=0.02, T=5, n_points=1000):
    # Simple normal distribution as proxy for PDF
    x = np.linspace(S0*(1-3*sigma), S0*(1+3*sigma), n_points)
    pdf = np.exp(-0.5*((x-S0)/(sigma*S0))**2)
    pdf /= pdf.sum()
    return x, pdf

# -------------------------
# Meta-blend
# -------------------------
def meta_blend(predictions_dict):
    # Simple average across available model outputs
    blended = np.zeros_like(next(iter(predictions_dict.values())))
    count = 0
    for key, arr in predictions_dict.items():
        if arr is not None:
            blended += arr
            count += 1
    if count > 0:
        blended /= count
    return blended

# -------------------------
# Predict function
# -------------------------
def predict_stock(ticker, model_dir="models"):
    print(f"Running predictions for {ticker}...")

    # Load historical price
    end = datetime.today()
    start = end - timedelta(days=365*3)
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end)
    except:
        raise RuntimeError("Error fetching data for ticker")

    S0 = df['Close'].iloc[-1]
    mu = df['Close'].pct_change().mean() * 252
    sigma = df['Close'].pct_change().std() * np.sqrt(252)
    dt = 1/252
    T = 5*dt

    # Load pre-trained models
    models = load_models(ticker, model_dir)

    predictions = {}

    # GBM
    gbm_paths = gbm_simulation(S0, mu, sigma, T, dt, n_paths=100)
    predictions["GBM"] = gbm_paths.mean(axis=0)

    # OU / Langevin
    ou_paths = ou_process(S0, theta=0.1, mu=S0, sigma=sigma, T=5, dt=dt, n_paths=100)
    predictions["OU"] = ou_paths.mean(axis=0)

    # Quantum proxy
    x, pdf = quantum_proxy(S0, sigma=sigma, T=5)
    predictions["Quantum"] = np.interp(np.arange(5), np.linspace(0, 5-1, len(x)), x)

    # Prophet
    if models.get("prophet") is not None:
        future = models["prophet"].make_future_dataframe(periods=5)
        forecast = models["prophet"].predict(future)
        predictions["Prophet"] = forecast['yhat'].iloc[-5:].values
    else:
        predictions["Prophet"] = np.full(5, S0)

    # LightGBM
    if models.get("lgb") is not None:
        X_pred = np.arange(len(df), len(df)+5).reshape(-1,1)
        predictions["LGB"] = models["lgb"].predict(X_pred)
    else:
        predictions["LGB"] = np.full(5, S0)

    # XGBoost
    if models.get("xgb") is not None:
        X_pred = np.arange(len(df), len(df)+5).reshape(-1,1)
        predictions["XGB"] = models["xgb"].predict(X_pred)
    else:
        predictions["XGB"] = np.full(5, S0)

    # LSTM
    if models.get("lstm") is not None:
        lstm_input = df['Close'].values[-10:].reshape(1,10,1)
        lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32)
        lstm_pred = []
        current_seq = lstm_input_tensor
        for _ in range(5):
            out = models["lstm"](current_seq)
            lstm_pred.append(out.item())
            next_seq = np.roll(current_seq.numpy(), -1)
            next_seq[0,-1,0] = out.item()
            current_seq = torch.tensor(next_seq, dtype=torch.float32)
        predictions["LSTM"] = np.array(lstm_pred)
    else:
        predictions["LSTM"] = np.full(5, S0)

    # Meta-blend
    meta = meta_blend(predictions)

    # Combine into DataFrame
    result_df = pd.DataFrame(index=range(5))
    for col, arr in predictions.items():
        result_df[col] = arr[:5]
    result_df["MetaBlend"] = meta

    return result_df
