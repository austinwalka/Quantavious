"""
train_all_models.py
Train Quantavius models (LightGBM, XGBoost, LSTM, Prophet) in Google Colab
and save them for inference on lightweight servers (Render, etc.)
"""

import os
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==== CONFIG ====
TICKERS = ["AAPL", "MSFT", "TSLA", "SCHW", "CRH", "GS", "MS", "AMZN", "GOOG", "NET", "NVDA", "AMD", "PLTR", "KO", "MO", "PO", "VZ", "PG", "JNJ", "ATO", "GIS", "FE", "WMT", "CVS", "UNH", "T"]   # change this list as needed
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== SIMPLE LSTM MODEL ====
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

# ==== LSTM HELPER ====
def train_lstm(prices, epochs=20):
    data = prices.values.reshape(-1, 1)
    X, y = [], []
    for i in range(len(data)-10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# ==== MAIN TRAIN FUNCTION ====
def train_models_for_ticker(ticker):
    print(f"Training models for {ticker}...")
    end = datetime.today()
    start = end - timedelta(days=365*3)
    df = yf.download(ticker, start=start, end=end)

    df.reset_index(inplace=True)
    df['ds'] = df['Date']
    df['y'] = df['Close']

    # ==== PROPHET ====
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df[['ds', 'y']])
    joblib.dump(prophet_model, os.path.join(SAVE_DIR, f"{ticker}_prophet.pkl"))

    # ==== LightGBM ====
    X = df.index.values.reshape(-1, 1)
    y = df['Close'].values
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X, y)
    joblib.dump(lgb_model, os.path.join(SAVE_DIR, f"{ticker}_lgb.pkl"))

    # ==== XGBoost ====
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, os.path.join(SAVE_DIR, f"{ticker}_xgb.pkl"))

    # ==== LSTM ====
    lstm_model = train_lstm(df['Close'])
    torch.save(lstm_model.state_dict(), os.path.join(SAVE_DIR, f"{ticker}_lstm.pt"))

    print(f"✅ Saved all models for {ticker} in {SAVE_DIR}/")


if __name__ == "__main__":
    for t in TICKERS:
        train_models_for_ticker(t)

    print("\nZipping models for download...")
    os.system(f"zip -r trained_models.zip {SAVE_DIR}")
    print("✅ Done! Download trained_models.zip from Colab sidebar.")
