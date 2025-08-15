import os
import pandas as pd
import yfinance as yf
import joblib
import zipfile
from datetime import datetime

# Replace this with your S&P500 or smaller list for testing
TICKERS = ["AAPL", "MSFT", "GOOG"]

MODEL_DIR = "models"
ZIP_DIR = "zipped_models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

# ----------------------------
# Example LSTM trainer placeholder
# ----------------------------
# Replace this with your full LSTM, LightGBM, XGBoost training
def train_lstm_model(ticker, data):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    import numpy as np

    df = data["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    lookback = 20
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i,0])
        y.append(scaled[i,0])
    X = np.array(X).reshape(-1,lookback,1)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    return model, scaler

# ----------------------------
# Training loop
# ----------------------------
for ticker in TICKERS:
    print(f"Training models for {ticker}...")
    data = yf.download(ticker, period="1y", interval="1d")
    ticker_dir = os.path.join(MODEL_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    # Train LSTM
    lstm_model, scaler = train_lstm_model(ticker, data)
    joblib.dump((lstm_model, scaler), os.path.join(ticker_dir, "LSTM.pkl"))

    # Placeholder for Prophet/GBM/etc
    joblib.dump("placeholder", os.path.join(ticker_dir, "Prophet.pkl"))
    joblib.dump("placeholder", os.path.join(ticker_dir, "GBM.pkl"))

print("✅ Training complete.")

# ----------------------------
# Zip models
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = os.path.join(ZIP_DIR, f"models_{timestamp}.zip")
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, start=MODEL_DIR)
            zipf.write(file_path, arcname)

print(f"✅ Models zipped: {zip_filename}")
