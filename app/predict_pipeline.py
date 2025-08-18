# predict_pipeline.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ----------------------------
# Technical Indicators
# ----------------------------
def compute_technical_indicators(df):
    """
    Adds SMA20, Bollinger Bands, RSI, MACD and MACD signal to a dataframe with Close column
    """
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["BB_upper"] = df["SMA20"] + 2*df["Close"].rolling(20).std()
    df["BB_lower"] = df["SMA20"] - 2*df["Close"].rolling(20).std()
    
    # RSI
    rsi = RSIIndicator(df["Close"])
    df["RSI"] = rsi.rsi()
    
    # MACD
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    
    df.fillna(method="bfill", inplace=True)
    return df

# ----------------------------
# Crash Risk (EVT/GARCH-style)
# ----------------------------
def compute_crash_risk(returns, window=252):
    """
    Simple rolling percentile EVT-style crash risk
    returns: daily pct returns (pd.Series)
    window: rolling window size
    returns: pd.Series of probabilities
    """
    prob = []
    for i in range(len(returns)):
        if i < window:
            prob.append(0.01)  # small default for initial window
        else:
            rolling_window = returns[i-window:i]
            # EVT-style threshold: 0.5% left-tail
            q = np.percentile(rolling_window[rolling_window < 0], 0.5)
            risk = 1.0 if returns[i] < q else 0.0
            prob.append(risk)
    return pd.Series(prob)

# ----------------------------
# Meta-blender
# ----------------------------
def meta_blend(pred_math, pred_lstm, pred_finbert, weights=(0.2,0.5,0.3)):
    """
    Combine three model predictions into a single blended output
    """
    blended = (
        weights[0]*np.array(pred_math) +
        weights[1]*np.array(pred_lstm) +
        weights[2]*np.array(pred_finbert)
    )
    return blended

# ----------------------------
# Generate LSTM next-day forecast (generative path)
# ----------------------------
def generative_forecast_lstm(lstm_model, last_features, days=30):
    """
    Simple next-day iterative generation for LSTM
    lstm_model: trained PyTorch LSTM
    last_features: np.array shape (seq_len, features)
    returns: list of predicted prices
    """
    import torch
    lstm_model.eval()
    forecast = []
    x = torch.tensor(last_features, dtype=torch.float32).unsqueeze(0)  # batch=1
    with torch.no_grad():
        for _ in range(days):
            pred = lstm_model(x)
            forecast.append(pred.item())
            # slide window
            x = torch.roll(x, shifts=-1, dims=1)
            x[0,-1,:] = pred
    return forecast

# ----------------------------
# Backtesting
# ----------------------------
def backtest_stock(actual_prices, predicted_prices):
    """
    Wrapper around RMSE metric
    """
    return np.sqrt(np.mean((np.array(actual_prices)-np.array(predicted_prices))**2))
