# src/predict_pipeline.py
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet

# --- Math Models --- #
def gbm_simulation(S0, mu, sigma, T, dt=1/252, n_paths=1000):
    N = int(T / dt)
    t = np.linspace(0, T, N+1)
    dW = np.random.standard_normal(size=(n_paths, N)) * np.sqrt(dt)
    W = np.cumsum(dW, axis=1)
    exp_term = (mu - 0.5 * sigma**2) * t[1:]
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), exp_term[np.newaxis, :] + sigma * W]))
    return paths

def ou_simulation(S0, theta=0.15, mu=0, sigma=0.2, T=5, dt=1/252, n_paths=1000):
    N = int(T / dt)
    X = np.zeros((n_paths, N+1))
    X[:, 0] = S0
    for i in range(N):
        dX = theta * (mu - X[:, i]) * dt + sigma * np.sqrt(dt) * np.random.randn(n_paths)
        X[:, i+1] = X[:, i] + dX
    return X

def schrodinger_proxy(S0, sigma=0.2, T=5, dt=1/252, n_paths=1000):
    # Placeholder: sample from GBM but can replace with full Schrödinger evolution later
    return gbm_simulation(S0, mu=0, sigma=sigma, T=T, dt=dt, n_paths=n_paths)

# --- Machine Learning Model --- #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def train_lstm(series, lookback=60, epochs=10):
    series = series.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler, lookback

def lstm_predict(model, scaler, lookback, series, forecast_days=5):
    seq = scaler.transform(series[-lookback:].reshape(-1, 1)).reshape(1, lookback, 1)
    preds = []
    for _ in range(forecast_days):
        p = model.predict(seq, verbose=0)[0,0]
        preds.append(p)
        seq = np.roll(seq, -1)
        seq[0, -1, 0] = p
    preds = np.array(preds).reshape(-1,1)
    return scaler.inverse_transform(preds).flatten()

# --- Prophet Model --- #
def prophet_predict(series, forecast_days=5):
    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    if len(df) < 2:
        return np.full(forecast_days, series.values[-1])
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    return forecast['yhat'].values[-forecast_days:]

# --- Predict Stock --- #
def predict_stock(ticker, forecast_days=5):
    try:
        data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
        if data.empty:
            raise ValueError("No data fetched")
        
        closes = data['Close']
        S0 = closes.iloc[-1]
        mu = closes.pct_change().mean() * 252
        sigma = closes.pct_change().std() * np.sqrt(252)
        
        result_df = pd.DataFrame(index=range(forecast_days))  # <- match forecast horizon
        
        # GBM
        gbm_paths = gbm_simulation(S0, mu, sigma, T=forecast_days/252, n_paths=500)
        result_df['GBM'] = gbm_paths.mean(axis=0)
        
        # OU
        ou_paths = ou_simulation(S0, theta=0.15, mu=S0, sigma=sigma, T=forecast_days/252, n_paths=500)
        result_df['OU'] = ou_paths.mean(axis=0)
        
        # Schrödinger proxy
        sch_paths = schrodinger_proxy(S0, sigma=sigma, T=forecast_days/252, n_paths=500)
        result_df['Schrodinger'] = sch_paths.mean(axis=0)
        
        # LSTM
        model, scaler, lookback = train_lstm(closes.values, epochs=5)
        result_df['LSTM'] = lstm_predict(model, scaler, lookback, closes.values, forecast_days)
        
        # Prophet
        result_df['Prophet'] = prophet_predict(closes, forecast_days)
        
        # Meta-blender: simple mean of all models
        result_df['MetaBlend'] = result_df.mean(axis=1)
        
        return result_df
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return pd.DataFrame(index=range(forecast_days))

# --- Example Usage --- #
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT']
    for t in tickers:
        df = predict_stock(t, forecast_days=5)
        print(f"\n{t} forecast:\n", df)
