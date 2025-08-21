import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta, time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
import yfinance as yf
import ta
import warnings
import json
import boto3
from boto3.session import Session
import logging
import argparse

# Disable GPU to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Logging setup
logging.basicConfig(
    filename="/root/quantavious.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
START_DATE = "2015-01-01"  # 10 years to capture earnings cycles
END_DATE = datetime.now().strftime("%Y-%m-%d")  # August 21, 2025
FORECAST_HORIZON_DAYS = 30
FORECAST_HORIZON_HOURS = 32  # ~5 days * 6.5 hours
LSTM_WINDOW_DAILY = 120  # Reduced to ensure sufficient data
LSTM_WINDOW_HOURLY = 240  # ~40 days * 6 hours
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 4
RETAIL_THRESHOLD = 1000
INSTITUTIONAL_THRESHOLD = 10000
TRADE_SAMPLE_DAYS = 30
TRADE_SAMPLE_HOURS = 60
META_WEIGHTS = {"math": 0.4, "lstm": 0.4, "volume": 0.2}
TRAIN_YEARS = 10
DRIVE_SAVE_DIR = "/root/quantavious_results"
ACCESS_KEY = "DO801BCYQYGPE2CXH697"  # Replace
SECRET_KEY = "virbY1dJdNa+BzEVxyPBZIC/mZRcntLxLqy0H6A8QVc"  # Replace
SPACE_NAME = "quantavious-data"
REGION = "nyc3"
client = RESTClient(api_key="oKCzovWve0OCkMjgJzYX7pNhTFXqswDu")  # Replace
MAX_WORKERS = min(multiprocessing.cpu_count(), 32)
BATCH_SIZE = MAX_WORKERS

# Initialize S3 client for Spaces
session = Session()
s3_client = session.client('s3', region_name=REGION, endpoint_url=f"https://{REGION}.digitaloceanspaces.com",
                          aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

def upload_to_spaces(file_path, key):
    try:
        s3_client.upload_file(file_path, SPACE_NAME, key, ExtraArgs={'ACL': 'public-read'})
        logging.info(f"Uploaded {file_path} to Spaces as {key}")
    except Exception as e:
        logging.error(f"Error uploading {file_path}: {e}")

def safe_save(df, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        key = path.replace(DRIVE_SAVE_DIR, "").lstrip("/")
        upload_to_spaces(path, key)
    except Exception as e:
        logging.error(f"Error saving {path}: {e}")

def download_bars_polygon(symbol, start_date, end_date, timeframe="day"):
    try:
        if timeframe == "day":
            bars = client.get_aggs(ticker=symbol, multiplier=1, timespan="day", from_=start_date, to=end_date, limit=5000)
        else:
            bars = client.get_aggs(ticker=symbol, multiplier=1, timespan="hour", from_=start_date, to=end_date, limit=5000)
        df = pd.DataFrame([{
            "timestamp": datetime.fromtimestamp(bar.timestamp / 1000),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        } for bar in bars])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        if timeframe == "hour":
            df = df[df["timestamp"].dt.time.between(time(9, 30), time(16, 0))]
        return df.sort_values("timestamp")
    except Exception as e:
        logging.error(f"Error downloading {symbol} ({timeframe}): {e}")
        return None

def download_ohlcv_yfinance(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert("US/Eastern")
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values("timestamp")
    except Exception as e:
        logging.error(f"yfinance fetch failed for {ticker}: {e}")
        return None

def compute_indicators(df):
    try:
        if df.empty or len(df) < 20:
            logging.warning(f"Insufficient data for indicators: {len(df)} rows")
            return None
        df_ind = df.copy()
        df_ind["returns"] = df_ind["close"].pct_change()
        df_ind["amount_change"] = df_ind["close"] - df_ind["open"]
        df_ind["pct_change"] = df_ind["amount_change"] / df_ind["open"]
        df_ind["direction"] = df_ind["pct_change"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        df_ind["mean_price"] = (df_ind["high"] + df_ind["low"] + df_ind["close"]) / 3
        df_ind["RSI"] = ta.momentum.RSIIndicator(df_ind["close"]).rsi()
        df_ind["MACD"] = ta.trend.MACD(df_ind["close"]).macd()
        df_ind["MACD_signal"] = ta.trend.MACD(df_ind["close"]).macd_signal()
        df_ind["SMA20"] = ta.trend.SMAIndicator(df_ind["close"], window=20).sma_indicator()
        df_ind["BB_upper"], df_ind["BB_lower"] = ta.volatility.BollingerBands(df_ind["close"]).bollinger_hband(), ta.volatility.BollingerBands(df_ind["close"]).bollinger_lband()
        df_ind["ATR"] = ta.volatility.AverageTrueRange(df_ind["high"], df_ind["low"], df_ind["close"]).average_true_range()
        df_ind["VWAP"] = ta.volume.VolumeWeightedAveragePrice(df_ind["high"], df_ind["low"], df_ind["close"], df_ind["volume"]).volume_weighted_average_price()
        df_ind["vol_momentum"] = df_ind["volume"].pct_change().rolling(window=10).mean()
        df_ind["vol_zscore"] = (df_ind["volume"] - df_ind["volume"].rolling(window=20).mean()) / df_ind["volume"].rolling(window=20).std()
        df_ind["normal_growth_rate"] = df_ind["returns"].rolling(window=20).mean().iloc[-1] if len(df_ind) >= 20 else 0
        df_ind["current_growth_rate"] = df_ind["returns"].iloc[-1] if len(df_ind) >= 1 else 0
        df_ind["volume_growth_signal"] = 1 if df_ind["current_growth_rate"] > df_ind["normal_growth_rate"] else 0
        return df_ind.fillna(method="ffill").fillna(0)
    except Exception as e:
        logging.error(f"Error computing indicators: {e}")
        return None

def train_lstm_model(data, window, forecast_horizon):
    try:
        features = ["returns", "mean_price", "high", "low", "amount_change", "open", "close", "pct_change", "direction"]
        data_features = data[features].dropna()
        if len(data_features) < window + forecast_horizon + 10:
            logging.warning(f"Insufficient data for LSTM training: {len(data_features)} rows")
            return None, None, None
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_features)
        X, y = [], []
        for i in range(len(scaled_data) - window - forecast_horizon):
            X.append(scaled_data[i:i + window])
            y.append(scaled_data[i + window:i + window + forecast_horizon, 0])  # Predict returns
        X, y = np.array(X), np.array(y)
        if len(X) < 10:
            logging.warning(f"Too few samples for LSTM: X={len(X)}")
            return None, None, None
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window, len(features))),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(forecast_horizon)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0)
        return model, scaler, scaled_data
    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")
        return None, None, None

def forecast_lstm(model, scaler, scaled_data, window, forecast_horizon, S0):
    try:
        last_window = scaled_data[-window:].reshape(1, window, scaled_data.shape[1])
        pred_scaled = model.predict(last_window, verbose=0)[0]  # Predicted returns
        prices = [S0]
        for ret in pred_scaled:
            prices.append(prices[-1] * (1 + ret))  # Cumulative price
        return np.array(prices[1:])
    except Exception as e:
        logging.error(f"Error forecasting LSTM: {e}")
        return np.array([S0] * forecast_horizon)

def train_hmm(data, n_components=3):
    try:
        returns = data["returns"].dropna().values
        if len(returns) < 10:
            logging.warning(f"Insufficient returns data for HMM: {len(returns)}")
            return None
        model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=2000, tol=1e-6)
        model.fit(returns.reshape(-1, 1))
        return model
    except Exception as e:
        logging.error(f"Error training HMM: {e}")
        return None

def forecast_hmm(model, data, forecast_horizon, S0):
    try:
        if model is None:
            return np.array([S0] * forecast_horizon)
        returns = data["returns"].dropna().values
        if len(returns) < 1:
            return np.array([S0] * forecast_horizon)
        last_state = model.predict(returns.reshape(-1, 1))[-1]
        sim_returns = model.sample(forecast_horizon)[0].flatten()
        prices = [S0]
        for ret in sim_returns:
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices[1:])
    except Exception as e:
        logging.error(f"Error forecasting HMM: {e}")
        return np.array([S0] * forecast_horizon)

def train_garch(data):
    try:
        returns = data["returns"].dropna() * 100
        if len(returns) < 10:
            logging.warning(f"Insufficient returns data for GARCH: {len(returns)}")
            return None
        model = arch_model(returns, vol="Garch", p=1, q=1)
        fitted = model.fit(disp="off")
        return fitted
    except Exception as e:
        logging.error(f"Error training GARCH: {e}")
        return None

def forecast_garch(model, forecast_horizon, S0):
    try:
        if model is None:
            return np.array([S0] * forecast_horizon)
        forecast = model.forecast(horizon=forecast_horizon)
        returns = forecast.mean.iloc[-1].values / 100
        prices = [S0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices[1:])
    except Exception as e:
        logging.error(f"Error forecasting GARCH: {e}")
        return np.array([S0] * forecast_horizon)

def gbm_paths(S0, mu, sigma, steps, n_paths=500):
    try:
        dt = 1 / 252  # Daily time step
        paths = np.zeros((n_paths, steps))
        paths[:, 0] = S0
        for t in range(1, steps):
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, n_paths))
        return paths
    except Exception as e:
        logging.error(f"Error in GBM paths: {e}")
        return np.zeros((n_paths, steps))

def ou_paths(S0, theta, mu, sigma, steps, n_paths=300):
    try:
        dt = 1 / 252
        paths = np.zeros((n_paths, steps))
        paths[:, 0] = S0
        for t in range(1, steps):
            paths[:, t] = paths[:, t-1] + theta * (mu - paths[:, t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, n_paths)
        return paths
    except Exception as e:
        logging.error(f"Error in OU paths: {e}")
        return np.zeros((n_paths, steps))

def boltzmann_proxy(S0, sigma, steps, n_paths=500):
    try:
        dt = 1 / 252
        paths = np.zeros((n_paths, steps))
        paths[:, 0] = S0
        for t in range(1, steps):
            paths[:, t] = paths[:, t-1] * np.exp(np.random.normal(0, sigma * np.sqrt(dt), n_paths))
        return paths
    except Exception as e:
        logging.error(f"Error in Boltzmann proxy: {e}")
        return np.zeros((n_paths, steps))

def schrodinger_proxy(S0, steps, n_paths=500):
    try:
        dt = 1 / 252
        paths = np.zeros((n_paths, steps))
        paths[:, 0] = S0
        for t in range(1, steps):
            paths[:, t] = paths[:, t-1] * np.exp(np.random.normal(0, 0.01 * np.sqrt(dt), n_paths))
        return paths
    except Exception as e:
        logging.error(f"Error in Schrodinger proxy: {e}")
        return np.zeros((n_paths, steps))

def walk_forward_backtest(data, window, forecast_horizon):
    try:
        results = []
        train_window = min(756, len(data) - forecast_horizon - 10)  # ~3 years
        if len(data) < train_window + forecast_horizon + 10:
            logging.warning(f"Insufficient data for backtest: {len(data)} rows")
            return {"fold_rmse": [], "avg_rmse": None}
        features = ["returns", "mean_price", "high", "low", "amount_change", "open", "close", "pct_change", "direction"]
        data_features = data[features].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_features)
        for i in range(0, len(data_features) - train_window - forecast_horizon, 63):  # Retrain every ~3 months
            train_data = data_features.iloc[i:i + train_window]
            test_data = data_features.iloc[i + train_window:i + train_window + forecast_horizon]["close"]
            model, _, _ = train_lstm_model(train_data, window, forecast_horizon)
            if model is None:
                continue
            last_window = scaled_data[i + train_window - window:i + train_window]
            if len(last_window) != window:
                continue
            pred = forecast_lstm(model, scaler, last_window, window, forecast_horizon, test_data.iloc[0]["close"])
            rmse = np.sqrt(np.mean((pred - test_data.values)**2))
            results.append(rmse)
        return {"fold_rmse": results, "avg_rmse": np.mean(results) if results else None}
    except Exception as e:
        logging.error(f"Error in walk-forward backtest: {e}")
        return {"fold_rmse": [], "avg_rmse": None}

def get_retail_institutional_metrics(symbol, start_date, end_date, timeframe="day"):
    try:
        if timeframe == "day":
            bars = client.get_aggs(ticker=symbol, multiplier=1, timespan="day", from_=start_date, to=end_date, limit=5000)
        else:
            bars = client.get_aggs(ticker=symbol, multiplier=1, timespan="hour", from_=start_date, to=end_date, limit=5000)
        df = pd.DataFrame([{
            "date": datetime.fromtimestamp(bar.timestamp / 1000),
            "volume": bar.volume,
            "vwap": bar.vwap
        } for bar in bars])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        if timeframe == "hour":
            df = df[df["date"].dt.time.between(time(9, 30), time(16, 0))]
        if df.empty:
            logging.warning(f"No trade data for {symbol} ({timeframe}), falling back to yfinance")
            yf_df = download_ohlcv_yfinance(symbol, start_date, end_date)
            if yf_df is None or yf_df.empty:
                return None
            df = yf_df.rename(columns={"timestamp": "date", "volume": "volume"})
            df["vwap"] = df["close"]  # Approximate VWAP
        df["trade_size"] = df["volume"] * df["vwap"]
        df["retail_volume"] = df["volume"].where(df["trade_size"] < RETAIL_THRESHOLD, 0)
        df["inst_volume"] = df["volume"].where(df["trade_size"] >= INSTITUTIONAL_THRESHOLD, 0)
        df["retail_pct"] = df["retail_volume"] / df["volume"].replace(0, np.nan)
        df["inst_pct"] = df["inst_volume"] / df["volume"].replace(0, np.nan)
        df["retail_buy_volume"] = df["retail_volume"] * 0.5
        df["inst_buy_volume"] = df["inst_volume"] * 0.5
        df["retail_sell_volume"] = df["retail_volume"] - df["retail_buy_volume"]
        df["inst_sell_volume"] = df["inst_volume"] - df["inst_buy_volume"]
        df["retail_buy_pct"] = df["retail_buy_volume"] / df["retail_volume"].replace(0, np.nan)
        df["inst_buy_pct"] = df["inst_buy_volume"] / df["inst_volume"].replace(0, np.nan)
        df["normal_growth_rate"] = df["retail_volume"].pct_change().rolling(window=20).mean().iloc[-1] if len(df) >= 20 else 0
        df["current_growth_rate"] = df["retail_volume"].pct_change().iloc[-1] if len(df) >= 1 else 0
        df["volume_growth_signal"] = 1 if df["current_growth_rate"] > df["normal_growth_rate"] else 0
        return df.fillna(method="ffill").fillna(0)
    except Exception as e:
        logging.error(f"Error getting retail/inst metrics for {symbol} ({timeframe}): {e}")
        return None

def forecast_retail_institutional(df, forecast_horizon):
    try:
        retail_pct = df["retail_pct"].iloc[-1]
        inst_pct = df["inst_pct"].iloc[-1]
        retail_buy_pct = df["retail_buy_pct"].iloc[-1]
        inst_buy_pct = df["inst_buy_pct"].iloc[-1]
        retail_volume = df["retail_volume"].iloc[-1]
        inst_volume = df["inst_volume"].iloc[-1]
        retail_buy_volume = df["retail_buy_volume"].iloc[-1]
        inst_buy_volume = df["inst_buy_volume"].iloc[-1]
        retail_sell_volume = df["retail_sell_volume"].iloc[-1]
        inst_sell_volume = df["inst_sell_volume"].iloc[-1]
        return pd.DataFrame({
            "day": range(1, forecast_horizon + 1),
            "retail_pct": [retail_pct] * forecast_horizon,
            "inst_pct": [inst_pct] * forecast_horizon,
            "retail_buy_pct": [retail_buy_pct] * forecast_horizon,
            "inst_buy_pct": [inst_buy_pct] * forecast_horizon,
            "retail_volume": [retail_volume] * forecast_horizon,
            "inst_volume": [inst_volume] * forecast_horizon,
            "retail_buy_volume": [retail_buy_volume] * forecast_horizon,
            "inst_buy_volume": [inst_buy_volume] * forecast_horizon,
            "retail_sell_volume": [retail_sell_volume] * forecast_horizon,
            "inst_sell_volume": [inst_sell_volume] * forecast_horizon
        })
    except Exception as e:
        logging.error(f"Error forecasting retail/institutional: {e}")
        return None

def compute_correlations(output_dir=DRIVE_SAVE_DIR, forecast_horizon=FORECAST_HORIZON_DAYS, timeframe="day"):
    try:
        corr_data = []
        metrics = ["math_mean", "lstm_mean", "meta_blended", "retail_volume", "inst_volume", "retail_buy_pct", "inst_buy_pct"]
        for ticker_dir in os.listdir(output_dir):
            base_dir = os.path.join(output_dir, ticker_dir, timeframe)
            if not os.path.isdir(base_dir):
                continue
            forecast_file = os.path.join(base_dir, f"forecast_{'5d_hourly' if timeframe == 'hour' else '30d'}.csv")
            retail_forecast_file = os.path.join(base_dir, "retail_inst_forecast.csv")
            if os.path.exists(forecast_file) and os.path.exists(retail_forecast_file):
                df_forecast = pd.read_csv(forecast_file)
                df_retail = pd.read_csv(retail_forecast_file)
                if len(df_forecast) < forecast_horizon or len(df_retail) < forecast_horizon:
                    logging.warning(f"Insufficient data for {ticker_dir} ({timeframe}): forecast={len(df_forecast)}, retail={len(df_retail)}")
                    continue
                combined_df = pd.DataFrame({
                    "math_mean": df_forecast["math_mean"],
                    "lstm_mean": df_forecast["lstm_mean"],
                    "meta_blended": df_forecast["meta_blended"],
                    "retail_volume": df_retail["retail_volume"],
                    "inst_volume": df_retail["inst_volume"],
                    "retail_buy_pct": df_retail["retail_buy_pct"],
                    "inst_buy_pct": df_retail["inst_buy_pct"]
                })
                corr_matrix = combined_df.corr(method="pearson")
                corr_dict = {"ticker": ticker_dir}
                for i, metric1 in enumerate(metrics):
                    for metric2 in metrics[i:]:
                        corr_dict[f"{metric1}_vs_{metric2}"] = corr_matrix.loc[metric1, metric2]
                corr_data.append(corr_dict)
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            avg_corr = corr_df.drop(columns="ticker").mean().to_dict()
            avg_corr_df = pd.DataFrame([
                {"metric_pair": k, "correlation": v} for k, v in avg_corr.items()
            ])
            corr_path = os.path.join(output_dir, f"correlation_analysis_{timeframe}.csv")
            safe_save(avg_corr_df, corr_path)
            logging.info(f"Correlation analysis saved to {corr_path}")
            return avg_corr_df
        else:
            logging.warning(f"No correlation data for {timeframe}")
            return None
    except Exception as e:
        logging.error(f"Error in compute_correlations: {e}")
        return None

def process_ticker(symbol, start_date, end_date, forecast_horizon, window, timeframe="day"):
    try:
        logging.info(f"Processing {symbol} ({timeframe})")
        df = download_bars_polygon(symbol, start_date, end_date, timeframe)
        if df is None or len(df) < window:
            logging.warning(f"Polygon failed for {symbol}, trying yfinance")
            df = download_ohlcv_yfinance(symbol, start_date, end_date)
            if df is None or len(df) < window:
                logging.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 'None'}")
                return {"symbol": symbol, "status": "failed", "error": "Insufficient data or API failure"}

        df_ind = compute_indicators(df)
        if df_ind is None:
            return {"symbol": symbol, "status": "failed", "error": "Indicator computation failed"}

        S0 = df_ind["close"].iloc[-1]
        lstm_model, scaler, scaled_data = train_lstm_model(df_ind, window, forecast_horizon)
        if lstm_model is None:
            lstm_pred = np.array([S0] * forecast_horizon)
        else:
            lstm_pred = forecast_lstm(lstm_model, scaler, scaled_data, window, forecast_horizon, S0)

        hmm_model = train_hmm(df_ind)
        hmm_pred = forecast_hmm(hmm_model, df_ind, forecast_horizon, S0)
        garch_model = train_garch(df_ind)
        garch_pred = forecast_garch(garch_model, forecast_horizon, S0)

        mu = df_ind["returns"].mean()
        sigma = df_ind["returns"].std()
        gbm_m = np.mean(gbm_paths(S0, mu, sigma, steps=forecast_horizon), axis=0)
        ou_m = np.mean(ou_paths(S0, theta=0.1, mu=df_ind["close"].mean(), sigma=sigma, steps=forecast_horizon), axis=0)
        boltz_m = np.mean(boltzmann_proxy(S0, sigma * S0, steps=forecast_horizon), axis=0)
        schr_m = np.mean(schrodinger_proxy(S0, steps=forecast_horizon), axis=0)
        math_mean = np.mean([gbm_m, ou_m, boltz_m, schr_m], axis=0)
        crash_prob = np.mean((hmm_pred / S0 - 1) < -0.05)

        retail_start = (datetime.now() - timedelta(days=TRADE_SAMPLE_DAYS if timeframe == "day" else TRADE_SAMPLE_HOURS)).strftime("%Y-%m-%d")
        retail_inst_df = get_retail_institutional_metrics(symbol, retail_start, end_date, timeframe)
        if retail_inst_df is None:
            return {"symbol": symbol, "status": "failed", "error": "Retail/institutional metrics failed"}

        retail_inst_forecast = forecast_retail_institutional(retail_inst_df, forecast_horizon)
        if retail_inst_forecast is None:
            return {"symbol": symbol, "status": "failed", "error": "Retail/institutional forecast failed"}

        wb = walk_forward_backtest(df_ind, window, forecast_horizon)
        meta = {
            "symbol": symbol,
            "timestamp": END_DATE,
            "retail_pct": retail_inst_df["retail_pct"].iloc[-1],
            "inst_pct": retail_inst_df["inst_pct"].iloc[-1],
            "retail_buy_pct": retail_inst_df["retail_buy_pct"].iloc[-1],
            "inst_buy_pct": retail_inst_df["inst_buy_pct"].iloc[-1],
            "backtest": wb
        }

        forecast_df = pd.DataFrame({
            "day" if timeframe == "day" else "hour": range(1, forecast_horizon + 1),
            "math_mean": math_mean,
            "lstm_mean": lstm_pred,
            "meta_blended": META_WEIGHTS["math"] * math_mean + META_WEIGHTS["lstm"] * lstm_pred + META_WEIGHTS["volume"] * retail_inst_df["volume"].iloc[-1] / retail_inst_df["volume"].mean(),
            "crash_prob": [crash_prob] * forecast_horizon
        })

        output_dir = os.path.join(DRIVE_SAVE_DIR, symbol, "hourly" if timeframe == "hour" else "daily")
        safe_save(forecast_df, os.path.join(output_dir, f"forecast_{'5d_hourly' if timeframe == 'hour' else '30d'}.csv"))
        safe_save(df_ind, os.path.join(output_dir, "indicators.csv"))
        safe_save(retail_inst_df, os.path.join(output_dir, "retail_inst_metrics.csv"))
        safe_save(retail_inst_forecast, os.path.join(output_dir, "retail_inst_forecast.csv"))
        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        upload_to_spaces(meta_path, f"{symbol}/{timeframe}/meta.json")

        logging.info(f"Successfully processed {symbol} ({timeframe})")
        return {"symbol": symbol, "status": "success", "rmse": wb["avg_rmse"]}
    except Exception as e:
        logging.error(f"Error processing {symbol} ({timeframe}): {e}")
        return {"symbol": symbol, "status": "failed", "error": str(e)}

def run_batch(tickers, start_date, end_date, forecast_horizon, window, timeframe="day"):
    try:
        results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_ticker, ticker, start_date, end_date, forecast_horizon, window, timeframe) for ticker in tickers]
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    logging.warning(f"Received None result for a ticker in {timeframe} batch")
        summary = pd.DataFrame(results)
        summary_path = os.path.join(DRIVE_SAVE_DIR, f"batch_summary_{timeframe}.csv")
        safe_save(summary, summary_path)
        logging.info(f"Batch summary saved for {timeframe}")
        return results
    except Exception as e:
        logging.error(f"Error in run_batch ({timeframe}): {e}")
        return []

def get_sp500_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        return df["Symbol"].tolist()[:500]
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Quantavious stock forecasting")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (e.g., AAPL,MSFT)")
    args = parser.parse_args()

    if args.tickers:
        TICKERS = args.tickers.split(",")
    else:
        TICKERS = get_sp500_tickers()
    if not TICKERS:
        logging.error("No tickers fetched. Exiting.")
        exit(1)

    start_date_daily = (datetime.now() - timedelta(days=TRAIN_YEARS * 365)).strftime("%Y-%m-%d")
    start_date_hourly = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    logging.info("Starting daily batch...")
    daily_results = run_batch(TICKERS, start_date_daily, END_DATE, FORECAST_HORIZON_DAYS, LSTM_WINDOW_DAILY, timeframe="day")
    compute_correlations(forecast_horizon=FORECAST_HORIZON_DAYS, timeframe="day")

    logging.info("Starting hourly batch...")
    hourly_results = run_batch(TICKERS, start_date_hourly, END_DATE, FORECAST_HORIZON_HOURS, LSTM_WINDOW_HOURLY, timeframe="hour")
    compute_correlations(forecast_horizon=FORECAST_HORIZON_HOURS, timeframe="hour")

    summary = {"daily": daily_results, "hourly": hourly_results}
    summary_path = os.path.join(DRIVE_SAVE_DIR, "run_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        upload_to_spaces(summary_path, "run_summary.json")
        logging.info(f"Run summary saved to {summary_path}")
    except Exception as e:
        logging.error(f"Error saving run summary: {e}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.config.threading.set_inter_op_parallelism_threads(MAX_WORKERS)
    tf.config.threading.set_intra_op_parallelism_threads(MAX_WORKERS)
    main()