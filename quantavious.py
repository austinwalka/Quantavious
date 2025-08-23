import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import yfinance as yf
import ta
import warnings
import json
import boto3
from boto3.session import Session
import logging
import argparse
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.arima.model import ARIMA
import time

# Configuration
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
FORECAST_HORIZON_DAYS = 30
FORECAST_HORIZON_HOURS = 32
LSTM_WINDOW_DAILY = 120
LSTM_WINDOW_HOURLY = 240
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 4
RETAIL_THRESHOLD = 1000  # Shares
INSTITUTIONAL_THRESHOLD = 10000  # Shares
TRADE_SAMPLE_DAYS = 7  # For trades/quotes
HOURLY_FETCH_DAYS = 30  # Extended for hourly data
API_TIMEOUT = 60  # Seconds
META_WEIGHTS = {"arima": 0.4, "lstm": 0.4, "volume": 0.2}
TRAIN_YEARS = 10
DRIVE_SAVE_DIR = "/root/quantavious_results"
ACCESS_KEY = "DO801BCYQYGPE2CXH697"
SECRET_KEY = "virbY1dJdNa+BzEVxyPBZIC/mZRcntLxLqy0H6A8QVc"
SPACE_NAME = "quantavious-data"
REGION = "nyc3"
client = RESTClient(api_key="oKCzovWve0OCkMjgJzYX7pNhTFXqswDu")
IS_COLAB = "google.colab" in sys.modules
MAX_WORKERS = 1 if IS_COLAB else min(multiprocessing.cpu_count(), 32)

# Logging setup
logging.basicConfig(
    filename="/root/quantavious.log" if not IS_COLAB else "/content/quantavious.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize S3 client
session = Session()
s3_client = session.client('s3', region_name=REGION, endpoint_url=f"https://{REGION}.digitaloceanspaces.com",
                          aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

def log_schedule():
    today = datetime.now().weekday()
    if today < 5:
        logging.info(f"Scheduled run for weekday: {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S %Z')}")
    else:
        logging.info(f"No run scheduled for weekend: {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S %Z')}")

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

def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap", None)
        if market_cap is None or market_cap <= 0:
            logging.warning(f"No valid market cap for {ticker}, using default")
            return 1e9  # Default 1B
        return market_cap
    except Exception as e:
        logging.error(f"Error fetching market cap for {ticker}: {e}")
        return 1e9

def download_bars_polygon(ticker, start_date, end_date, timeframe="day"):
    try:
        start_time = time.time()
        if timeframe == "day":
            bars = client.get_aggs(ticker=ticker, multiplier=1, timespan="day",
                                  from_=start_date, to=end_date, limit=50000)
        else:
            bars = client.get_aggs(ticker=ticker, multiplier=1, timespan="hour",
                                  from_=start_date, to=end_date, limit=50000)
        if time.time() - start_time > API_TIMEOUT:
            logging.warning(f"API timeout for {ticker} ({timeframe})")
            return None
        data = [{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        logging.info(f"Fetched {len(df)} {timeframe} bars for {ticker}")
        if len(df) < (LSTM_WINDOW_DAILY if timeframe == "day" else LSTM_WINDOW_HOURLY):
            logging.warning(f"Insufficient data for {ticker}: {len(df)}")
            return None
        return df
    except Exception as e:
        logging.warning(f"Polygon failed for {ticker}, trying yfinance: {e}")
        return download_ohlcv_yfinance(ticker, start_date, end_date, timeframe)

def download_ohlcv_yfinance(ticker, start_date, end_date, timeframe="day"):
    try:
        stock = yf.Ticker(ticker)
        if timeframe == "day":
            df = stock.history(start=start_date, end=end_date, interval="1d")
        else:
            df = stock.history(start=start_date, end=end_date, interval="1h")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "timestamp"
        logging.info(f"Fetched {len(df)} {timeframe} bars for {ticker} via yfinance")
        if len(df) < (LSTM_WINDOW_DAILY if timeframe == "day" else LSTM_WINDOW_HOURLY):
            logging.warning(f"Insufficient data for {ticker}: {len(df)}")
            return None
        return df
    except Exception as e:
        logging.error(f"yfinance failed for {ticker}: {e}")
        return None

def download_trades_polygon(ticker, start_date, end_date):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        trades = []
        current_date = start_date
        start_time = time.time()
        while current_date <= end_date:
            trade_list = client.list_trades(
                ticker=ticker,
                timestamp_gte=int(current_date.timestamp() * 1_000_000_000),
                timestamp_lte=int((current_date + timedelta(days=1)).timestamp() * 1_000_000_000),
                limit=10000
            )
            for trade in trade_list:
                trades.append({
                    "timestamp": trade.sip_timestamp,
                    "price": trade.price,
                    "size": trade.size,
                    "exchange": trade.exchange
                })
            current_date += timedelta(days=1)
            if time.time() - start_time > API_TIMEOUT:
                logging.warning(f"Trade data timeout for {ticker}")
                return None
        df = pd.DataFrame(trades)
        if df.empty:
            logging.warning(f"No trade data for {ticker}")
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        logging.error(f"Polygon trades failed for {ticker}: {e}")
        return None

def download_quotes_polygon(ticker, start_date, end_date):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        quotes = []
        current_date = start_date
        start_time = time.time()
        while current_date <= end_date:
            quote_list = client.list_quotes(
                ticker=ticker,
                timestamp_gte=int(current_date.timestamp() * 1_000_000_000),
                timestamp_lte=int((current_date + timedelta(days=1)).timestamp() * 1_000_000_000),
                limit=50000
            )
            for quote in quote_list:
                quotes.append({
                    "timestamp": quote.sip_timestamp,
                    "bid_price": quote.bid_price,
                    "ask_price": quote.ask_price,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size
                })
            current_date += timedelta(days=1)
            if time.time() - start_time > API_TIMEOUT:
                logging.warning(f"Quote data timeout for {ticker}")
                return None
        df = pd.DataFrame(quotes)
        if df.empty:
            logging.warning(f"No quote data for {ticker}")
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
        df.set_index("timestamp", inplace=True)
        logging.info(f"Fetched {len(df)} quotes for {ticker}")
        return df
    except Exception as e:
        logging.error(f"Polygon quotes failed for {ticker}: {e}")
        return None

def detect_split_orders(df_trades):
    try:
        df_trades["timestamp_sec"] = df_trades.index.astype(int) / 1e9
        features = df_trades[["timestamp_sec", "price", "size"]].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        df_trades["split_cluster"] = dbscan.fit_predict(scaled_features)
        for cluster in df_trades["split_cluster"].unique():
            if cluster == -1:
                continue
            cluster_trades = df_trades[df_trades["split_cluster"] == cluster]
            total_size = cluster_trades["size"].sum()
            if total_size > INSTITUTIONAL_THRESHOLD:
                df_trades.loc[cluster_trades.index, "is_institutional"] = True
                df_trades.loc[cluster_trades.index, "is_retail"] = False
        return df_trades.drop(columns=["timestamp_sec"])
    except Exception as e:
        logging.error(f"Split order detection error: {e}")
        return df_trades

def classify_trades(df_trades, df_quotes=None, df_bars=None):
    try:
        df_trades["is_retail"] = df_trades["size"] <= RETAIL_THRESHOLD
        df_trades["is_institutional"] = df_trades["size"] > INSTITUTIONAL_THRESHOLD
        df_trades["is_buy"] = np.nan
        classified_count = 0
        total_trades = len(df_trades)
        if df_quotes is not None and not df_quotes.empty:
            df_quotes = df_quotes.sort_index()
            df_trades = df_trades.sort_index()
            df_trades = df_trades.reset_index()
            df_quotes = df_quotes.reset_index()
            df_merged = pd.merge_asof(
                df_trades,
                df_quotes,
                on="timestamp",
                tolerance=pd.Timedelta("1s"),
                direction="backward"
            )
            df_merged.set_index("timestamp", inplace=True)
            for idx, row in df_merged.iterrows():
                if pd.notnull(row["bid_price"]) and pd.notnull(row["ask_price"]):
                    bid = row["bid_price"]
                    ask = row["ask_price"]
                    price = row["price"]
                    if price > (bid + ask) / 2:
                        df_merged.at[idx, "is_buy"] = True
                    elif price < (bid + ask) / 2:
                        df_merged.at[idx, "is_buy"] = False
                    classified_count += 1
            df_trades = df_merged[["timestamp", "price", "size", "exchange", "is_retail", "is_institutional", "is_buy"]]
            df_trades.set_index("timestamp", inplace=True)
        if df_bars is not None and not df_bars.empty and df_trades["is_buy"].isna().any():
            df_bars = df_bars.sort_index()
            df_trades = df_trades.reset_index()
            df_bars = df_bars.reset_index()
            df_merged = pd.merge_asof(
                df_trades,
                df_bars[["timestamp", "close"]],
                on="timestamp",
                tolerance=pd.Timedelta("1h"),
                direction="backward"
            )
            df_merged.set_index("timestamp", inplace=True)
            for idx, row in df_merged[df_merged["is_buy"].isna()].iterrows():
                if pd.notnull(row["close"]):
                    prev_close = row["close"]
                    price = row["price"]
                    if price > prev_close:
                        df_merged.at[idx, "is_buy"] = True
                    elif price < prev_close:
                        df_merged.at[idx, "is_buy"] = False
                    classified_count += 1
            df_trades = df_merged[["timestamp", "price", "size", "exchange", "is_retail", "is_institutional", "is_buy"]]
            df_trades.set_index("timestamp", inplace=True)
        df_trades["is_buy"] = df_trades["is_buy"].fillna(True)  # Conservative default
        df_trades = detect_split_orders(df_trades)
        logging.info(f"Classified {classified_count}/{total_trades} trades for buy/sell")
        return df_trades
    except Exception as e:
        logging.error(f"Trade classification error: {e}")
        return None

def compute_indicators(df, timeframe="day"):
    try:
        df["returns"] = df["close"].pct_change()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df["volume"] = df["volume"]
        df.dropna(inplace=True)
        if len(df) < (LSTM_WINDOW_DAILY if timeframe == "day" else LSTM_WINDOW_HOURLY):
            logging.warning(f"Insufficient data after indicators for {timeframe}: {len(df)}")
            return None
        return df
    except Exception as e:
        logging.error(f"Indicator error: {e}")
        return None

def compute_predicted_technicals(df, forecast_horizon, window, timeframe="day"):
    try:
        scaler = StandardScaler()
        features = ["rsi", "sma_20", "macd", "atr"]
        scaled_data = scaler.fit_transform(df[features])
        X, y = [], []
        for i in range(len(scaled_data) - window - forecast_horizon):
            X.append(scaled_data[i:i + window])
            y.append(scaled_data[i + window:i + window + forecast_horizon])
        if len(X) < 10:
            logging.warning(f"Insufficient data for technicals LSTM: {len(X)} samples")
            return pd.DataFrame({f"pred_{f}": [df[f].mean()] * forecast_horizon for f in features}, index=range(1, forecast_horizon + 1))
        X, y = np.array(X), np.array(y)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window, len(features)), kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(50, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(forecast_horizon * len(features))
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y.reshape(len(y), -1), epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0,
                  callbacks=[EarlyStopping(patience=5)])
        last_sequence = scaled_data[-window:].reshape(1, window, len(features))
        forecast_scaled = model.predict(last_sequence, verbose=0).reshape(forecast_horizon, len(features))
        forecast = scaler.inverse_transform(forecast_scaled)
        return pd.DataFrame({f"pred_{f}": forecast[:, i] for i, f in enumerate(features)}, index=range(1, forecast_horizon + 1))
    except Exception as e:
        logging.error(f"Technicals LSTM error: {e}")
        return pd.DataFrame({f"pred_{f}": [df[f].mean()] * forecast_horizon for f in features}, index=range(1, forecast_horizon + 1))

def compute_arima_forecast(df, forecast_horizon, timeframe="day"):
    try:
        model = ARIMA(df["close"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        return forecast.values
    except Exception as e:
        logging.error(f"ARIMA error: {e}")
        return np.array([df["close"].mean()] * forecast_horizon)

def compute_lstm_forecast(df, pred_technicals, forecast_horizon, window, timeframe="day"):
    try:
        scaler = StandardScaler()
        features = ["close", "rsi", "sma_20", "macd", "atr", "volume"]
        scaled_data = scaler.fit_transform(df[features])
        X, y = [], []
        for i in range(len(scaled_data) - window - forecast_horizon):
            X.append(scaled_data[i:i + window])
            y.append(scaled_data[i + window:i + window + forecast_horizon, 0])  # Predict 'close' only
        if len(X) < 10:
            logging.warning(f"Insufficient data for LSTM training: {len(X)} samples")
            return np.array([df["close"].mean()] * forecast_horizon)
        X, y = np.array(X), np.array(y)
        logging.info(f"LSTM training data shapes - X: {X.shape}, y: {y.shape}")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window, len(features)), kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(50, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(forecast_horizon)  # Output exactly forecast_horizon values
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0,
                  callbacks=[EarlyStopping(patience=5)])
        last_sequence = scaled_data[-window:].reshape(1, window, len(features))
        forecast_scaled = model.predict(last_sequence, verbose=0)  # Shape: (1, forecast_horizon)
        forecast_scaled = forecast_scaled[0]  # Shape: (forecast_horizon,)
        forecast_padded = np.zeros((forecast_horizon, len(features)))
        forecast_padded[:, 0] = forecast_scaled  # Place forecast in 'close' column
        forecast = scaler.inverse_transform(forecast_padded)[:, 0]  # Extract 'close' column
        logging.info(f"LSTM forecast shape: {forecast.shape}")
        return forecast
    except Exception as e:
        logging.error(f"LSTM error: {e}")
        return np.array([df["close"].mean()] * forecast_horizon)

def backtest_forecast(df, forecast, timeframe="day"):
    try:
        historical = df["close"].iloc[-60:]  # Last 60 periods
        if len(historical) < len(forecast):
            logging.warning(f"Insufficient historical data for backtest: {len(historical)} < {len(forecast)}")
            return {"rmse": 0.0, "crash_prob": 0.0}
        rmse = np.sqrt(np.mean((historical[-len(forecast):] - forecast) ** 2))
        returns = pd.Series(forecast).pct_change()
        crash_prob = np.mean(returns < -0.05) if len(returns) > 1 else 0.0
        return {"rmse": rmse, "crash_prob": crash_prob}
    except Exception as e:
        logging.error(f"Backtest error: {e}")
        return {"rmse": 0.0, "crash_prob": 0.0}

def compute_correlations(output_dir=DRIVE_SAVE_DIR, forecast_horizon=FORECAST_HORIZON_DAYS, timeframe="day"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        corr_data = []
        metrics = ["arima_mean", "lstm_mean", "meta_blended", "retail_volume", "inst_volume", "retail_buy_pct", "inst_buy_pct"]
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
                    continue
                combined_df = pd.DataFrame({
                    "arima_mean": df_forecast["arima_mean"],
                    "lstm_mean": df_forecast["lstm_mean"],
                    "meta_blended": df_forecast["meta_blended"],
                    "retail_volume": df_retail["retail_volume"],
                    "inst_volume": df_retail["inst_volume"],
                    "retail_buy_pct": df_retail["retail_buy_pct"],
                    "inst_buy_pct": df_retail["inst_buy_pct"]
                }).dropna()
                if len(combined_df) < 2:
                    continue
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
            logging.info(f"Correlations saved to {corr_path}")
            return avg_corr_df
        logging.warning(f"No valid data for correlations ({timeframe})")
        return None
    except Exception as e:
        logging.error(f"Error in compute_correlations: {e}")
        return None

def compute_portfolio_aggregates(output_dir=DRIVE_SAVE_DIR, forecast_horizon=FORECAST_HORIZON_DAYS, timeframe="day"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        agg_data = {
            "day": list(range(1, forecast_horizon + 1)),
            "avg_arima_mean": [],
            "avg_lstm_mean": [],
            "avg_meta_blended": [],
            "wtd_avg_arima_mean": [],
            "wtd_avg_lstm_mean": [],
            "wtd_avg_meta_blended": [],
            "total_retail_volume": [],
            "total_inst_volume": [],
            "avg_retail_buy_pct": [],
            "avg_inst_buy_pct": [],
            "wtd_avg_retail_buy_pct": [],
            "wtd_avg_inst_buy_pct": [],
            "avg_crash_prob": []
        }
        ticker_data = []
        total_market_cap = 0
        for ticker_dir in os.listdir(output_dir):
            base_dir = os.path.join(output_dir, ticker_dir, timeframe)
            if not os.path.isdir(base_dir):
                continue
            forecast_file = os.path.join(base_dir, f"forecast_{'5d_hourly' if timeframe == 'hour' else '30d'}.csv")
            retail_forecast_file = os.path.join(base_dir, "retail_inst_forecast.csv")
            meta_file = os.path.join(base_dir, "meta.json")
            if os.path.exists(forecast_file) and os.path.exists(retail_forecast_file) and os.path.exists(meta_file):
                df_forecast = pd.read_csv(forecast_file)
                df_retail = pd.read_csv(retail_forecast_file)
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                if len(df_forecast) < forecast_horizon or len(df_retail) < forecast_horizon:
                    logging.warning(f"Insufficient data for {ticker_dir} in portfolio aggregates")
                    continue
                market_cap = get_market_cap(ticker_dir)
                ticker_data.append({
                    "ticker": ticker_dir,
                    "market_cap": market_cap,
                    "arima_mean": df_forecast["arima_mean"].values,
                    "lstm_mean": df_forecast["lstm_mean"].values,
                    "meta_blended": df_forecast["meta_blended"].values,
                    "retail_volume": df_retail["retail_volume"].values,
                    "inst_volume": df_retail["inst_volume"].values,
                    "retail_buy_pct": df_retail["retail_buy_pct"].values,
                    "inst_buy_pct": df_retail["inst_buy_pct"].values,
                    "crash_prob": [meta["backtest"]["crash_prob"]] * forecast_horizon
                })
                total_market_cap += market_cap
        if not ticker_data:
            logging.warning(f"No valid data for portfolio aggregates ({timeframe})")
            return None
        for key in ["arima_mean", "lstm_mean", "meta_blended", "retail_buy_pct", "inst_buy_pct", "crash_prob"]:
            values = [d[key] for d in ticker_data]
            if values:
                agg_data[f"avg_{key}"] = np.mean(values, axis=0)
                if key in ["arima_mean", "lstm_mean", "meta_blended", "retail_buy_pct", "inst_buy_pct"]:
                    weights = [d["market_cap"] / total_market_cap for d in ticker_data]
                    agg_data[f"wtd_avg_{key}"] = np.average(values, axis=0, weights=weights)
            else:
                agg_data[f"avg_{key}"] = [0] * forecast_horizon
                if key in ["arima_mean", "lstm_mean", "meta_blended", "retail_buy_pct", "inst_buy_pct"]:
                    agg_data[f"wtd_avg_{key}"] = [0] * forecast_horizon
        for key in ["retail_volume", "inst_volume"]:
            values = [d[key] for d in ticker_data]
            agg_data[f"total_{key}"] = np.sum(values, axis=0) if values else [0] * forecast_horizon
        agg_df = pd.DataFrame(agg_data)
        agg_path = os.path.join(output_dir, f"portfolio_aggregates_{timeframe}.csv")
        safe_save(agg_df, agg_path)
        logging.info(f"Portfolio aggregates saved to {agg_path}")
        return agg_df
    except Exception as e:
        logging.error(f"Error in compute_portfolio_aggregates: {e}")
        return None

def run_ticker(ticker, start_date, end_date, forecast_horizon, window, timeframe="day"):
    try:
        logging.info(f"Processing {ticker} ({timeframe})...")
        df = download_bars_polygon(ticker, start_date, end_date, timeframe)
        if df is None:
            logging.warning(f"Data download failed for {ticker}")
            forecast_df = pd.DataFrame({
                "day": range(1, forecast_horizon + 1),
                "arima_mean": [0.0] * forecast_horizon,
                "lstm_mean": [0.0] * forecast_horizon,
                "meta_blended": [0.0] * forecast_horizon
            })
            retail_inst_df = pd.DataFrame({
                "day": range(1, forecast_horizon + 1),
                "retail_volume": [0.0] * forecast_horizon,
                "inst_volume": [0.0] * forecast_horizon,
                "retail_buy_pct": [50.0] * forecast_horizon,
                "inst_buy_pct": [50.0] * forecast_horizon
            })
            technicals_df = pd.DataFrame({
                "day": range(1, forecast_horizon + 1),
                "pred_rsi": [50.0] * forecast_horizon,
                "pred_sma_20": [0.0] * forecast_horizon,
                "pred_macd": [0.0] * forecast_horizon,
                "pred_atr": [0.0] * forecast_horizon
            })
            base_dir = os.path.join(DRIVE_SAVE_DIR, ticker, timeframe)
            forecast_path = os.path.join(base_dir, f"forecast_{'5d_hourly' if timeframe == 'hour' else '30d'}.csv")
            retail_inst_path = os.path.join(base_dir, "retail_inst_forecast.csv")
            technicals_path = os.path.join(base_dir, "predicted_technicals.csv")
            meta_path = os.path.join(base_dir, "meta.json")
            safe_save(forecast_df, forecast_path)
            safe_save(retail_inst_df, retail_inst_path)
            safe_save(technicals_df, technicals_path)
            meta = {"backtest": {"crash_prob": 0.0, "rmse": 0.0}}
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            upload_to_spaces(meta_path, f"{ticker}/{timeframe}/meta.json")
            return {"ticker": ticker, "status": "failed", "error": "Data download failed"}
        
        df_indicators = compute_indicators(df, timeframe)
        if df_indicators is None:
            logging.warning(f"Indicator computation failed for {ticker}")
            forecast_df = pd.DataFrame({
                "day": range(1, forecast_horizon + 1),
                "arima_mean": [df["close"].mean()] * forecast_horizon,
                "lstm_mean": [df["close"].mean()] * forecast_horizon,
                "meta_blended": [df["close"].mean()] * forecast_horizon
            })
            retail_inst_df = pd.DataFrame({
                "day": range(1, forecast_horizon + 1),
                "retail_volume": [0.0] * forecast_horizon,
                "inst_volume": [0.0] * forecast_horizon,
                "retail_buy_pct": [50.0] * forecast_horizon,
                "inst_buy_pct": [50.0] * forecast_horizon
            })
            technicals_df = pd.DataFrame({
                "day": range(1, forecast_horizon + 1),
                "pred_rsi": [50.0] * forecast_horizon,
                "pred_sma_20": [df["close"].mean()] * forecast_horizon,
                "pred_macd": [0.0] * forecast_horizon,
                "pred_atr": [0.0] * forecast_horizon
            })
            base_dir = os.path.join(DRIVE_SAVE_DIR, ticker, timeframe)
            forecast_path = os.path.join(base_dir, f"forecast_{'5d_hourly' if timeframe == 'hour' else '30d'}.csv")
            retail_inst_path = os.path.join(base_dir, "retail_inst_forecast.csv")
            technicals_path = os.path.join(base_dir, "predicted_technicals.csv")
            meta_path = os.path.join(base_dir, "meta.json")
            safe_save(forecast_df, forecast_path)
            safe_save(retail_inst_df, retail_inst_path)
            safe_save(technicals_df, technicals_path)
            meta = {"backtest": {"crash_prob": 0.0, "rmse": 0.0}}
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            upload_to_spaces(meta_path, f"{ticker}/{timeframe}/meta.json")
            return {"ticker": ticker, "status": "failed", "error": "Indicator computation failed"}
        
        pred_technicals = compute_predicted_technicals(df_indicators, forecast_horizon, window, timeframe)
        arima_forecast = compute_arima_forecast(df_indicators, forecast_horizon, timeframe)
        lstm_forecast = compute_lstm_forecast(df_indicators, pred_technicals, forecast_horizon, window, timeframe)
        
        trade_start_date = (datetime.now() - timedelta(days=TRADE_SAMPLE_DAYS)).strftime("%Y-%m-%d")
        df_trades = download_trades_polygon(ticker, trade_start_date, end_date)
        if df_trades is None:
            logging.warning(f"Trade data download failed for {ticker}")
            retail_volume = inst_volume = 0
            retail_buy_pct = inst_buy_pct = 50.0
        else:
            df_quotes = download_quotes_polygon(ticker, trade_start_date, end_date)
            df_trades = classify_trades(df_trades, df_quotes, df)
            if df_trades is None:
                logging.warning(f"Trade classification failed for {ticker}")
                retail_volume = inst_volume = 0
                retail_buy_pct = inst_buy_pct = 50.0
            else:
                retail_trades = df_trades[df_trades["is_retail"]]
                inst_trades = df_trades[df_trades["is_institutional"]]
                retail_volume = retail_trades["size"].sum() if not retail_trades.empty else 0
                inst_volume = inst_trades["size"].sum() if not inst_trades.empty else 0
                retail_buy_pct = (retail_trades["is_buy"].mean() * 100) if not retail_trades.empty else 50.0
                inst_buy_pct = (inst_trades["is_buy"].mean() * 100) if not inst_trades.empty else 50.0
                logging.info(f"{ticker} retail_buy_pct: {retail_buy_pct:.2f}, inst_buy_pct: {inst_buy_pct:.2f}")
        
        forecast_df = pd.DataFrame({
            "day": range(1, forecast_horizon + 1),
            "arima_mean": arima_forecast,
            "lstm_mean": lstm_forecast,
            "meta_blended": META_WEIGHTS["arima"] * arima_forecast +
                           META_WEIGHTS["lstm"] * lstm_forecast +
                           META_WEIGHTS["volume"] * df_indicators["volume"].mean()
        })
        retail_inst_df = pd.DataFrame({
            "day": range(1, forecast_horizon + 1),
            "retail_volume": [retail_volume] * forecast_horizon,
            "inst_volume": [inst_volume] * forecast_horizon,
            "retail_buy_pct": [retail_buy_pct] * forecast_horizon,
            "inst_buy_pct": [inst_buy_pct] * forecast_horizon
        })
        technicals_df = pred_technicals
        backtest_results = backtest_forecast(df_indicators, lstm_forecast, timeframe)
        
        base_dir = os.path.join(DRIVE_SAVE_DIR, ticker, timeframe)
        forecast_path = os.path.join(base_dir, f"forecast_{'5d_hourly' if timeframe == 'hour' else '30d'}.csv")
        indicators_path = os.path.join(base_dir, "indicators.csv")
        retail_inst_path = os.path.join(base_dir, "retail_inst_forecast.csv")
        technicals_path = os.path.join(base_dir, "predicted_technicals.csv")
        meta_path = os.path.join(base_dir, "meta.json")
        
        safe_save(forecast_df, forecast_path)
        safe_save(df_indicators.reset_index(), indicators_path)
        safe_save(retail_inst_df, retail_inst_path)
        safe_save(technicals_df, technicals_path)
        
        meta = {"backtest": backtest_results}
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        upload_to_spaces(meta_path, f"{ticker}/{timeframe}/meta.json")
        
        return {"ticker": ticker, "status": "success", "error": ""}
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return {"ticker": ticker, "status": "failed", "error": str(e)}

def run_batch(tickers, start_date, end_date, forecast_horizon, window, timeframe="day"):
    try:
        if IS_COLAB:
            return [run_ticker(ticker, start_date, end_date, forecast_horizon, window, timeframe) for ticker in tickers]
        else:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                return list(executor.map(run_ticker, tickers,
                                        [start_date] * len(tickers),
                                        [end_date] * len(tickers),
                                        [forecast_horizon] * len(tickers),
                                        [window] * len(tickers),
                                        [timeframe] * len(tickers)))
    except Exception as e:
        logging.error(f"Error in run_batch: {e}")
        return [{"ticker": ticker, "status": "failed", "error": str(e)} for ticker in tickers]

def main(argv=None):
    parser = argparse.ArgumentParser(description="Quantavious stock forecasting")
    parser.add_argument("--tickers", type=str, default="AAPL", help="Comma-separated list of tickers (e.g., AAPL)")
    args, _ = parser.parse_known_args(argv)

    log_schedule()
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    if datetime.now().weekday() >= 5 and not IS_COLAB:
        logging.info("Exiting: No run on weekends")
        return

    TICKERS = args.tickers.split(",")
    if not TICKERS:
        logging.error("No tickers provided. Exiting.")
        sys.exit(1)

    start_date_daily = (datetime.now() - timedelta(days=TRAIN_YEARS * 365)).strftime("%Y-%m-%d")
    start_date_hourly = (datetime.now() - timedelta(days=HOURLY_FETCH_DAYS)).strftime("%Y-%m-%d")

    logging.info("Starting daily batch...")
    daily_results = run_batch(TICKERS, start_date_daily, END_DATE, FORECAST_HORIZON_DAYS, LSTM_WINDOW_DAILY, "day")
    compute_correlations(forecast_horizon=FORECAST_HORIZON_DAYS, timeframe="day")
    compute_portfolio_aggregates(forecast_horizon=FORECAST_HORIZON_DAYS, timeframe="day")

    logging.info("Starting hourly batch...")
    hourly_results = run_batch(TICKERS, start_date_hourly, END_DATE, FORECAST_HORIZON_HOURS, LSTM_WINDOW_HOURLY, "hour")
    compute_correlations(forecast_horizon=FORECAST_HORIZON_HOURS, timeframe="hour")
    compute_portfolio_aggregates(forecast_horizon=FORECAST_HORIZON_HOURS, timeframe="hour")

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
    main(sys.argv)