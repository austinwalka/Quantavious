# src/predict_pipeline.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from math import ceil
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Local helpers (assumes these modules exist in src/)
from src.data_fetch import fetch_prices, bs_price, fetch_option_chain, enrich_chain_with_iv
from src.news_sentiment import get_sentiment_score
from src.models import LSTMModel, lstm_train_predict, gbm_simulation, black_scholes_paths, quantum_predict_distribution

try:
  Prophet = None  # type: ignore
  _HAS_PROPHET = False
except Exception:
  Prophet = None  # type: ignore
  _HAS_PROPHET = False

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def _dates_for_horizon(last_date, horizon_days):
    last = pd.to_datetime(last_date)
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=horizon_days)


def _holdout_split(df, holdout_days):
    """
    Splits df into train/test by last holdout_days business days.
    df: DataFrame with datetime index and 'Close'
    """
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")
    df = df.asfreq('B').ffill()
    if len(df) <= holdout_days + 10:
        # too short: use 80/20 split
        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]
    else:
        train = df.iloc[:-holdout_days]
        test = df.iloc[-holdout_days:]
    return train, test


def _train_lightgbm_and_forecast(train, test, horizon):
    # features: use simple lag features or index as feature
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['Close'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = LGBMRegressor()
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    # retrain on full history and forecast horizon
    X_full = np.arange(len(train) + len(test)).reshape(-1, 1)
    y_full = np.concatenate([y_train, test['Close'].values])
    model_full = LGBMRegressor()
    model_full.fit(X_full, y_full)
    X_fore = np.arange(len(X_full), len(X_full) + horizon).reshape(-1, 1)
    forecast = model_full.predict(X_fore)

    rmse = float(mean_squared_error(test['Close'].values, preds_test, squared=False))
    return forecast, rmse


def _train_xgboost_and_forecast(train, test, horizon):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['Close'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = XGBRegressor(objective='reg:squarederror', verbosity=0)
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    # full retrain
    X_full = np.arange(len(train) + len(test)).reshape(-1, 1)
    y_full = np.concatenate([y_train, test['Close'].values])
    model_full = XGBRegressor(objective='reg:squarederror', verbosity=0)
    model_full.fit(X_full, y_full)
    X_fore = np.arange(len(X_full), len(X_full) + horizon).reshape(-1, 1)
    forecast = model_full.predict(X_fore)

    rmse = float(mean_squared_error(test['Close'].values, preds_test, squared=False))
    return forecast, rmse


def _train_prophet_and_forecast(train, test, horizon):
    # Check if Prophet is available
    if not _HAS_PROPHET or Prophet is None:
        forecast = np.repeat(train['Close'].iloc[-1], horizon)
        rmse = float('nan')
        return forecast, rmse

    # Prophet requires ds/y
    df_train = train.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_test = test.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=False)
    m.fit(df_train)
    # test predictions (on test ds)
    future_test = df_test[['ds']].copy()
    pred_test = m.predict(future_test)
    preds_test = pred_test['yhat'].values

    # retrain on full history and forecast horizon
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    m_full = Prophet(daily_seasonality=False)
    m_full.fit(df_full)
    future = m_full.make_future_dataframe(periods=horizon, freq='B')  # business days
    forecast_df = m_full.predict(future)
    # pick last horizon rows
    forecast = forecast_df[['ds', 'yhat']].tail(horizon)['yhat'].values

    rmse = float(mean_squared_error(df_test['y'].values, preds_test, squared=False))
    return forecast, rmse


def _train_lstm_and_forecast(train, test, horizon, lstm_epochs=20, lookback=60, device='cpu'):
    """
    Use helper lstm_train_predict on full series for final forecast.
    For holdout RMSE, we will train on train only and predict for test period then compute rmse.
    """
    # Prepare series arrays
    train_series = train['Close'].values
    test_series = test['Close'].values

    # RMSE: train on train_series and predict len(test_series)
    try:
        _, _, _, _ = lstm_train_predict  # check existence
    except Exception:
        # fallback: naive persistence
        preds_test = np.repeat(train_series[-1], len(test_series))
        rmse = float(mean_squared_error(test_series, preds_test, squared=False))
        # final forecast: repeat last price
        final_forecast = np.repeat(train_series[-1], horizon)
        return final_forecast, rmse

    # Train on train and predict test horizon
    preds_test_path, pred_t_mean, p5, p95 = lstm_train_predict(train_series, forecast_days=len(test_series),
                                                              lookback=lookback, epochs=max(3, min(20, lstm_epochs)))
    # preds_test_path are predicted values for test horizon; compute rmse
    preds_test = np.array(preds_test_path[:len(test_series)])
    rmse = float(mean_squared_error(test_series, preds_test, squared=False))

    # Final forecast: retrain on full series and predict horizon
    full_series = np.concatenate([train_series, test_series])
    preds_full_path, final_mean, fp5, fp95 = lstm_train_predict(full_series, forecast_days=horizon,
                                                                lookback=lookback, epochs=max(3, min(30, lstm_epochs)))
    forecast = np.array(preds_full_path)[:horizon]
    return forecast, rmse


def _black_scholes_forecast(train, test, horizon, r=0.0427, n_paths=500):
    """
    Estimate sigma from history, then simulate risk-neutral GBM with drift r
    Return forecast (mean path) and RMSE comparing simulated mean to test closes (on their final day)
    We'll compute a per-day simulated mean and compare to test series.
    """
    series = train['Close'].values
    if len(series) < 2:
        series = np.append(series, series[-1] if len(series) else 1.0)
    logret = np.log(series[1:] / series[:-1])
    sigma = float(np.std(logret) * np.sqrt(252)) if len(logret) > 1 else 0.2
    S0 = float(series[-1])

    # simulate for test horizon to compute RMSE
    test_h = len(test)
    if test_h <= 0:
        test_h = 1
    sim_test = black_scholes_paths(S0, r, sigma, test_h, n_paths=n_paths)
    sim_test_mean = np.mean(sim_test, axis=0)
    # compare last test_h days to test['Close']
    rmse = float(mean_squared_error(test['Close'].values, sim_test_mean[:len(test['Close'])], squared=False))

    # final forecast: simulate horizon days and return mean path
    sim_fore = black_scholes_paths(float(train['Close'].iloc[-1]), r, sigma, horizon, n_paths=n_paths)
    forecast = np.mean(sim_fore, axis=0)
    return forecast, rmse


def _quantum_forecast(train, test, horizon):
    """
    Use quantum_predict_distribution to produce endpoint mean/p5/p95.
    Create geometric interpolation to produce day-by-day path to that endpoint.
    For RMSE, compare endpoint to test final price.
    """
    series = train['Close'].values
    if len(series) < 2:
        series = np.append(series, series[-1] if len(series) else 1.0)
    logret = np.log(series[1:] / series[:-1])
    mu = float(np.mean(logret) * 252) if len(logret) else 0.0
    sigma = float(np.std(logret) * np.sqrt(252)) if len(logret) > 1 else 0.2
    S0 = float(series[-1])

    q = quantum_predict_distribution(S0, mu, sigma, horizon)
    endpoint = q.get('mean', S0)
    # build geom interpolation
    n = horizon
    path = np.array([S0 * (endpoint / S0) ** ((i+1) / n) for i in range(n)])
    # RMSE: compare final endpoint to test final price (if exists)
    if len(test) > 0:
        rmse = float(mean_squared_error([test['Close'].values[-1]], [endpoint], squared=False))
    else:
        rmse = 0.0
    return path, rmse


def predict_stock(ticker,
                  start="2019-01-01",
                  end=None,
                  forecast_days=15,
                  holdout_days=None,
                  apply_sentiment=True,
                  bias_strength=0.15,
                  lstm_epochs=20,
                  n_paths_bs=500,
                  verbose=False):
    """
    Returns: (result_df, rmse_scores)
    - result_df: DataFrame with Date column + columns per model + Ensemble + Ensemble_p5 + Ensemble_p95
    - rmse_scores: dict mapping model name -> RMSE (computed on holdout/test)
    """

    end = end or datetime.today().strftime('%Y-%m-%d')
    df = fetch_prices(ticker, start=start, end=end)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")

    df = df.asfreq('B').ffill()
    last_date = df.index[-1]

    # holdout: default equal to forecast_days if sufficient length, else min( max(5, forecast_days), len//5)
    if holdout_days is None:
        holdout_days = min(forecast_days, max(5, int(len(df) * 0.1)))
    train, test = _holdout_split(df, holdout_days)

    # sentiment
    sentiment = 0.0
    if apply_sentiment:
        try:
            sentiment = float(get_sentiment_score(ticker))
        except Exception:
            sentiment = 0.0

    # Run models: produce final forecast arrays and rmse
    model_forecasts = {}
    rmse_scores = {}

    # LightGBM
    try:
        fore_lgb, rmse_lgb = _train_lightgbm_and_forecast(train, test, forecast_days)
        model_forecasts['LightGBM'] = np.array(fore_lgb)
        rmse_scores['LightGBM'] = rmse_lgb
    except Exception as e:
        if verbose: print("LGB error", e)
        model_forecasts['LightGBM'] = np.repeat(train['Close'].iloc[-1], forecast_days)
        rmse_scores['LightGBM'] = np.nan

    # XGBoost
    try:
        fore_xgb, rmse_xgb = _train_xgboost_and_forecast(train, test, forecast_days)
        model_forecasts['XGBoost'] = np.array(fore_xgb)
        rmse_scores['XGBoost'] = rmse_xgb
    except Exception as e:
        if verbose: print("XGB error", e)
        model_forecasts['XGBoost'] = np.repeat(train['Close'].iloc[-1], forecast_days)
        rmse_scores['XGBoost'] = np.nan

    # Prophet
    try:
        fore_prophet, rmse_prophet = _train_prophet_and_forecast(train, test, forecast_days)
        model_forecasts['Prophet'] = np.array(fore_prophet)
        rmse_scores['Prophet'] = rmse_prophet
    except Exception as e:
        if verbose: print("Prophet error", e)
        model_forecasts['Prophet'] = np.repeat(train['Close'].iloc[-1], forecast_days)
        rmse_scores['Prophet'] = np.nan

    # LSTM
    try:
        fore_lstm, rmse_lstm = _train_lstm_and_forecast(train, test, forecast_days, lstm_epochs=lstm_epochs)
        model_forecasts['LSTM'] = np.array(fore_lstm)
        rmse_scores['LSTM'] = rmse_lstm
    except Exception as e:
        if verbose: print("LSTM error", e)
        model_forecasts['LSTM'] = np.repeat(train['Close'].iloc[-1], forecast_days)
        rmse_scores['LSTM'] = np.nan

    # Black-Scholes (risk-neutral GBM)
    try:
        fore_bs, rmse_bs = _black_scholes_forecast(train, test, forecast_days, n_paths=n_paths_bs)
        model_forecasts['BlackScholes'] = np.array(fore_bs)
        rmse_scores['BlackScholes'] = rmse_bs
    except Exception as e:
        if verbose: print("BS error", e)
        model_forecasts['BlackScholes'] = np.repeat(train['Close'].iloc[-1], forecast_days)
        rmse_scores['BlackScholes'] = np.nan

    # Quantum (endpoint -> geometric interpolation)
    try:
        fore_quant, rmse_quant = _quantum_forecast(train, test, forecast_days)
        model_forecasts['Quantum'] = np.array(fore_quant)
        rmse_scores['Quantum'] = rmse_quant
    except Exception as e:
        if verbose: print("Quantum error", e)
        model_forecasts['Quantum'] = np.repeat(train['Close'].iloc[-1], forecast_days)
        rmse_scores['Quantum'] = np.nan

    # Clean duplicates before adding forecasts
    if result_df.index.duplicated().any():
        dup_count = result_df.index.duplicated().sum()
        print(f"[WARN] Dropping {dup_count} duplicate index rows from result_df")
        result_df = result_df.loc[~result_df.index.duplicated(keep='first')]
    result_df = result_df.sort_index()

    # Build result DataFrame
    dates = _dates_for_horizon(last_date, forecast_days)
    result_df = pd.DataFrame({'Date': dates})
    model_cols = []
    for name, arr in model_forecasts.items():
        col = name
        result_df[col] = pd.Series(
                arr[:forecast_days],
                index=result_df.index[:forecast_days]
            )
        model_cols.append(col)

    # Ensemble: mean across models
    ensemble = result_df[model_cols].mean(axis=1).values

    # Apply sentiment bias (simple directional multiplier)
    if apply_sentiment and abs(sentiment) > 1e-6 and bias_strength > 0:
        for col in model_cols:
            preds = result_df[col].values
            # determine sign of movement relative to S0
            S0 = float(df['Close'].iloc[-1])
            sign = np.sign(preds - S0)
            result_df[col] = preds * (1.0 + bias_strength * sentiment * sign)

        ensemble = result_df[model_cols].mean(axis=1).values

    # Ensemble uncertainty via bootstrap across models (simple)
    n_samples = 400
    rng = np.random.default_rng(12345)
    samples = []
    for i in range(forecast_days):
        per_model_vals = result_df[model_cols].iloc[i].values.astype(float)
        # approximate per-model std small fraction
        stds = np.maximum(np.abs(per_model_vals) * 0.02, 1e-6)
        sampled = np.array([rng.normal(loc=per_model_vals[j], scale=stds[j], size=n_samples) for j in range(len(per_model_vals))])
        # average across models -> one ensemble sample set
        avg_sample = np.mean(sampled, axis=0)
        samples.append(avg_sample)
    samples = np.array(samples)  # (horizon, n_samples)
    ensemble_mean = np.mean(samples, axis=1)
    ensemble_p5 = np.percentile(samples, 5, axis=1)
    ensemble_p95 = np.percentile(samples, 95, axis=1)

    result_df['Ensemble'] = ensemble_mean
    result_df['Ensemble_p5'] = ensemble_p5
    result_df['Ensemble_p95'] = ensemble_p95
    result_df['Sentiment'] = float(sentiment)

    return result_df, rmse_scores
