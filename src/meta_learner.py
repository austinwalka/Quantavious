# src/meta_learner.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from .models import gbm_simulation, gauss_x, Schrodinger, LSTMModel, gbm_simulation
from .data_fetch import fetch_prices
from .features import add_technical_indicators
from tqdm import tqdm

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def create_meta_dataset_for_ticker(ticker, start='2015-01-01', end=None, lookback_min=90, trading_days=15, step=5, n_paths=200, max_samples=None):
    """
    Walk through historical dates for `ticker` and run constituent models using only
    data up to each timestamp. Returns a DataFrame with columns:
      [date, gbm_mean, gbm_p5, gbm_p95, bs_mean, bs_p5, bs_p95, quant_mean, quant_p5, quant_p95, lstm_mean, lstm_p5, lstm_p95, target]
    target is the close price at date + trading_days.
    Warning: expensive for long histories. Use step/max_samples to constrain.
    """
    df = fetch_prices(ticker, start, end)
    df = df.asfreq('B').ffill()
    idx_positions = list(range(lookback_min, len(df)-trading_days, step))
    if max_samples:
        idx_positions = idx_positions[:max_samples]
    rows = []
    for pos in tqdm(idx_positions, desc=f"Building meta dataset {ticker}"):
        hist = df.iloc[:pos+1].copy()
        try:
            S0 = float(hist['Close'].iloc[-1])
        except Exception:
            continue
        # run short-run models (reuse functions from models)
        # compute log returns
        logret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        mu = float(logret.mean() * 252) if len(logret)>0 else 0.0
        sigma = float(logret.std() * np.sqrt(252)) if len(logret)>1 else 0.01
        # GBM
        gbm_paths = gbm_simulation(S0, mu, sigma, trading_days, n_paths=n_paths)
        gbm_final = gbm_paths[:, -1]
        gbm_mean = float(np.mean(gbm_final)); gbm_p5, gbm_p95 = np.percentile(gbm_final, [5,95])
        # BS (risk-neutral)
        bs_paths = gbm_simulation(S0, 0.02, sigma, trading_days, n_paths=n_paths)
        bs_final = bs_paths[:, -1]
        bs_mean = float(np.mean(bs_final)); bs_p5, bs_p95 = np.percentile(bs_final, [5,95])
        # Quantum (coarse)
        from .models import quantum_predict_distribution
        q = quantum_predict_distribution(S0, mu, sigma, trading_days, N_x=512)
        quant_mean, quant_p5, quant_p95 = q['mean'], q['p5'], q['p95']
        # LSTM
        from .models import lstm_train_predict
        _, lstm_mean, lstm_p5, lstm_p95 = lstm_train_predict(hist['Close'].values, forecast_days=trading_days, epochs=5, lookback=60)
        # target
        target = float(df['Close'].iloc[pos + trading_days])
        row = {
            'date': df.index[pos],
            'gbm_mean': gbm_mean, 'gbm_p5': gbm_p5, 'gbm_p95': gbm_p95,
            'bs_mean': bs_mean, 'bs_p5': bs_p5, 'bs_p95': bs_p95,
            'quant_mean': quant_mean, 'quant_p5': quant_p5, 'quant_p95': quant_p95,
            'lstm_mean': lstm_mean, 'lstm_p5': lstm_p5, 'lstm_p95': lstm_p95,
            'target': target
        }
        rows.append(row)
    meta_df = pd.DataFrame(rows).set_index('date')
    return meta_df

def train_meta_model(meta_df, model_name="meta_lgb", params=None):
    """
    Train a LightGBM regressor on meta_df (features -> target).
    """
    params = params or {'objective':'regression','metric':'rmse','verbosity':-1}
    X = meta_df.drop(columns=['target']).fillna(0.0)
    y = meta_df['target'].values
    tscv = TimeSeriesSplit(n_splits=5)
    val_scores = []
    fold_preds = np.zeros(len(y))
    best_iter = 100
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        gbm = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=False)
        fold_preds[val_idx] = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        val_scores.append(mean_squared_error(y_val, fold_preds[val_idx], squared=False))
        best_iter = int(gbm.best_iteration or best_iter)
    print("CV RMSEs:", val_scores)
    # final model
    final = lgb.LGBMRegressor(n_estimators=best_iter or 100)
    final.fit(X, y)
    # save
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(final, path)
    return final

def build_and_train_full_meta_for_ticker(ticker, start='2015-01-01', end=None, trading_days=15, step=5, n_paths=200, max_samples=None):
    meta_df = create_meta_dataset_for_ticker(ticker, start=start, end=end, trading_days=trading_days, step=step, n_paths=n_paths, max_samples=max_samples)
    model = train_meta_model(meta_df, model_name=f"meta_{ticker}")
    return model, meta_df
