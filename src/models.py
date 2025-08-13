# src/models.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Existing model stubs (replace with your working ones) ---
def train_gbm(X_train, y_train):
    # your actual GBM training here
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor()
    m.fit(X_train, y_train)
    return m

def train_lstm(X_train, y_train, X_test):
    # your actual LSTM inference here; return np.array of preds for X_test
    # placeholder: simple GBM as a stand-in to keep example consistent
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=200, random_state=42)
    m.fit(X_train, y_train)
    return m.predict(X_test)

def train_quantum(X_train, y_train, X_test):
    # placeholder deterministic prediction
    base = np.mean(y_train)
    return np.repeat(base, len(X_test))

# --- Meta-model (stacking) ---
def fit_meta_model(pred_frame: pd.DataFrame, y_true: np.ndarray):
    """
    pred_frame: columns are model predictions aligned to y_true index/order
    y_true: actuals
    Returns a sklearn pipeline meta-model that blends base models into one.
    """
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=0.5, random_state=42))
    ])
    model.fit(pred_frame.values, y_true)
    return model

def predict_meta(meta_model, pred_frame: pd.DataFrame) -> np.ndarray:
    return meta_model.predict(pred_frame.values)
