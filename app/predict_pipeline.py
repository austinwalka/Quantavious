# predict_pipeline.py
import pandas as pd
import os
from datetime import datetime, timedelta

PREDICTIONS_DIR = "https://drive.google.com/drive/folders/14xT-hgxHFUwZDItzUm7nSFyZuVvW6Qqb?usp=drive_link"  # folder where Colab saves CSV/Parquet per stock

def load_stock_predictions(ticker: str):
    """
    Loads precomputed predictions for a ticker.
    Returns:
        df_pred: DataFrame with Day, Blended_Price, Crash_Risk, Math, LSTM, FinBERT
        is_stale: bool, True if data is >1 day old
    """
    # normalize ticker for filename
    file_safe_ticker = ticker.replace(".", "_").upper()
    file_path_csv = os.path.join(PREDICTIONS_DIR, f"{file_safe_ticker}.csv")
    file_path_parquet = os.path.join(PREDICTIONS_DIR, f"{file_safe_ticker}.parquet")

    df_pred = None
    if os.path.exists(file_path_parquet):
        df_pred = pd.read_parquet(file_path_parquet)
    elif os.path.exists(file_path_csv):
        df_pred = pd.read_csv(file_path_csv)
    else:
        return None, False  # no data

    # check freshness
    if "Timestamp" in df_pred.columns:
        last_updated = pd.to_datetime(df_pred["Timestamp"].max())
        is_stale = last_updated < datetime.now() - timedelta(days=1)
    else:
        is_stale = True  # no timestamp means stale

    return df_pred, is_stale
