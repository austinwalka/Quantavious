# utils.py
import os
import pandas as pd
import numpy as np

from google.colab import drive

# ----------------------------
# Drive Mount / Colab Access
# ----------------------------
def mount_drive():
    """
    Mount Google Drive to access pre-trained predictions.
    Call this once at the top of your Colab.
    """
    try:
        drive.mount('/content/drive')
    except Exception as e:
        print("Drive may already be mounted or access denied:", e)

# ----------------------------
# Load Colab Predictions
# ----------------------------
def load_colab_predictions(ticker, folder_path="https://drive.google.com/drive/folders/14xT-hgxHFUwZDItzUm7nSFyZuVvW6Qqb?usp=sharing"):
    """
    Loads pre-trained predictions for a ticker.
    Returns dict with:
        - predictions: blended forecast (list)
        - individual: dict of individual models
        - crash_risk: list of 30-day crash probabilities
    """
    file_path = os.path.join(folder_path, f"{ticker}.pkl")
    if not os.path.exists(file_path):
        print(f"No pre-trained predictions found for {ticker}")
        return None

    try:
        df = pd.read_pickle(file_path)
        return {
            "predictions": df["blended"].values.tolist(),
            "individual": {k: v.tolist() for k, v in df["individual"].items()},
            "crash_risk": df["crash_risk"].values.tolist()
        }
    except Exception as e:
        print(f"Error loading predictions for {ticker}: {e}")
        return None

# ----------------------------
# Backtesting / Metrics
# ----------------------------
def walk_forward_rmse(actual_prices, predicted_prices):
    """
    Simple RMSE walk-forward backtest
    """
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)
    return np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
