import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Example model loading — adjust to match your paths
MODELS_DIR = Path(__file__).resolve().parent / "models"

def load_model(ticker):
    model_path = MODELS_DIR / f"{ticker}_model.pkl"
    return joblib.load(model_path)

def predict_stock(ticker, features_df):
    """
    Run prediction for given ticker and input features DataFrame.
    Handles both single-day and multi-day predictions without shape errors.
    """
    try:
        model = load_model(ticker)
    except FileNotFoundError:
        raise RuntimeError(f"No model found for {ticker}")

    # Ensure 2D array input for model
    features = features_df.values
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Run model prediction
    preds = model.predict(features)

    # Flatten if model returns 2D
    preds = np.array(preds).flatten()

    # If only 1 prediction, return single-row DataFrame
    if len(preds) == 1:
        return pd.DataFrame({
            "date": [pd.Timestamp.today().normalize()],
            "ticker": [ticker],
            "prediction": preds
        })

    # If multiple days, generate forecast dates
    forecast_dates = pd.date_range(
        start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        periods=len(preds)
    )

    return pd.DataFrame({
        "date": forecast_dates,
        "ticker": [ticker] * len(preds),
        "prediction": preds
    })

# Example single-ticker run
if __name__ == "__main__":
    # Fake features example
    fake_features = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]])  # Replace with real features
    result = predict_stock("AAPL", fake_features)
    print(result)
