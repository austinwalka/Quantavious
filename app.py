import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from src.predict_pipeline import predict_stock, backtest_stock, compute_indicators

# ---------------------
# App Description
# ---------------------
st.set_page_config(page_title="Quantavious", layout="wide")
st.title("Quantavious by Austin Walker: Advanced Stock Forecasting & Risk Analysis")

st.markdown("""
This app predicts short- and medium-term stock price movements using a **kitchen-sink approach**:

**Math-based Models**
- Geometric Brownian Motion (GBM) – stochastic trend
- Langevin Equation – mean-reverting dynamics
- Boltzmann Equation – volatility-driven stochastic paths
- Schrödinger Equation proxy – captures uncertainty as spread evolution

**Machine Learning Models**
- LSTM – deep recurrent network for sequential price modeling
- Gradient Boosting / Random Forest – ensemble-based predictions

**Technical Indicators & Sentiment**
- RSI, MACD, SMA20, Bollinger Bands
- FinBERT sentiment weighting based on recent headlines

**Meta-blender**
- Combines all model outputs into a single “best guess” forecast
- Weighted by sentiment to tilt toward trend or mean-reversion

**Backtesting**
- Walk-forward RMSE evaluation for historical accuracy
- Enables performance validation

**Usage**
- Enter a stock ticker and select your forecast window
- Hit **Predict** to run models and see outputs
""")

# ---------------------
# Sidebar Inputs
# ---------------------
st.sidebar.header("Stock Forecast Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[1, 3, 5, 10, 21, 63, 126],  # 1d, 3d, 5d, 2w, 1m, 3m, 6m approx
    format_func=lambda x: f"{x} trading days (~{int(x/21*1.0)} months)" if x>=21 else f"{x} days"
)

run_button = st.sidebar.button("Predict")

# ---------------------
# Main Forecast Logic
# ---------------------
if run_button:
    with st.spinner(f"Running models for {ticker}..."):
        result = predict_stock(ticker, days=forecast_days)
        if result is None:
            st.error(f"Could not generate forecast for {ticker}.")
        else:
            blended = result["Predictions"]
            individual = result["Individual"]

            # Display blended prediction
            st.subheader(f"Meta-blended Forecast for {ticker} ({forecast_days} days)")
            df_pred = pd.DataFrame({
                "Day": np.arange(1, forecast_days+1),
                "Price Forecast": blended
            })
            st.line_chart(df_pred.set_index("Day"))

            # Display individual model contributions
            st.subheader("Individual Model Forecasts")
            df_models = pd.DataFrame(individual)
            st.dataframe(df_models)

            # Optional backtest
            st.subheader("Walk-forward Backtest (RMSE)")
            rmse = backtest_stock(ticker, forecast_days=forecast_days)
            st.metric(label="Historical RMSE", value=f"{rmse:.2f}")

            # Show raw values for copy/paste if desired
            st.subheader("Raw Prediction Values")
            st.dataframe(df_pred)

# ---------------------
# 90-Day Technical Indicator Chart
# ---------------------

