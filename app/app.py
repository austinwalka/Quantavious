import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os
from prediction_pipeline import load_forecast, download_last_90_days
from utils import should_retrain, trigger_colab_training

st.set_page_config(page_title="Quantavious", layout="wide")
st.title("Quantavious: Advanced Stock & Crash Risk Dashboard")

# ---------------------
# Sidebar
# ---------------------
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker (S&P 500)", value="AAPL").upper()

forecast_days = 60

# Path to forecast CSV
from pathlib import Path
data_dir = Path(__file__).parent / "data"
forecast_file = data_dir / f"{ticker}_forecast.csv"

# ---------------------
# Check if retraining needed
# ---------------------
if should_retrain(forecast_file):
    st.warning(f"Forecast for {ticker} is missing or older than 1 day. Please run Colab training.")
    if st.button("Trigger Colab Training"):
        trigger_colab_training(ticker)

# ---------------------
# Load Forecast
# ---------------------
forecast_df = load_forecast(ticker)
if forecast_df is not None:
    st.subheader(f"60-Day Forecast & Crash Risk for {ticker}")

    # Line chart for Meta-Blender & Crash Risk
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["Day"],
        y=forecast_df["Meta_Blender"],
        mode="lines+markers",
        name="Meta-Blender Forecast"
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["Day"],
        y=forecast_df["Crash_Risk_Percent"],
        mode="lines+markers",
        name="Crash Risk (%)",
        yaxis="y2"
    ))

    # Layout
    fig.update_layout(
        title=f"{ticker} Forecast & Crash Risk",
        xaxis_title="Day",
        yaxis_title="Price Forecast",
        yaxis2=dict(title="Crash Risk (%)", overlaying="y", side="right"),
        height=600
    )

    st.plotly_chart(fig)

    # Display individual model predictions
    st.subheader("Individual Model Forecasts")
    st.dataframe(forecast_df[["Day","Math_Pred","LSTM_Pred","FinBERT_Pred","Meta_Blender","Crash_Risk_Percent"]])

# ---------------------
# Technical Indicators
# ---------------------
st.subheader(f"{ticker} - Last 90 Days Technical Indicators")
df_tech = download_last_90_days(ticker)
if df_tech is not None:
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(
        x=df_tech.index,
        open=df_tech['Open'],
        high=df_tech['High'],
        low=df_tech['Low'],
        close=df_tech['Close'],
        name='Candlestick'
    ))
    for col, color in [("SMA20","orange"), ("BB_upper","green"), ("BB_lower","red")]:
        if col in df_tech.columns:
            fig2.add_trace(go.Scatter(
                x=df_tech.index,
                y=df_tech[col],
                mode="lines",
                name=col,
                line=dict(color=color)
            ))
    fig2.update_layout(title=f"{ticker} Technical Chart (90 Days)", height=600)
    st.plotly_chart(fig2)
else:
    st.warning("Could not download recent technical data.")
