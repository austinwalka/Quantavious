# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from utils import trigger_colab_retrain
from predict_pipeline import load_stock_predictions

# ---------------------
# App Description
# ---------------------
st.set_page_config(page_title="Quantavious", layout="wide")
st.title("Quantavious by Austin Walker: Advanced Stock Forecasting & Crash Risk")

st.markdown("""
**Quantavious** is a kitchen-sink approach to predicting **short- to medium-term stock price movements** and **crash risk**:

**1) Math-based Models (20% weight)**  
- Deterministic scores based on GARCH volatility, EVT tail index, HMM regimes, VIX term slope, and breadth crashiness  
- Combined into a normalized risk score [0,1]  

**2) LSTM Sequence Model (50% weight)**  
- Trained on daily features (past 60 days of prices, vol, and indicators)  
- Predicts probability of next-day crash  
- Supports walk-forward training and cross-validation  

**3) FinBERT Sentiment Model (30% weight)**  
- Uses headlines to compute daily sentiment features (neg/pos mean, ratios, volume)  
- Maps sentiment to crash risk signal  

**Meta-Blender**  
- Weighted combination: 0.2*math + 0.5*LSTM + 0.3*FinBERT  
- Produces a **blended price forecast** and **daily crash risk probability**  

**Technical Indicators**  
- SMA20, Bollinger Bands, RSI, MACD, EMA, etc.  
- Shown alongside price predictions for context  

**Backtesting**  
- Optional walk-forward RMSE evaluation for historical accuracy  

**Workflow in Streamlit**  
- Loads **precomputed predictions and crash risk from Colab**  
- Warns if the stock is missing or data is >1 day old  
- Lets user open Colab notebook to retrain if needed  
- Displays 30-day forecast, daily crash probability, individual model contributions, and indicators
""")

# ---------------------
# Sidebar Inputs
# ---------------------
st.sidebar.header("Stock Forecast Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, BRK.B)", value="AAPL").upper()

forecast_days = 30  # fixed for day-by-day predictions

run_button = st.sidebar.button("Load Predictions")

colab_notebook_url = "https://colab.research.google.com/drive/1-liW2v2mK56QY5TfxrGlMq2v2JXN5iVw#scrollTo=fa2iby4eEXxU"  # Replace with your Colab notebook URL

# ---------------------
# Main Forecast Logic
# ---------------------
if run_button:
    with st.spinner(f"Loading predictions for {ticker}..."):
        df_pred, is_stale = load_stock_predictions(ticker)
        if df_pred is None:
            st.error(f"No prediction data found for {ticker}.")
            st.button("Retrain in Colab", on_click=trigger_colab_retrain, args=(colab_notebook_url,))
        else:
            if is_stale:
                st.warning(f"Prediction data for {ticker} is older than 1 day. Consider retraining.")
                st.button("Retrain in Colab", on_click=trigger_colab_retrain, args=(colab_notebook_url,))

            # ---------------------
            # Blended Forecast & Crash Risk
            # ---------------------
            st.subheader(f"{ticker} - 30-Day Blended Price Forecast & Crash Risk")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pred["Day"],
                y=df_pred["Blended_Price"],
                mode="lines+markers",
                name="Blended Price",
                line=dict(color="blue", width=2)
            ))
            fig.add_trace(go.Bar(
                x=df_pred["Day"],
                y=df_pred["Crash_Risk"]*100,
                name="Crash Risk (%)",
                marker_color="red",
                opacity=0.4,
                yaxis="y2"
            ))
            fig.update_layout(
                title=f"{ticker} - 30-Day Forecast & Crash Risk",
                xaxis_title="Day",
                yaxis_title="Price",
                yaxis2=dict(title="Crash Risk (%)", overlaying="y", side="right", range=[0,100]),
                legend=dict(orientation="h"),
                height=600
            )
            st.plotly_chart(fig)

            # ---------------------
            # Individual Model Contributions
            # ---------------------
            st.subheader("Individual Model Predictions")
            df_indiv = df_pred[["Day","Math","LSTM","FinBERT"]]
            st.dataframe(df_indiv)

            # ---------------------
            # Optional: Technical Indicators (last 90 days)
            # ---------------------
            st.subheader(f"{ticker} - Last 90 Days Technical Indicators")
            if "Indicators" in df_pred.columns:
                df_ind = pd.DataFrame(df_pred["Indicators"].tolist(), index=df_pred["Day"])
                st.line_chart(df_ind)
            else:
                st.info("Technical indicators not available in Colab output.")

            # ---------------------
            # Backtesting (optional)
            # ---------------------
            if "Backtest_RMSE" in df_pred.columns:
                st.subheader("Walk-forward Backtesting")
                st.metric(label="Historical RMSE", value=f"{df_pred['Backtest_RMSE'].iloc[0]:.2f}")

