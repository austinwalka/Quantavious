# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

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
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
run_button = st.sidebar.button("Load Forecast")

# -----------------------
# Helper functions
# -----------------------
DATA_DIR = Path("./data")  # Change if needed

def load_colab_data(ticker):
    stock_dir = DATA_DIR / ticker
    if not stock_dir.exists():
        st.error(f"No precomputed data found for {ticker}. Please upload to {stock_dir}.")
        return None
    try:
        forecast = pd.read_csv(stock_dir / "forecast_30d.csv")
        crash = pd.read_csv(stock_dir / "crash_30d.csv")
        indicators = pd.read_csv(stock_dir / "indicators.csv")
        with open(stock_dir / "meta.json") as f:
            meta = json.load(f)
        with open(stock_dir / "backtest.json") as f:
            backtest = json.load(f)
        return {"forecast": forecast, "crash": crash, "indicators": indicators, "meta": meta, "backtest": backtest}
    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
        return None

# -----------------------
# Main logic
# -----------------------
if run_button:
    with st.spinner(f"Loading precomputed data for {ticker}..."):
        data = load_colab_data(ticker)
        if data:
            # --------- Unified Price Forecast Chart ---------
            df_forecast = data["forecast"]
            df_ind = data["indicators"]
            df_crash = data["crash"]
            df_backtest = data["backtest"]

            fig = go.Figure()

            # Candlestick
            if {"Open", "High", "Low", "Close"}.issubset(df_ind.columns):
                fig.add_trace(go.Candlestick(
                    x=df_ind["date"],
                    open=df_ind["Open"],
                    high=df_ind["High"],
                    low=df_ind["Low"],
                    close=df_ind["Close"],
                    name="Price"
                ))

            # Bollinger Bands (shaded)
            if {"BB_upper", "BB_lower"}.issubset(df_ind.columns):
                fig.add_trace(go.Scatter(
                    x=df_ind["date"],
                    y=df_ind["BB_upper"],
                    line=dict(color="lightgray"),
                    name="BB Upper",
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=df_ind["Date"],
                    y=df_ind["BB_lower"],
                    line=dict(color="lightgray"),
                    fill="tonexty",
                    fillcolor="rgba(200,200,200,0.2)",
                    name="Bollinger Band",
                    showlegend=True
                ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=df_forecast["blended"],
                mode="lines+markers",
                line=dict(color="red", dash="dash"),
                name="Meta-Blended Forecast"
            ))

             # Forecast line
            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=df_forecast["gbm"],
                mode="lines+markers",
                line=dict(color="blue", dash="dash"),
                name="GBM Forecast"
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=df_forecast["ou"],
                mode="lines+markers",
                line=dict(color="green", dash="dash"),
                name="OU Forecast"
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=df_forecast["lstm"],
                mode="lines+markers",
                line=dict(color="orange", dash="dash"),
                name="LSTM Forecast"
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=df_forecast["finbert_adjust_ref"],
                mode="lines+markers",
                line=dict(color="lightblue", dash="dash"),
                name="FinBERT Adjusted Reference"
            ))

            # Shaded forecast region
            fig.add_vrect(
                x0=df_forecast["date"].iloc[0],
                x1=df_forecast["date"].iloc[-1],
                fillcolor="yellow",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="Forecast Region"
            )

            # Crash probability as secondary y-axis
            fig.add_trace(go.Scatter(
                x=df_crash["date"],
                y=df_crash["p_crash"]*100,
                mode="lines+markers",
                line=dict(color="purple"),
                name="Crash Probability (%)",
                yaxis="y2"
            ))

            # Layout
            fig.update_layout(
                title=f"{ticker} - 30-Day Forecast & Crash Risk",
                yaxis=dict(title="Price"),
                yaxis2=dict(title="Crash Probability (%)", overlaying="y", side="right", range=[0,100]),
                xaxis=dict(title="Date"),
                legend=dict(orientation="h"),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # --------- Individual Model Contributions ---------
            #st.subheader("Individual Model Contributions")
            #st.dataframe(pd.DataFrame(data["meta"]))

            # --------- Technical Indicators ---------
            st.subheader("Recent Technical Indicators")
            st.dataframe(df_ind.tail(30))

            # --------- Backtest ---------
            st.subheader("Backtest Metrics")
            fdf_math_rmse = pd.DataFrame(list(df_backtest["math_rmse"].items()), columns=["Days", "RMSE"])
            fdf_math_rmse["Days"] = fdf_math_rmse["Days"].astype(int)

            # Create a summary table with LSTM RMSE
            df_summary = pd.DataFrame({
                "Model": ["Math Model", "LSTM 1-Day"],
                "RMSE": [fdf_math_rmse["RMSE"].tolist(), df_backtest["lstm_rmse_1d"]],
                "Description": [df_backtest["note"], "1-step walk-forward LSTM"]
            })

            # Display both
            st.subheader("Math Model RMSE by Forecast Horizon")
            st.table(fdf_math_rmse)

            st.subheader("Summary of Backtest RMSE")
            st.table(df_summary
