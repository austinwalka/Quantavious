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

            fig = go.Figure()

            # Ensure 'Date' exists; if not, use index
            if "Date" in df_ind.columns:
                x_values = df_ind["Date"]
            else:
                x_values = df_ind.index

            # Candlestick
            if {"Open", "High", "Low", "Close"}.issubset(df_ind.columns):
                fig.add_trace(go.Candlestick(
                    x=x_values,
                    open=df_ind["Open"],
                    high=df_ind["High"],
                    low=df_ind["Low"],
                    close=df_ind["Close"],
                    name="Price"
                ))

            # Bollinger Bands (shaded)
            if {"BB_upper", "BB_lower"}.issubset(df_ind.columns):
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=df_ind["BB_upper"],
                    line=dict(color="lightgray"),
                    name="BB Upper",
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=df_ind["BB_lower"],
                    line=dict(color="lightgray"),
                    fill="tonexty",
                    fillcolor="rgba(200,200,200,0.2)",
                    name="Bollinger Band",
                    showlegend=True
                ))

            # Ensure 'Date' exists; if not, use index
            if "Date" in df_forecast.columns:
                f_values = df_forecast["Date"]
            else:
                f_values = df_forecast.index

            # Forecast line
            fig.add_trace(go.Scatter(
                x=f_values,
                y=df_forecast["Price_Forecast"],
                mode="lines+markers",
                line=dict(color="red", dash="dash"),
                name="Meta-Blended Forecast"
            ))

            # Shaded forecast region
            fig.add_vrect(
                x0=f_values.iloc[0],
                x1=f_values.iloc[-1],
                fillcolor="yellow",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="Forecast Region"
            )

            # Ensure 'Date' exists; if not, use index
            if "Date" in df_crash.columns:
                c_values = df_crash["Date"]
            else:
                c_values = df_crash.index


            # Crash probability as secondary y-axis
            fig.add_trace(go.Scatter(
                x=c_values,
                y=df_crash["Crash_Prob"]*100,
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
            st.subheader("Individual Model Contributions")
            st.dataframe(pd.DataFrame(data["meta"]))

            # --------- Technical Indicators ---------
            st.subheader("Recent Technical Indicators")
            st.dataframe(df_ind.tail(30))

            # --------- Backtest ---------
            st.subheader("Backtest Metrics")
            for k,v in data["backtest"].items():
                st.metric(label=k, value=v)
