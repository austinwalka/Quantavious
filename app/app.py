# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from utils import load_colab_predictions
from ta.momentum import RSIIndicator
from ta.trend import MACD

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
- FinBERT sentiment (if available) – maps news sentiment to risk signal

**Technical Indicators**
- RSI, MACD, SMA20, Bollinger Bands

**Meta-blender**
- Combines all model outputs into a single “best guess” forecast
- Weighted by sentiment to tilt toward trend or mean-reversion

**Crash Risk**
- EVT/GARCH-style risk predictions for next 30 days

**Backtesting**
- Walk-forward RMSE evaluation for historical accuracy

**Usage**
- Enter a stock ticker
- Hit **Predict** to run models and see outputs
- Forecast horizon is **fixed at 30 days**
""")

# ---------------------
# Sidebar Inputs
# ---------------------
st.sidebar.header("Stock Forecast Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()
run_button = st.sidebar.button("Predict")

FORECAST_DAYS = 30  # Fixed horizon

# ---------------------
# Main Forecast Logic
# ---------------------
if run_button:
    with st.spinner(f"Loading predictions for {ticker}..."):
        result = load_colab_predictions(ticker)

        if result is None:
            st.error(f"Could not load forecast for {ticker}. Make sure the file exists or Google Drive link is correct.")
        else:
            blended = result["predictions"][:FORECAST_DAYS]
            individual = result["individual"]
            crash_risk = result["crash_risk"][:FORECAST_DAYS]

            # Display blended prediction
            st.subheader(f"Meta-blended Forecast for {ticker} ({FORECAST_DAYS} days)")
            df_pred = pd.DataFrame({
                "Day": np.arange(1, FORECAST_DAYS+1),
                "Price Forecast": blended
            })
            st.line_chart(df_pred.set_index("Day"))

            # Display individual model contributions
            st.subheader("Individual Model Forecasts")
            df_models = pd.DataFrame(individual)
            st.dataframe(df_models)

            # Display crash risk
            st.subheader(f"Crash Risk Probability (%) Next {FORECAST_DAYS} Days")
            df_crash = pd.DataFrame({
                "Day": np.arange(1, FORECAST_DAYS+1),
                "Crash Risk %": [r*100 for r in crash_risk]
            })
            st.line_chart(df_crash.set_index("Day"))

# ---------------------
# 90-Day Technical Indicator Chart
# ---------------------
st.subheader(f"{ticker} - Last 90 Days Technical Indicators & Candlesticks")

try:
    df_90 = yf.download(ticker, period="90d", interval="1d")
    if not df_90.empty:
        df_90["SMA20"] = df_90["Close"].rolling(20).mean()
        df_90["BB_upper"] = df_90["SMA20"] + 2*df_90["Close"].rolling(20).std()
        df_90["BB_lower"] = df_90["SMA20"] - 2*df_90["Close"].rolling(20).std()
        rsi = RSIIndicator(df_90["Close"])
        df_90["RSI"] = rsi.rsi()
        macd = MACD(df_90["Close"])
        df_90["MACD"] = macd.macd()
        df_90["MACD_signal"] = macd.macd_signal()

        df_plot = df_90.tail(90).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='Candlestick'
        ))

        indicator_lines = {
            'SMA20': {'color':'orange'},
            'BB_upper': {'color':'green', 'dash':'dash'},
            'BB_lower': {'color':'red', 'dash':'dash'},
            'RSI': {'color':'purple', 'yaxis':'y2'},
            'MACD': {'color':'blue', 'yaxis':'y3'},
            'MACD_signal': {'color':'red', 'dash':'dash', 'yaxis':'y3'}
        }

        for ind, opts in indicator_lines.items():
            if ind in df_plot.columns:
                line_props = dict(color=opts.get('color','black'))
                if 'dash' in opts:
                    line_props['dash'] = opts['dash']
                fig.add_trace(go.Scatter(
                    x=df_plot.index,
                    y=df_plot[ind],
                    mode='lines',
                    name=ind,
                    line=line_props,
                    yaxis=opts.get('yaxis','y')
                ))

        fig.update_layout(
            title=f"{ticker} - Last 90 Days Technical Chart",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0,100]) if 'RSI' in df_plot.columns else None,
            yaxis3=dict(title='MACD', overlaying='y', side='right', position=0.95) if 'MACD' in df_plot.columns else None,
            legend=dict(orientation='h'),
            height=600
        )

        st.plotly_chart(fig)
    else:
        st.warning(f"No recent data for {ticker}")
except Exception as e:
    st.error(f"Failed to generate chart: {e}")
