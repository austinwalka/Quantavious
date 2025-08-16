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
# 30-Day Technical Indicator Chart
# ---------------------

st.subheader(f"{ticker} - Last 30 Days Technical Indicators & Candlesticks")

try:
    # Download last 90 days for indicator calculation
    df_90 = yf.download(ticker, period="90d", interval="1d")
    if not df_90.empty:
        # Compute indicators
        df_90 = compute_indicators(df_90)
        
        # Ensure MACD_signal exists
        if 'MACD_signal' not in df_90.columns:
            macd = MACD(df_90['Close'])
            df_90['MACD'] = macd.macd()
            df_90['MACD_signal'] = macd.macd_signal()
        
        # Take last 30 rows
        df_plot = df_90.tail(30).copy()
        
        # Drop rows with any missing values in the columns we plot
        required_cols = ['Open','High','Low','Close','SMA20','BB_upper','BB_lower','RSI','MACD','MACD_signal']
        df_plot = df_plot.dropna(subset=[c for c in required_cols if c in df_plot.columns])
        
        fig = go.Figure()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='Candlestick'
        ))

        # SMA20
        if 'SMA20' in df_plot:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['SMA20'],
                mode='lines',
                name='SMA20',
                line=dict(color='orange')
            ))

        # Bollinger Bands
        if 'BB_upper' in df_plot:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['BB_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='green', dash='dash')
            ))
        if 'BB_lower' in df_plot:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['BB_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='red', dash='dash')
            ))

        # RSI on secondary y-axis
        if 'RSI' in df_plot:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['RSI'],
                mode='lines',
                name='RSI',
                yaxis='y2',
                line=dict(color='purple')
            ))

        # MACD + Signal line on third y-axis
        if 'MACD' in df_plot and 'MACD_signal' in df_plot:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['MACD'],
                mode='lines',
                name='MACD',
                yaxis='y3',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['MACD_signal'],
                mode='lines',
                name='MACD Signal',
                yaxis='y3',
                line=dict(color='red', dash='dash')
            ))

        # Layout with multiple y-axes
        fig.update_layout(
            title=f"{ticker} - Last 30 Days Technical Chart",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0,100]),
            yaxis3=dict(title='MACD', anchor='free', overlaying='y', side='right', position=0.95),
            legend=dict(orientation='h'),
            height=600
        )

        st.plotly_chart(fig)
    else:
        st.warning(f"No recent data for {ticker}")

except Exception as e:
    st.error(f"Failed to generate chart: {e}")