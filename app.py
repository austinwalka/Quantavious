import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.predict_pipeline import predict_stock

# ----------------------------
# APP HEADER
# ----------------------------
st.title("📈 Quantavious Stock Prediction Tool")

st.markdown("""
Welcome to **Quantavious** 🚀  

This tool combines **mathematical models, machine learning, and risk analysis** to produce forward-looking 
stock predictions. Instead of relying on a single forecast, it blends multiple approaches:

---

### 🔢 Models & Methods Used:
- **Geometric Brownian Motion (GBM):**  
  Simulates stock paths under stochastic volatility. Helps capture random-walk dynamics.  
- **Ornstein–Uhlenbeck Process (OU):**  
  Models mean reversion, useful for assets that revert to a long-term equilibrium.  
- **Schrödinger-Inspired Probability Distribution:**  
  Captures uncertainty in multiple potential outcomes, not just one path.  
- **ARIMA / Statistical Forecasting:**  
  Time-series decomposition into trend, seasonality, and residual noise.  
- **Machine Learning (Neural Nets):**  
  Pattern recognition on nonlinear features (momentum, volatility clustering).  
- **Risk Layer (VaR / CVaR):**  
  Measures downside exposure and tail-risk under simulated paths.  

---

### ⚡ What this brings:
- **Short-term signals (intraday–1w):** dominated by momentum, noise filtering, OU mean-reversion.  
- **Medium-term (1m–3m):** blends GBM scenarios + ML classification of directional bias.  
- **Long-term (>6m):** scenario distributions (GBM & Schrödinger) + risk-adjusted expected value.  

""")

# ----------------------------
# USER INPUT
# ----------------------------
st.sidebar.header("Custom Stock Analysis")

custom_ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA)", "")

time_horizon = st.sidebar.selectbox(
    "Select prediction horizon",
    ["1d", "5d (1w)", "1mo", "3mo", "6mo"],
    index=1
)

# ----------------------------
# CUSTOM TICKER PREDICTION
# ----------------------------
if custom_ticker:
    try:
        result = predict_stock(custom_ticker.upper(), retrain_if_missing=False, horizon=time_horizon)
        if result:
            st.subheader(f"Prediction for {custom_ticker.upper()} ({time_horizon} horizon)")
            st.write(pd.DataFrame([result]))

            # Show recent price history
            hist = yf.download(custom_ticker.upper(), period="6mo", interval="1d")
            st.line_chart(hist["Close"])
            
            st.markdown(f"""
            **Interpretation:**  
            - `Last Close` → observed market price.  
            - `Predicted Price` → blended forecast across all models.  
            - `GBM / OU / Schrödinger` → scenario-specific outputs.  
            - `Risk` → Value-at-Risk adjustments for downside tail.  
            """)
    except Exception as e:
        st.error(f"Error processing {custom_ticker}: {e}")

else:
    st.info("Enter a ticker symbol in the sidebar to generate predictions.")
