import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.predict_pipeline import predict_stock

import os

print("DEBUG: Render PORT =", os.getenv("PORT"))
print("DEBUG: NEWS_API_KEY =", "SET" if os.getenv("NEWS_API_KEY") else "MISSING")


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    layout="wide",
    page_title="Quantavius by Walker Wealth",
    page_icon="📊"
)

if "tickers" not in st.session_state:
    st.session_state.tickers = []

# ---------------------------
# Title & Description
# ---------------------------
st.title("📊 Quantavius by Walker Wealth")

st.markdown("""
**Quantavius** is an AI-powered market forecasting tool that combines **multiple predictive models** into one ensemble.  
Models included:
- **LightGBM** & **XGBoost** – tree-based gradient boosting  
- **Prophet** – trend + seasonality decomposition  
- **LSTM** – deep learning recurrent network for time-series  
- **Black–Scholes** – options theory-based price estimation  
- **FinBERT Sentiment** – adjusts forecasts based on news tone  
- *(Optional)* Quantum optimizer for model weight tuning  

You can:
1. Create a list of symbols  
2. Choose your forecast horizon  
3. Run all models  
4. Compare results in tables & charts  
5. Download forecasts for offline analysis
---
""")

# ---------------------------
# Sidebar - Symbol Controls
# ---------------------------
st.sidebar.header("📌 Symbol List")

ticker_input = st.sidebar.text_input("Add Symbol", "")
if st.sidebar.button("Add Symbol"):
    if ticker_input and ticker_input.upper() not in st.session_state.tickers:
        st.session_state.tickers.append(ticker_input.upper())

if st.session_state.tickers:
    st.sidebar.write("### Current Symbols")
    for t in st.session_state.tickers:
        if st.sidebar.button(f"❌ Remove {t}"):
            st.session_state.tickers.remove(t)

forecast_days = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 14)
run_button = st.sidebar.button("🚀 Run Forecasts")

# ---------------------------
# Main Processing
# ---------------------------
if run_button and st.session_state.tickers:
    all_results = []

    for ticker in st.session_state.tickers:
        st.subheader(f"📈 Forecast Results: {ticker}")

        # Predict
        result_df, rmse_scores = predict_stock(
            ticker,
            start="2023-01-01",
            end="2024-12-31",
            forecast_days=forecast_days
        )

        # RMSE Table
        rmse_df = pd.DataFrame(list(rmse_scores.items()), columns=["Model", "RMSE"]).sort_values("RMSE")
        st.markdown("**Model Performance (Lower RMSE is better)**")
        st.dataframe(rmse_df, use_container_width=True)

        # Forecast Table
        st.markdown("**Forecast Table**")
        st.dataframe(result_df, use_container_width=True)

        # Plot Forecasts
        fig = go.Figure()
        for col in result_df.columns:
            if col != "Date":
                fig.add_trace(go.Scatter(
                    x=result_df["Date"],
                    y=result_df[col],
                    mode='lines',
                    name=col
                ))
        fig.update_layout(
            title=f"{ticker} Model Forecasts",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Model",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Store for combined table
        all_results.append(result_df.assign(Ticker=ticker))

    # Combined Results
    merged_df = pd.concat(all_results, ignore_index=True)
    st.subheader("📊 Combined Results for All Symbols")
    st.dataframe(merged_df, use_container_width=True)

    # Download Button
    csv = merged_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download All Forecasts as CSV",
        csv,
        "thorp_forecasts.csv",
        "text/csv"
    )

elif run_button:
    st.warning("Please add at least one symbol before running forecasts.")
