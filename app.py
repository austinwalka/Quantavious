import streamlit as st
import pandas as pd
from src.predict_pipeline import predict_stock

st.set_page_config(page_title="Quantavius Stock Predictor", layout="wide")

st.title("📈 Quantavius Stock Predictor")
st.write("""
This app predicts stock prices using:
- **Math-based models**: Geometric Brownian Motion, Langevin, Boltzmann, Schrödinger proxy
- **Machine learning**: LightGBM, XGBoost, LSTM
- **Prophet** for trend/seasonality
- **Meta-blender** to combine results
""")

tickers_input = st.text_input(
    "Enter stock ticker(s) (comma-separated)", 
    value="AAPL"
)
run_btn = st.button("Run Predictions")

if run_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results_all = []

    for ticker in tickers:
        st.subheader(f"Ticker: {ticker}")
        try:
            # TODO: replace with your real features for this ticker
            fake_features = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]])
            df_pred = predict_stock(ticker, fake_features)

            results_all.append(df_pred)

            if len(df_pred) > 1:
                st.line_chart(df_pred.set_index("date")["prediction"])
            st.dataframe(df_pred)

        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")

    if results_all:
        combined = pd.concat(results_all, ignore_index=True)
        csv = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all results as CSV",
            csv,
            "predictions.csv",
            "text/csv",
            key="download-csv"
        )
