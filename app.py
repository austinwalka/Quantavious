import streamlit as st
import pandas as pd
from src.predict_pipeline import predict_stock

st.set_page_config(page_title="Quantavius", layout="wide")

st.title("Quantavius: Multi-Model Stock Forecasts")
st.markdown("""
This app forecasts short-term stock prices using:

✅ Math-based models (GBM, OU/Langevin, Boltzmann, Schrödinger proxy)  
✅ Prophet (trend + seasonality)  
✅ Machine Learning models (LSTM, LightGBM, XGBoost placeholders)  
✅ Meta-blender combining all predictions  

Batch mode allows multiple tickers.
""")

tickers_input = st.text_area("Enter tickers separated by commas", "AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

forecast_days = st.number_input("Forecast days (short-term)", min_value=1, max_value=30, value=5)

if st.button("Run Forecasts"):
    all_results = {}
    for ticker in tickers:
        try:
            results = predict_stock(ticker, forecast_days=forecast_days)
            all_results[ticker] = results
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")

    df = pd.DataFrame(all_results).T
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv().encode('utf-8')
    st.download_button("Download CSV", csv, file_name="forecast_results.csv", mime="text/csv")
