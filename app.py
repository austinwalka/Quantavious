import streamlit as st
import pandas as pd
from src.predict_pipeline import predict_stock
import yfinance as yf

# Load S&P 500 tickers and filter out tickers with "."
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tickers_df = pd.read_html(sp500_url)[0]
TICKERS = [t for t in tickers_df['Symbol'].to_list() if '.' not in t]

st.title("S&P 500 Stock Prediction Dashboard")

all_results = []

# Run predictions for each ticker
for ticker in TICKERS:
    try:
        result = predict_stock(ticker, retrain_if_missing=False)  # only retrain if needed
        if result:
            result["Ticker"] = ticker
            all_results.append(result)
    except Exception as e:
        st.write(f"Error processing {ticker}: {e}")

# Convert results to DataFrame (safe from scalar value error)
if all_results:
    df = pd.DataFrame(all_results)
    df = df[["Ticker", "Last Close", "Predicted Price", "GBM", "OU", "Schrodinger", "Prediction Date"]]
    st.dataframe(df)
else:
    st.warning("No predictions generated. Check model files or data source.")

# Optional: Let user pick a ticker and see chart
st.subheader("Stock Price History")
selected_ticker = st.selectbox("Select a Ticker", TICKERS)
if selected_ticker:
    hist = yf.download(selected_ticker, period="6mo", interval="1d")
    st.line_chart(hist["Close"])
