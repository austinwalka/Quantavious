import streamlit as st
import pandas as pd
import boto3
from boto3.session import Session
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DigitalOcean Spaces configuration
ACCESS_KEY = "DO801BCYQYGPE2CXH697"  # Replace with your key
SECRET_KEY = "virbY1dJdNa+BzEVxyPBZIC/mZRcntLxLqy0H6A8QVc"  # Replace with your key
REGION = "nyc3"
SPACE_NAME = "quantavious-data"
ENDPOINT_URL = f"https://{REGION}.digitaloceanspaces.com"

# Initialize S3 client
session = Session()
s3_client = session.client('s3', region_name=REGION, endpoint_url=ENDPOINT_URL,
                          aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

# Cache data fetching to improve performance
@st.cache_data(ttl=3600)
def fetch_file_from_spaces(key):
    try:
        response = s3_client.get_object(Bucket=SPACE_NAME, Key=key)
        if key.endswith('.csv'):
            df = pd.read_csv(response['Body'])
            return df
        elif key.endswith('.json'):
            data = json.load(response['Body'])
            return data
        return None
    except Exception as e:
        logger.error(f"Error fetching {key} from Spaces: {e}")
        return None

# Fetch S&P 500 tickers
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(df["Symbol"].tolist()[:500])
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []

# Main app
st.set_page_config(page_title="Quantavious Dashboard", layout="wide")
st.title("Quantavious S&P 500 Forecasting Dashboard")
st.markdown("Visualize stock price forecasts, correlations, and backtest results for S&P 500 tickers.")

# Sidebar for ticker selection
tickers = get_sp500_tickers()
if not tickers:
    st.error("Failed to fetch S&P 500 tickers. Please try again later.")
    st.stop()

ticker = st.sidebar.selectbox("Select Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Hourly"], index=0)
timeframe_lower = timeframe.lower()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Forecast", "Correlations", "Backtest Metrics"])

# Forecast Tab
with tab1:
    st.header(f"{ticker} {timeframe} Forecast")
    forecast_key = f"{ticker}/{timeframe_lower}/forecast_{'5d_hourly' if timeframe_lower == 'hourly' else '30d'}.csv"
    forecast_df = fetch_file_from_spaces(forecast_key)
    
    if forecast_df is not None and not forecast_df.empty:
        st.write(f"Forecast data for {ticker} ({timeframe})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df[timeframe_lower], y=forecast_df["math_mean"], name="Math Mean", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast_df[timeframe_lower], y=forecast_df["lstm_mean"], name="LSTM Mean", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=forecast_df[timeframe_lower], y=forecast_df["meta_blended"], name="Meta Blended", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=forecast_df[timeframe_lower], y=forecast_df["crash_prob"], name="Crash Probability", line=dict(color="orange"), yaxis="y2"))
        fig.update_layout(
            title=f"{ticker} {timeframe} Forecast",
            xaxis_title="Day" if timeframe_lower == "daily" else "Hour",
            yaxis_title="Price",
            yaxis2=dict(title="Crash Probability", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(forecast_df)
    else:
        st.warning(f"No forecast data available for {ticker} ({timeframe}). This may be due to insufficient data (e.g., WMB) or processing errors (e.g., AFL).")

# Correlations Tab
with tab2:
    st.header("Correlation Analysis")
    corr_key = f"correlation_analysis_{timeframe_lower}.csv"
    corr_df = fetch_file_from_spaces(corr_key)
    
    if corr_df is not None and not corr_df.empty:
        st.write(f"Correlation analysis for {timeframe} data across all tickers")
        fig = px.bar(corr_df, x="metric_pair", y="correlation", title=f"{timeframe} Correlation Analysis")
        fig.update_layout(xaxis_title="Metric Pair", yaxis_title="Pearson Correlation", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.write(corr_df)
    else:
        st.warning(f"No correlation data available for {timeframe}.")

# Backtest Metrics Tab
with tab3:
    st.header(f"{ticker} Backtest Metrics")
    meta_key = f"{ticker}/{timeframe_lower}/meta.json"
    meta_data = fetch_file_from_spaces(meta_key)
    
    if meta_data:
        st.write(f"Backtest results for {ticker} ({timeframe})")
        backtest = meta_data.get("backtest", {})
        avg_rmse = backtest.get("avg_rmse")
        fold_rmse = backtest.get("fold_rmse", [])
        
        if avg_rmse is not None:
            st.metric("Average RMSE", f"{avg_rmse:.4f}")
        else:
            st.warning("No average RMSE available.")
        
        if fold_rmse:
            rmse_df = pd.DataFrame({"Fold": range(1, len(fold_rmse) + 1), "RMSE": fold_rmse})
            fig = px.line(rmse_df, x="Fold", y="RMSE", title=f"{ticker} Walk-Forward Backtest RMSE")
            st.plotly_chart(fig, use_container_width=True)
            st.write(rmse_df)
        else:
            st.warning("No fold RMSE data available.")
        
        st.subheader("Meta Data")
        st.json({
            "Symbol": meta_data.get("symbol"),
            "Timestamp": meta_data.get("timestamp"),
            "Retail %": meta_data.get("retail_pct"),
            "Institutional %": meta_data.get("inst_pct"),
            "Retail Buy %": meta_data.get("retail_buy_pct"),
            "Institutional Buy %": meta_data.get("inst_buy_pct")
        })
    else:
        st.warning(f"No backtest data available for {ticker} ({timeframe}).")

# Footer
st.markdown("---")
st.markdown(f"Data processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by Streamlit & DigitalOcean Spaces")