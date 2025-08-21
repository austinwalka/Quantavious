import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import requests
import json
from datetime import datetime, timedelta
from io import StringIO

st.set_page_config(page_title="Quantavious Dashboard", layout="wide")
st.title("Quantavious Stock Forecast Dashboard")

SPACE_URL = "https://quantavious-data.nyc3.digitaloceanspaces.com"

@st.cache_data(ttl=3600)
def fetch_file(path):
    try:
        response = requests.get(f"{SPACE_URL}/{path}")
        response.raise_for_status()
        if path.endswith(".csv"):
            return pd.read_csv(StringIO(response.text))
        elif path.endswith(".json"):
            return json.loads(response.text)
        return None
    except Exception as e:
        st.warning(f"Error fetching {path}: {e}")
        return None

@st.cache_data
def get_tickers():
    summary = fetch_file("run_summary.json")
    if summary:
        return list(set([r["symbol"] for r in summary["daily"] + summary["hourly"] if r["status"] == "success"]))
    return []

def load_ticker_data(ticker, timeframe="daily"):
    forecast_file = f"{ticker}/{timeframe}/forecast_{'5d_hourly' if timeframe == 'hourly' else '30d'}.csv"
    retail_forecast_file = f"{ticker}/{timeframe}/retail_inst_forecast.csv"
    indicators_file = f"{ticker}/{timeframe}/indicators.csv"
    meta_file = f"{ticker}/{timeframe}/meta.json"
    
    forecast_df = fetch_file(forecast_file)
    retail_forecast_df = fetch_file(retail_forecast_file)
    indicators_df = fetch_file(indicators_file)
    if indicators_df is not None:
        indicators_df["timestamp"] = pd.to_datetime(indicators_df["timestamp"], errors="coerce")
    meta_data = fetch_file(meta_file)
    
    return forecast_df, retail_forecast_df, indicators_df, meta_data

def plot_forecast(df, title, x_label="Day"):
    if df is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_label.lower()], y=df["math_mean"], name="Math", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df[x_label.lower()], y=df["lstm_mean"], name="LSTM", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df[x_label.lower()], y=df["meta_blended"], name="Blended", line=dict(color="red", width=3)))
    fig.add_trace(go.Scatter(x=df[x_label.lower()], y=df["crash_prob"] * df["meta_blended"].max(), name="Crash Prob", line=dict(color="orange", dash="dash")))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Price / Crash Probability", template="plotly_white")
    return fig

def plot_retail_inst(df, metric, title, x_label="Day"):
    if df is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_label.lower()], y=df[metric], name=metric.replace("_", " ").title(), line=dict(color="purple")))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=metric.replace("_", " ").title(), template="plotly_white")
    return fig

def plot_indicators(df, y_col, title, date_range=None):
    if df is None:
        return None
    df = df.copy()
    if date_range and "timestamp" in df.columns:
        start_date, end_date = date_range
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[y_col], name=y_col, line=dict(color="purple")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_col, template="plotly_white")
    return fig

def find_biggest_movers(tickers, days_out, num_movers, metric="meta_blended", timeframe="daily"):
    movers = []
    for ticker in tickers:
        forecast_df, _, _, _ = load_ticker_data(ticker, timeframe)
        if forecast_df is not None and len(forecast_df) >= days_out:
            current = forecast_df["meta_blended"].iloc[0]
            future = forecast_df[metric].iloc[:days_out].mean()
            change = (future - current) / current * 100 if current != 0 else 0
            movers.append({"ticker": ticker, "percentage_change": change})
    movers_df = pd.DataFrame(movers).sort_values("percentage_change", ascending=False)
    return movers_df.head(num_movers)

def plot_correlation_heatmap(corr_df, title):
    if corr_df is None or corr_df.empty:
        return None
    metrics = ["math_mean", "lstm_mean", "meta_blended", "retail_volume", "inst_volume", "retail_buy_pct", "inst_buy_pct"]
    corr_matrix = pd.DataFrame(index=metrics, columns=metrics)
    for _, row in corr_df.iterrows():
        metric1, metric2 = row["metric_pair"].split("_vs_")
        corr_matrix.loc[metric1, metric2] = row["correlation"]
        corr_matrix.loc[metric2, metric1] = row["correlation"]
    corr_matrix = corr_matrix.fillna(1.0)
    
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=metrics,
        y=metrics,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        annotation_text=corr_matrix.round(2).astype(str).values,
        showscale=True
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Metric", tickangle=45),
        yaxis=dict(title="Metric"),
        template="plotly_white"
    )
    return fig

tickers = get_tickers()
if not tickers:
    st.error("No data found in Spaces")
else:
    mode = st.sidebar.selectbox("Mode", ["Single Ticker", "Comparison", "Aggregates", "Biggest Movers"])
    timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Hourly"])

    if mode == "Single Ticker":
        ticker = st.selectbox("Select Stock", tickers)
        forecast_df, retail_forecast_df, indicators_df, meta_data = load_ticker_data(ticker, "hourly" if timeframe == "Hourly" else "daily")
        
        st.subheader(f"{timeframe} Results for {ticker}")
        if forecast_df is not None:
            st.plotly_chart(plot_forecast(forecast_df, f"{timeframe} Price Forecast for {ticker}", "Hour" if timeframe == "Hourly" else "Day"), use_container_width=True)
            st.dataframe(forecast_df)
        
        if retail_forecast_df is not None:
            st.subheader(f"{timeframe} Retail/Inst Forecast")
            metrics = ["retail_buy_pct", "inst_buy_pct"]
            for metric in metrics:
                fig = plot_retail_inst(retail_forecast_df, metric, f"{metric.replace('_', ' ').title()}", "Hour" if timeframe == "Hourly" else "Day")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            st.dataframe(retail_forecast_df)
        
        if indicators_df is not None:
            st.subheader(f"{timeframe} Indicators")
            min_date = indicators_df["timestamp"].min().date()
            max_date = indicators_df["timestamp"].max().date()
            date_range = st.date_input("Date Range", value=(max_date - timedelta(days=30), max_date), min_value=min_date, max_value=max_date)
            indicators = ["close", "RSI"]
            for ind in indicators:
                fig = plot_indicators(indicators_df, ind, f"{ind.upper()}", date_range if len(date_range) == 2 else None)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            st.dataframe(indicators_df.tail(10))
        
        if meta_data:
            st.subheader(f"{timeframe} Meta")
            st.write(f"Retail Buy %: {meta_data.get('retail_buy_pct', 'N/A'):.2%}")
            st.write(f"Inst Buy %: {meta_data.get('inst_buy_pct', 'N/A'):.2%}")

    elif mode == "Comparison":
        st.subheader(f"{timeframe} Comparison")
        all_forecasts = {ticker: load_ticker_data(ticker, "hourly" if timeframe == "Hourly" else "daily")[0] for ticker in tickers}
        all_forecasts = {k: v for k, v in all_forecasts.items() if v is not None}
        if all_forecasts:
            metric = st.selectbox("Metric", ["meta_blended", "math_mean", "lstm_mean"])
            fig = go.Figure()
            for ticker, df in all_forecasts.items():
                fig.add_trace(go.Scatter(x=df["hour" if timeframe == "Hourly" else "day"], y=df[metric], name=ticker))
            fig.update_layout(title=f"{timeframe} {metric.replace('_', ' ').title()} Comparison", xaxis_title="Hour" if timeframe == "Hourly" else "Day", yaxis_title=metric.replace("_", " ").title())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No {timeframe.lower()} forecast data available")

    elif mode == "Aggregates":
        st.subheader(f"{timeframe} Aggregated Averages")
        aggregate_file = f"aggregate_summary_{'hourly' if timeframe == 'Hourly' else 'daily'}.csv"
        df_agg = fetch_file(aggregate_file)
        if df_agg is not None:
            # Define metrics to plot
            metrics = [
                ("avg_math_mean", "Average Math Mean (GBM, OU, etc.)", "blue", "Price"),
                ("avg_lstm_mean", "Average LSTM Mean", "green", "Price"),
                ("avg_meta_blended", "Average Meta Blended", "red", "Price"),
                ("avg_crash_prob", "Average Crash Probability", "orange", "Probability"),
                ("avg_retail_buy_pct", "Average Retail Buy %", "purple", "Percentage"),
                ("avg_inst_buy_pct", "Average Institutional Buy %", "cyan", "Percentage"),
                ("avg_retail_volume", "Average Retail Volume", "magenta", "Volume"),
                ("avg_inst_volume", "Average Institutional Volume", "yellow", "Volume"),
                ("avg_retail_buy_volume", "Average Retail Buy Volume", "lime", "Volume"),
                ("avg_inst_buy_volume", "Average Institutional Buy Volume", "brown", "Volume"),
                ("avg_retail_sell_volume", "Average Retail Sell Volume", "pink", "Volume"),
                ("avg_inst_sell_volume", "Average Institutional Sell Volume", "black", "Volume")
            ]
            
            # Group metrics by y-axis type for better scaling
            price_metrics = [m for m in metrics if m[3] == "Price"]
            prob_metrics = [m for m in metrics if m[3] == "Probability"]
            pct_metrics = [m for m in metrics if m[3] == "Percentage"]
            vol_metrics = [m for m in metrics if m[3] == "Volume"]
            
            # Plot Price Metrics
            st.markdown("### Price Metrics")
            for metric, title, color, _ in price_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_agg["hour" if timeframe == "Hourly" else "day"],
                    y=df_agg[metric],
                    name=title,
                    line=dict(color=color)
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Hour" if timeframe == "Hourly" else "Day",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot Probability Metrics
            st.markdown("### Probability Metrics")
            for metric, title, color, _ in prob_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_agg["hour" if timeframe == "Hourly" else "day"],
                    y=df_agg[metric],
                    name=title,
                    line=dict(color=color)
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Hour" if timeframe == "Hourly" else "Day",
                    yaxis_title="Probability",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot Percentage Metrics
            st.markdown("### Percentage Metrics")
            for metric, title, color, _ in pct_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_agg["hour" if timeframe == "Hourly" else "day"],
                    y=df_agg[metric],
                    name=title,
                    line=dict(color=color)
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Hour" if timeframe == "Hourly" else "Day",
                    yaxis_title="Percentage",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot Volume Metrics
            st.markdown("### Volume Metrics")
            for metric, title, color, _ in vol_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_agg["hour" if timeframe == "Hourly" else "day"],
                    y=df_agg[metric],
                    name=title,
                    line=dict(color=color)
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Hour" if timeframe == "Hourly" else "Day",
                    yaxis_title="Volume",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot Correlation Analysis
            st.markdown("### Correlation Analysis")
            corr_file = f"correlation_analysis_{'hourly' if timeframe == 'Hourly' else 'daily'}.csv"
            corr_df = fetch_file(corr_file)
            if corr_df is not None:
                fig = plot_correlation_heatmap(corr_df, f"{timeframe} Correlation Matrix")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(corr_df)
            else:
                st.warning(f"No {timeframe.lower()} correlation data found")
            
            # Display DataFrame
            st.markdown("### All Aggregated Metrics")
            st.dataframe(df_agg)

    elif mode == "Biggest Movers":
        st.subheader(f"{timeframe} Biggest Movers")
        num_movers = st.number_input("Number of Movers", min_value=1, max_value=50, value=10)
        days_out = st.number_input("Days Out" if timeframe == "Daily" else "Hours Out", min_value=1, max_value=30 if timeframe == "Daily" else 32, value=30 if timeframe == "Daily" else 32)
        metric = st.selectbox("Metric", ["meta_blended", "math_mean", "lstm_mean"])
        movers_df = find_biggest_movers(tickers, days_out, num_movers, metric, "hourly" if timeframe == "Hourly" else "daily")
        if not movers_df.empty:
            st.dataframe(movers_df)
            fig = go.Figure()
            for _, row in movers_df.iterrows():
                ticker = row["ticker"]
                forecast_df = load_ticker_data(ticker, "hourly" if timeframe == "Hourly" else "daily")[0]
                if forecast_df is not None:
                    fig.add_trace(go.Scatter(x=forecast_df["hour" if timeframe == "Hourly" else "day"], y=forecast_df[metric], name=ticker))
            fig.update_layout(title=f"{timeframe} {metric.replace('_', ' ').title()} for Top Movers", xaxis_title="Hour" if timeframe == "Hourly" else "Day", yaxis_title=metric.replace("_", " ").title())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No {timeframe.lower()} forecast data for biggest movers")