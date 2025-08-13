# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

from src.predict_pipeline import predict_stock, load_finbert_pipeline_cached

st.set_page_config(page_title="Quantavius — Short-Term (Meta Blend + LSTM + Schrodinger)", layout="wide")
st.title("Quantavius — Short-Term Forecasting (LSTM + Schrodinger + Meta-Blend)")

st.markdown("""
**What this app does**
- Runs short-term forecasts (intraday 1m/5m/15m or daily) using:
  - GBM Monte Carlo (mean path)
  - Ornstein-Uhlenbeck (Langevin) mean-reverting model
  - LSTM (PyTorch) trained on recent price history
  - Schrödinger quantum evolution (PDF terminal proxy)
- Blends model outputs into a **Best_Guess** using RMSE-based weights and a **finBERT** sentiment tilt.
- Batch mode: run multiple tickers at once and download CSV of combined forecasts.
""")

with st.sidebar:
    st.header("Run settings")
    tickers_text = st.text_area("Tickers (comma-separated)", value="AAPL, MSFT, NVDA")
    mode = st.selectbox("Mode", ["intraday", "daily"], help="intraday uses 1m/5m intervals; daily uses business days")
    interval = st.selectbox("Interval (intraday)", ["1m", "5m", "15m"], index=0)
    lookback_days = st.slider("Lookback days", min_value=1, max_value=14, value=5)
    forecast_steps = st.slider("Forecast steps (bars/periods)", min_value=10, max_value=240, value=60)
    n_paths = st.slider("Simulation paths (GBM/OU)", min_value=100, max_value=2000, value=800, step=100)
    use_finbert = st.checkbox("Load finBERT (may be heavy)", value=False)
    news_key = os.getenv("NEWS_API_KEY", None)
    st.caption("NEWS_API_KEY loaded: " + ("✅" if news_key else "⚠️ Missing; sentiment will be heuristic"))

col_left, col_right = st.columns([3, 1])

# optionally pre-load finbert pipeline
preload_btn = st.button("Pre-load finBERT model (cached)")
if use_finbert and preload_btn:
    with st.spinner("Loading finBERT (this may take a while)..."):
        pipe = load_finbert_pipeline_cached()
        if pipe is None:
            st.warning("finBERT not available or failed to load; falling back to heuristic.")
        else:
            st.success("finBERT pipeline loaded (cached).")

run_btn = st.button("Run batch forecasts")

def parse_tickers(text):
    return [t.strip().upper() for t in text.split(",") if t.strip()]

if run_btn:
    tickers = parse_tickers(tickers_text)
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        combined = []
        summary_rows = []
        for t in tickers:
            with st.spinner(f"Running {t}..."):
                try:
                    forecast_df, rmse_dict, weights = predict_stock(
                        ticker=t,
                        start_date=str(datetime.now().date()),   # not used much, fetch uses lookback
                        end_date=str(datetime.now().date()),
                        mode=mode,
                        interval=interval,
                        lookback_days=lookback_days,
                        forecast_steps=forecast_steps,
                        n_paths=n_paths,
                        news_api_key=news_key,
                        tilt_strength=0.5,
                        lstm_lookback=60,
                        lstm_epochs=20
                    )
                    forecast_df = forecast_df.reset_index().rename(columns={"index":"Step"})
                    forecast_df["Ticker"] = t
                    combined.append(forecast_df)
                    summary_rows.append({
                        "Ticker": t,
                        "RMSE_GBM": rmse_dict.get("GBM", np.nan),
                        "RMSE_OU": rmse_dict.get("OU", np.nan),
                        "RMSE_ML": rmse_dict.get("ML", np.nan),
                        "Sentiment": float(forecast_df["Sentiment"].iloc[0]) if "Sentiment" in forecast_df.columns else np.nan
                    })
                    # show quick plot for this ticker
                    with col_left:
                        st.subheader(f"{t} — Best Guess vs components")
                        st.line_chart(forecast_df.set_index("Step")[["GBM","OU","ML","Best_Guess"]])
                    with col_right:
                        st.metric(label=f"{t} Schrodinger mean", value=f"{forecast_df['Schrodinger_mean'].iloc[0]:.2f}")
                        st.write("Blend weights")
                        st.json(weights)
                except Exception as e:
                    st.error(f"{t} failed: {e}")

        if combined:
            all_df = pd.concat(combined, ignore_index=True)
            st.subheader("Combined batch forecasts")
            st.dataframe(all_df)
            csv = all_df.to_csv(index=False)
            b = csv.encode("utf-8")
            st.download_button("Download CSV of combined forecasts", b, file_name="quantavius_forecasts.csv", mime="text/csv")

            summary_df = pd.DataFrame(summary_rows)
            st.subheader("Summary RMSE & Sentiment")
            st.dataframe(summary_df.set_index("Ticker"))

st.markdown("---")
st.caption("Notes: LSTM is trained on short history and cached per process. finBERT is optional; if unavailable the sentiment uses a heuristic. For production/backtesting use larger history and walk-forward CV.")
