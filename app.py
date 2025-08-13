# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

from src.predict_pipeline import predict_stock

st.set_page_config(page_title="Quantavius - Full Stack", layout="wide")
st.title("Quantavius — Full Stack Forecast (LSTM / XGB / LGBM / Prophet / Quantum)")

st.markdown("""
This app runs short-term forecasts and blends them:
- Math models: GBM, Ornstein-Uhlenbeck (Langevin), Schrödinger
- ML: LSTM (PyTorch), optional LightGBM/XGBoost
- Prophet: optional trend/seasonality
- Meta-blend weighted by RMSE and tilted by finBERT sentiment (optional)
""")

with st.sidebar:
    st.header("Settings")
    tickers_in = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA")
    mode = st.selectbox("Mode", ["intraday", "daily"])
    interval = st.selectbox("Intraday interval", ["1m","5m","15m"])
    lookback_days = st.slider("Lookback days", 1, 14, 5)
    steps = st.slider("Forecast steps", 10, 240, 60)
    n_paths = st.slider("GBM/OU simulation paths", 100, 2000, 800, step=100)
    use_prophet = st.checkbox("Enable Prophet (optional)", value=False)
    use_xgb = st.checkbox("Enable XGBoost (optional)", value=False)
    use_lgb = st.checkbox("Enable LightGBM (optional)", value=False)
    enable_finbert = st.checkbox("Use finBERT sentiment (may be heavy)", value=False)
    st.caption("Set NEWS_API_KEY in env to enable NewsAPI headlines.")

col_left, col_right = st.columns([3,1])

def parse_tickers(text):
    return [t.strip().upper() for t in text.split(",") if t.strip()]

if st.button("Run batch"):
    tickers = parse_tickers(tickers_in)
    if not tickers:
        st.warning("Enter at least one ticker")
    else:
        combined = []
        summary = []
        for t in tickers:
            with st.spinner(f"Running {t}..."):
                try:
                    forecast_df, rmse, weights = predict_stock(
                        ticker=t,
                        mode=mode,
                        interval=interval,
                        lookback_days=lookback_days,
                        forecast_steps=steps,
                        n_paths=n_paths,
                        news_api_key=os.getenv("NEWS_API_KEY", None),
                        tilt_strength=0.5,
                        use_prophet=use_prophet,
                        use_xgb=use_xgb,
                        use_lgb=use_lgb,
                        lstm_lookback=60,
                        lstm_epochs=25,
                        device="cpu"
                    )
                    forecast_df = forecast_df.reset_index().rename(columns={"index":"Step"})
                    forecast_df["Ticker"] = t
                    combined.append(forecast_df)
                    summary.append({
                        "Ticker": t,
                        "RMSE_GBM": rmse.get("GBM", np.nan),
                        "RMSE_OU": rmse.get("OU", np.nan),
                        "RMSE_ML": rmse.get("ML", np.nan),
                        "Sentiment": float(forecast_df["Sentiment"].iloc[0]) if "Sentiment" in forecast_df.columns else np.nan
                    })
                    # display plot
                    with col_left:
                        st.subheader(f"{t} - Best Guess vs components")
                        st.line_chart(forecast_df.set_index("Step")[["GBM","OU","ML","Best_Guess"]])
                    with col_right:
                        st.metric(f"{t} Schr. mean", f"{forecast_df['Schrodinger_mean'].iloc[0]:.2f}")
                        st.write("Blend weights")
                        st.json(weights)
                except Exception as e:
                    st.error(f"{t} failed: {e}")
        if combined:
            all_df = pd.concat(combined, ignore_index=True)
            st.subheader("Combined forecasts")
            st.dataframe(all_df)
            # CSV download
            csv = all_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="quantavius_forecasts.csv", mime="text/csv")
            st.subheader("Summary")
            st.dataframe(pd.DataFrame(summary).set_index("Ticker"))

st.markdown("---")
st.caption("Notes: heavy packages (Prophet, finBERT, LightGBM, XGBoost) are optional. LSTM uses PyTorch and is cached between runs.")
