import streamlit as st
from predict_pipeline import run_prediction
import pandas as pd

# -----------------------
# App Title & Description
# -----------------------
st.set_page_config(page_title="Quantavius AI", layout="wide")

st.title("📈 Quantavius AI – Advanced Market Prediction Tool")

st.markdown("""
### 🔍 What This App Does

✅ **Math-based models**
- Geometric Brownian Motion (GBM)
- Langevin Equation
- Boltzmann Equation
- Schrödinger equation time-evolution proxy (full version optional upgrade)

✅ **Prophet**
- Facebook Prophet for short- and medium-term trend + seasonality decomposition.

✅ **Meta-blender**
- Combines:
  - Math-based models (GBM, OU/Langevin, Boltzmann)
  - Machine learning models (LightGBM, XGBoost, LSTM)
  - Prophet predictions
  - **finBERT sentiment weighting** to tilt predictions toward mean-reversion or trend-following based on news sentiment.

This tool is designed for **intraday and 5-day short-term forecasting** with hedge fund-style quantitative approaches.
""")

# -----------------------
# UI Input Section
# -----------------------
symbols = st.text_input("Enter stock symbols (comma separated):", "AAPL, MSFT, TSLA")
run_button = st.button("Run Predictions")

# -----------------------
# Prediction Execution
# -----------------------
if run_button:
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    results = []

    for sym in symbol_list:
        try:
            df = run_prediction(sym)
            df['Symbol'] = sym
            results.append(df)
        except Exception as e:
            st.error(f"Error processing {sym}: {e}")

    if results:
        combined_df = pd.concat(results)
        st.subheader("📊 Prediction Results")
        st.dataframe(combined_df)

        # CSV download option
        csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download results as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
