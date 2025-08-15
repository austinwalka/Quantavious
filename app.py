import io
import pandas as pd
import streamlit as st

from src.predict_pipeline import run_prediction

st.set_page_config(page_title="Quantavious — Ensemble Stock Forecaster", layout="wide")

st.title("📈 Quantavious — Ensemble Stock Forecaster")
st.markdown("""
This app predicts short-term stock movements using a blended ensemble:

**Math-based models**: GBM, OU/Langevin, Boltzmann, Schrödinger proxy  
**Prophet**: trend & seasonality  
**ML trees**: LightGBM, XGBoost  
**Meta-blend**: simple average of all models

Enter one or more tickers (comma-separated). Choose 1–5 days ahead for short-term forecasts.
""")

tickers_txt = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT")
days = st.slider("Days ahead", min_value=1, max_value=5, value=5, step=1)

if st.button("Run predictions"):
    tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
    all_rows = []
    errors = []

    with st.spinner("Running models..."):
        for t in tickers:
            try:
                df = run_prediction(t, days=days)
                all_rows.append(df)
            except Exception as e:
                errors.append(f"{t}: {e}")

    if errors:
        st.warning("Some tickers failed:\n\n" + "\n".join(f"- {e}" for e in errors))

    if all_rows:
        results = pd.concat(all_rows, ignore_index=True)
        st.subheader("Forecast Table")
        st.dataframe(results, use_container_width=True)

        # Simple chart: show MetaBlend by ticker
        st.subheader("MetaBlend forecast (per ticker)")
        for t in results["ticker"].unique():
            st.line_chart(
                results.loc[results["ticker"] == t, ["date", "MetaBlend"]]
                .set_index("date")
            )

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="quantavious_predictions.csv",
            mime="text/csv",
        )
