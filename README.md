# Quantavious: Advanced Stock Forecasting & Crash Risk

This repo contains a **Streamlit dashboard** for S&P 500 stocks showing:

- Price predictions (60 days)
- Crash risk percentage
- Individual model contributions (Math/LSTM/FinBERT)
- Technical indicators (SMA, Bollinger Bands, MACD, RSI)

### Architecture
- **Colab**: trains models & saves forecasts to CSV
- **Streamlit**: visualizes forecasts, technicals, crash risk
- **Meta-blender**: combines math + LSTM + sentiment
- **Optional retraining**: triggers Colab if data missing/stale

### Features
- Kitchen-sink modeling: stochastic math + ML + sentiment
- Walk-forward backtesting
- Fully GitHub-ready & Streamlit Cloud deployable
