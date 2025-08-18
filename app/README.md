# Quantavious App

This Streamlit app visualizes **price forecasts and crash risk** for S&P 500 stocks using a **kitchen-sink approach**:

### Models
- **Math Signals (20%)**: GARCH volatility, EVT tail, HMM regimes, VIX term slope, breadth crashiness
- **LSTM (50%)**: Sequence model on past 60 days
- **FinBERT (30%)**: Daily headline sentiment (neg/pos scores)

### Meta-Blender
- Combines all predictions:
  `p_meta = 0.2 * p_math + 0.5 * p_lstm + 0.3 * p_finbert`

### Features
- **60-day forecasts** with price & crash risk %
- **Individual model contributions**
- **Technical indicators** (SMA20, Bollinger Bands, RSI, MACD)
- **Backtesting support** via Colab precomputed CSVs

### Usage
1. Run `streamlit run app.py`
2. Enter a stock ticker
3. If forecast CSV is missing or older than 1 day, run Colab training
4. Visualize forecasts & technical indicators
