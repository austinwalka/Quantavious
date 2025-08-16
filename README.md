# Quantavious - AI Stock Ensemble Pipeline

Quantavious is an AI-powered stock price prediction pipeline inspired by Ed Quantavious's quantitative trading philosophy.  
It combines multiple models — including gradient boosting, deep learning, and sentiment analysis — to produce predictions with uncertainty estimates.

## Features
- **Data Ingestion** from Yahoo Finance & news APIs
- **Feature Engineering** with technical indicators & sentiment
- **Model Ensemble** (LightGBM, XGBoost, Prophet, Transformers)
- **Meta-Learner** trained on historical model performance
- **Uncertainty Estimation**
- **Backtesting**
- **Streamlit UI** for interactive predictions


🧠 What each component does (and why)

GBM (Geometric Brownian Motion)
Assumes log-returns are normally distributed with drift μ and volatility σ. Good baseline for medium/long-term drift behavior.

We compute μ and σ from recent log-returns.

Use closed-form expectation for the point forecast.

Use Monte-Carlo to estimate VaR/CVaR tail risk.

OU (Ornstein–Uhlenbeck)
Mean-reverting process on log-prices. Good for short-term pullbacks or reversion to equilibrium.

We estimate κ (speed), μ_log (equilibrium), and daily noise via OLS.

Forecast is the OU mean at the chosen horizon.

ARIMA(1,1,1)
Captures short-term autocorrelation after differencing. Helpful for 1d–1w horizons.

If data is too short/unstable, we gracefully drop it and re-weigh the blend.

Schrödinger-inspired FFT model
We detect dominant cycles in detrended returns via FFT and project a sinusoid on log-returns — a proxy for multi-modal uncertainty / “probability amplitude” behavior. It often helps on 1w–3mo when there are seasonal/weekly patterns.

Meta-Blend (adaptive)
Horizon-aware weights:

1d: ARIMA/OU dominate, a little GBM & FFT

5d: still short-term heavy

1mo+: increasingly GBM & FFT, less ARIMA

Risk (VaR/CVaR)
From GBM Monte-Carlo distribution over the horizon:

VaR(95%): a 1-in-20 worst-loss threshold

CVaR(95%): average loss beyond VaR (tail severity)


## Installation
```bash
git clone https://github.com/austinwalka/Quantavious.git
cd Quantavious
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
