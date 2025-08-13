# Quantavius - AI Stock Ensemble Pipeline

Quantavius is an AI-powered stock price prediction pipeline inspired by Ed Quantavius's quantitative trading philosophy.  
It combines multiple models — including gradient boosting, deep learning, and sentiment analysis — to produce predictions with uncertainty estimates.

## Features
- **Data Ingestion** from Yahoo Finance & news APIs
- **Feature Engineering** with technical indicators & sentiment
- **Model Ensemble** (LightGBM, XGBoost, Prophet, Transformers)
- **Meta-Learner** trained on historical model performance
- **Uncertainty Estimation**
- **Backtesting**
- **Streamlit UI** for interactive predictions

## Installation
```bash
git clone https://github.com/austinwalka/Quantavius.git
cd Quantavius
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
