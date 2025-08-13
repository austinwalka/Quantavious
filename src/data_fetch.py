# src/data_fetch.py
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

def fetch_prices(ticker, start="2019-01-01", end=None):
    """
    Fetch daily OHLCV data from Yahoo Finance and return DataFrame with 'Close' prices.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        return None
    data.index = pd.to_datetime(data.index)
    data = data[['Close']]
    data.index.name = 'Date'
    return data


def fetch_option_chain(ticker, expiration=None):
    """
    Fetch option chain data for the nearest expiration or specific date.
    Returns two DataFrames: calls and puts.
    """
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None, None

    if expiration is None or expiration not in expirations:
        expiration = expirations[0]

    opt_chain = stock.option_chain(expiration)
    calls = opt_chain.calls
    puts = opt_chain.puts

    # Convert date columns to datetime if exist
    for df in [calls, puts]:
        if 'lastTradeDate' in df.columns:
            df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])

    return calls, puts


def bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes price formula for European options.
    S: spot price
    K: strike price
    T: time to maturity (in years)
    r: risk-free rate
    sigma: volatility (annualized)
    option_type: 'call' or 'put'
    """
    if T <= 0:
        # Option expired
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def iv_from_price(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=200):
    """
    Implied volatility calculation via bisection method.
    S: spot price
    K: strike price
    T: time to maturity (years)
    r: risk-free rate
    market_price: observed option price
    option_type: 'call' or 'put'
    """
    if market_price < 0.01:
        return np.nan  # Price too low or invalid

    sigma_low, sigma_high = 1e-5, 5.0
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = bs_price(S, K, T, r, sigma_mid, option_type)
        diff = price_mid - market_price
        if abs(diff) < tol:
            return sigma_mid
        if diff > 0:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return sigma_mid


def enrich_chain_with_iv(calls, puts, S, r):
    """
    Add implied volatility column to calls and puts DataFrames.
    """
    today = pd.Timestamp.today()
    for df, opt_type in [(calls, 'call'), (puts, 'put')]:
        ivs = []
        for _, row in df.iterrows():
            T = (row['expiration'] - today).days / 365.0
            if T <= 0:
                ivs.append(np.nan)
                continue
            iv = iv_from_price(S, row['strike'], T, r, row['lastPrice'], opt_type)
            ivs.append(iv)
        df['impliedVolatility'] = ivs
    return calls, puts
