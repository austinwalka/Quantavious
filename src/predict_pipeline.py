import numpy as np
import pandas as pd
import yfinance as yf

# Optional: tiny lib; if absent, we continue without caching
try:
    import joblib  # noqa: F401
except Exception:
    joblib = None  # not used in this lightweight runtime

# ML libs
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb

# --- Utilities ---------------------------------------------------------------

def _fetch_history(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    """Fetch OHLCV and return df with columns: Date, Close, ds, y."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({period}, {interval}).")
    df = df.reset_index()
    # Some intervals name the index differently (Date vs Datetime)
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df.rename(columns={date_col: "ds", "Close": "y"}, inplace=True)
    df = df[["ds", "y"]].dropna()
    if df.shape[0] < 10:
        raise ValueError(f"Not enough rows for {ticker} to model (have {df.shape[0]}).")
    return df

def _future_dates(last_date: pd.Timestamp, days: int) -> pd.DatetimeIndex:
    # Next trading days approximation (calendar days; for exact trading days integrate a calendar)
    start = (pd.to_datetime(last_date).normalize() + pd.Timedelta(days=1))
    return pd.date_range(start, periods=days, freq="D")

# --- Math models -------------------------------------------------------------

def _gbm_series(S0, mu, sigma, days):
    dt = 1 / 252
    prices = [S0]
    for _ in range(days):
        dW = np.random.normal(scale=np.sqrt(dt))
        next_S = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(next_S)
    return np.array(prices[1:])

def _ou_series(S0, mu, theta, sigma, days):
    dt = 1 / 252
    x = [S0]
    for _ in range(days):
        dx = theta * (mu - x[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x.append(x[-1] + dx)
    return np.array(x[1:])

def _boltzmann_proxy_series(S0, days):
    # simple small random kicks; proxy for energy diffusion
    shocks = np.random.normal(0, 0.006, size=days)  # ~0.6% daily stdev
    return S0 * (1 + shocks).cumprod()

def _schrodinger_proxy_series(S0, days):
    # proxy w/ slightly fatter tails than boltzmann
    shocks = np.random.normal(0, 0.009, size=days)
    return S0 * (1 + shocks).cumprod()

# --- Prophet ----------------------------------------------------------------

def _prophet_series(hist_df: pd.DataFrame, days: int) -> np.ndarray:
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(hist_df.rename(columns={"ds": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=days, freq="D", include_history=False)
    fc = m.predict(future)
    return fc["yhat"].to_numpy()

# --- Lightweight ML (no deep learning) --------------------------------------

def _build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["t"] = np.arange(len(tmp))
    tmp["ret"] = tmp["y"].pct_change().fillna(0.0)
    tmp["ma5"] = tmp["y"].rolling(5).mean().bfill()
    tmp["ma10"] = tmp["y"].rolling(10).mean().bfill()
    tmp["std5"] = tmp["y"].rolling(5).std().bfill().fillna(0.0)
    tmp["std10"] = tmp["y"].rolling(10).std().bfill().fillna(0.0)
    return tmp

def _future_feature_frame(last_row: pd.Series, start_t: int, days: int) -> pd.DataFrame:
    # We only extrapolate 't'; MAs/stds are unknown for the future – the trees still use time index.
    future = pd.DataFrame({"t": np.arange(start_t + 1, start_t + 1 + days)})
    # Fill placeholder numeric columns used in training (keeps columns aligned)
    for col in ["ret", "ma5", "ma10", "std5", "std10"]:
        future[col] = float(last_row[col]) if col in last_row else 0.0
    return future

def _lgb_series(feat: pd.DataFrame, y: pd.Series, days: int) -> np.ndarray:
    X = feat[["t", "ret", "ma5", "ma10", "std5", "std10"]]
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y)
    future_X = _future_feature_frame(feat.iloc[-1], feat["t"].iloc[-1], days)[X.columns]
    return model.predict(future_X)

def _xgb_series(feat: pd.DataFrame, y: pd.Series, days: int) -> np.ndarray:
    X = feat[["t", "ret", "ma5", "ma10", "std5", "std10"]]
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X, y)
    future_X = _future_feature_frame(feat.iloc[-1], feat["t"].iloc[-1], days)[X.columns]
    return model.predict(future_X)

# --- Public API --------------------------------------------------------------

def run_prediction(ticker: str, days: int = 5) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with one row per forecast day and columns:
      ['ticker','date','GBM','OU','Boltzmann','Schrodinger','Prophet','LightGBM','XGBoost','MetaBlend']
    """
    hist = _fetch_history(ticker, period="1y", interval="1d")
    last_price = float(hist["y"].iloc[-1])
    last_date = pd.to_datetime(hist["ds"].iloc[-1])
    future_dates = _future_dates(last_date, days)

    # Params for math models
    ret = hist["y"].pct_change().dropna()
    mu = float(ret.mean()) if len(ret) else 0.0
    sigma = float(ret.std()) if len(ret) else 0.02
    theta = 0.5  # OU pull-to-mean

    # Math series
    gbm = _gbm_series(last_price, mu, sigma, days)
    ou = _ou_series(last_price, last_price, theta, sigma, days)
    boltz = _boltzmann_proxy_series(last_price, days)
    sch = _schrodinger_proxy_series(last_price, days)

    # Prophet
    prop = _prophet_series(hist[["ds", "y"]], days)

    # ML trees
    feat = _build_basic_features(hist[["ds", "y"]].copy())
    lgb_ser = _lgb_series(feat, feat["y"], days)
    xgb_ser = _xgb_series(feat, feat["y"], days)

    # Meta-blend (equal weights to keep simple)
    stack = np.vstack([gbm, ou, boltz, sch, prop, lgb_ser, xgb_ser])
    meta = stack.mean(axis=0)

    out = pd.DataFrame({
        "ticker": ticker,
        "date": future_dates,
        "GBM": gbm,
        "OU": ou,
        "Boltzmann": boltz,
        "Schrodinger": sch,
        "Prophet": prop,
        "LightGBM": lgb_ser,
        "XGBoost": xgb_ser,
        "MetaBlend": meta,
    })
    return out
