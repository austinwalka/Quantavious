# src/backtest.py
import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    ex_ret = returns - risk_free_daily
    if ex_ret.std() == 0 or np.isnan(ex_ret.std()):
        return 0.0
    return float(np.sqrt(252) * ex_ret.mean() / ex_ret.std())

def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max) - 1.0
    return float(dd.min())

def walk_forward_backtest(
    prices: pd.Series,
    feature_frame: pd.DataFrame,
    make_predictions_fn,
    train_window: int = 252,
    test_horizon: int = 5
) -> dict:
    """
    Generic walk-forward:
    - prices: Close prices (index aligned with feature_frame)
    - feature_frame: engineered features (aligned index)
    - make_predictions_fn(train_df, test_df) -> pd.Series of preds on test_df.index
    Returns: metrics and a DataFrame of predictions vs actuals.
    """
    preds = []
    acts = []
    idxs = []

    for start in range(0, len(prices) - train_window - test_horizon, test_horizon):
        train_slice = slice(start, start + train_window)
        test_slice = slice(start + train_window, start + train_window + test_horizon)

        train_df = feature_frame.iloc[train_slice]
        test_df = feature_frame.iloc[test_slice]
        y_train = prices.iloc[train_slice]
        y_test = prices.iloc[test_slice]

        # produce preds (Series indexed to test_df.index)
        yhat = make_predictions_fn(train_df, y_train, test_df)
        yhat = pd.Series(yhat, index=y_test.index)

        preds.append(yhat)
        acts.append(y_test)
        idxs.extend(list(y_test.index))

    preds = pd.concat(preds) if preds else pd.Series(dtype=float)
    acts = pd.concat(acts) if acts else pd.Series(dtype=float)

    # Convert to daily returns for metrics
    ret = acts.pct_change().fillna(0)
    # Strategy: long if next-day forecast > last price (naive); here use yhat shift
    strat_ret = (preds.pct_change().fillna(0))  # proxy; replace with signal logic if desired

    # Equity curves
    eq_bh = (1 + ret).cumprod()
    eq_strat = (1 + strat_ret).cumprod()

    metrics = {
        "sharpe_buy_hold": sharpe_ratio(ret),
        "sharpe_strategy": sharpe_ratio(strat_ret),
        "maxdd_buy_hold": max_drawdown(eq_bh),
        "maxdd_strategy": max_drawdown(eq_strat),
    }

    out = pd.DataFrame({
        "actual": acts,
        "prediction": preds
    }).sort_index()

    return {"metrics": metrics, "detail": out}
