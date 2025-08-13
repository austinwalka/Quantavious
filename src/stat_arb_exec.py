# src/stat_arb_exec.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int            # +1 = long spread (long A short B), -1 short spread
    entry_price: float   # spread price at entry
    exit_price: float    # spread price at exit
    notional: float      # dollars allocated to trade (positive)
    pnl: float           # realized profit & loss in dollars
    return_pct: float    # pnl / notional
    fees: float          # total fees paid

def zscore(spread: pd.Series, win: int = 60) -> pd.Series:
    mu = spread.rolling(win).mean()
    sd = spread.rolling(win).std(ddof=1)
    return (spread - mu) / sd

def pair_execution_backtest(priceA: pd.Series,
                            priceB: pd.Series,
                            hedge_ratio: float,
                            entry_z: float = 2.0,
                            exit_z: float = 0.5,
                            window: int = 60,
                            slippage_bps: float = 0.0005,
                            commission_bps: float = 0.0005,
                            notional_per_trade: float = 10000.0) -> Dict:
    """
    Execution-style backtest for a single pair:
    - priceA, priceB: aligned price series (pd.Series with same index)
    - hedge_ratio: units of B to hedge A (A - hr*B)
    - entry_z, exit_z: thresholds
    - slippage_bps/commission_bps: fraction (0.0005 = 0.05%)
    - notional_per_trade: dollars sized per trade
    Returns: dictionary with trades list, trade-level PnL, equity curve, metrics.
    """
    df = pd.DataFrame({"A": priceA, "B": priceB}).dropna()
    spread = df["A"] - hedge_ratio * df["B"]
    z = zscore(spread, win=window).fillna(0.0)

    in_trade = False
    trades: List[Trade] = []
    entry_idx = None
    entry_side = 0
    entry_spread = 0.0
    fees_total = 0.0

    equity = []  # cumulative pnl over time (per timestamp)
    cum_pnl = 0.0

    for t in range(window, len(df)):
        idx = df.index[t]
        zt = z.iloc[t]
        st_price = spread.iloc[t]

        if not in_trade:
            # entry logic
            if zt > entry_z:
                # spread rich: short spread (sell A, buy hr*B) -> side = -1
                side = -1
            elif zt < -entry_z:
                # spread cheap: long spread -> side = +1
                side = +1
            else:
                side = 0

            if side != 0:
                in_trade = True
                entry_idx = idx
                entry_side = side
                entry_spread = st_price
                # fees and slippage assume round-trip later; we can estimate entry fees now
                entry_fee = notional_per_trade * (slippage_bps + commission_bps)
                fees_total += entry_fee
        else:
            # in trade: check exit
            if abs(zt) < exit_z:
                # exit
                exit_idx = idx
                exit_spread = st_price
                # pnl: for a long spread (side=1) pnl = (exit - entry)
                pnl_per_notional = (exit_spread - entry_spread) * entry_side
                # Convert spread points to dollars: assume notional maps to spread units via entry_spread scaling
                # Simplify: treat spread returns as proportional to notional: return_pct = pnl_per_notional / entry_spread
                if entry_spread == 0:
                    return_pct = 0.0
                else:
                    return_pct = pnl_per_notional / abs(entry_spread)
                pnl = notional_per_trade * return_pct
                # fees
                exit_fee = notional_per_trade * (slippage_bps + commission_bps)
                fees_total += exit_fee
                total_fees = entry_fee + exit_fee if 'entry_fee' in locals() else exit_fee
                cum_pnl += pnl - total_fees
                trade = Trade(entry_time=entry_idx, exit_time=exit_idx, side=entry_side,
                              entry_price=entry_spread, exit_price=exit_spread,
                              notional=notional_per_trade, pnl=pnl - total_fees,
                              return_pct=return_pct, fees=total_fees)
                trades.append(trade)
                in_trade = False
                entry_idx = None
                entry_side = 0
                entry_spread = 0.0
                # reset entry_fee variable
                if 'entry_fee' in locals():
                    del entry_fee
        equity.append(cum_pnl)

    equity_idx = df.index[window:window+len(equity)]
    equity_curve = pd.Series(equity, index=equity_idx).fillna(method='ffill').fillna(0.0)

    # metrics
    returns = equity_curve.diff().fillna(0.0)
    annualized_return = (equity_curve.iloc[-1] / (notional_per_trade + 1e-9)) * (252.0 / max(1, len(returns)))
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252) if volatility > 0 else 0.0
    max_dd = float((equity_curve / equity_curve.cummax() - 1.0).min())

    summary = {
        "trades": trades,
        "equity_curve": equity_curve,
        "total_pnl": float(cum_pnl),
        "n_trades": len(trades),
        "annualized_return": annualized_return,
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "total_fees": float(fees_total),
    }
    return summary
