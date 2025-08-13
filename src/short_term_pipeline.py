# src/short_term_pipeline.py
import os
import numpy as np
import pandas as pd
from src.intraday_fetch import get_price_data
from src.math_models import simulate_gbm, estimate_ou_params, simulate_ou, schrodinger_pdf, fokker_planck_pdf
from src.patterns_ml import short_horizon_ml, template_match

def short_term_forecast(ticker: str, mode: str = "intraday", days: int = 5, interval: str = "1m"):
    df = get_price_data(ticker, mode=mode, days=days, interval=interval)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    last = df["Close"].iloc[-1]
    # step/time setup
    if mode == "intraday":
        # predict next N bars up to ~1 day; aggregate to 5d as needed
        steps = 60  # ~ next 60 intervals
        dt = 1/390  # one 1-min bar as fraction of trading day (approx)
    else:
        steps = min(5, len(df))  # next 5 sessions
        dt = 1/252

    # empirical drift/vol from recent window
    log_ret = np.log(df["Close"]).diff().dropna().values
    mu_hat = np.mean(log_ret)/dt
    sigma_hat = np.std(log_ret)/np.sqrt(dt)

    # --- Math models ---
    gbm_paths = simulate_gbm(last, mu_hat, sigma_hat, steps=steps, dt=dt, n_paths=1000)
    gbm_mean_path = gbm_paths.mean(axis=0)

    # OU on returns, integrate to price
    if len(log_ret) > 10:
        ou_params = estimate_ou_params(log_ret, dt=dt)
        ou_r = simulate_ou(0.0, ou_params, steps=steps, dt=dt, n_paths=1000)
        ou_price = last * np.exp(np.cumsum(ou_r.mean(axis=0)))
    else:
        ou_price = np.repeat(last, steps+1)

    # Schrödinger PDF (for terminal distribution)
    S_grid, px = schrodinger_pdf(np.log(last), sigma_hat, T=steps*dt, N_x=1024)
    sch_mean = np.trapz(S_grid*px, S_grid)

    # Fokker-Planck PDF for returns
    r_grid, pr = fokker_planck_pdf(0.0, mu=np.mean(log_ret), sigma=np.std(log_ret), steps=max(1, steps//5), dt=dt)
    boltz_mean = last * (1 + np.trapz(r_grid*pr, r_grid))

    # --- Data-driven ML (short horizon) ---
    ml_path = short_horizon_ml(df, horizon_steps=steps)
    if len(ml_path) < steps:
        ml_path = np.pad(ml_path, (0, steps - len(ml_path)), constant_values=last)

    # Pattern recognition (nearest motifs) -> boost
    motifs = template_match(df["Close"], lookback=60, k=5)
    motif_edge = np.nanmean(motifs) if motifs.size else 0.0
    ml_path = ml_path * (1 + motif_edge)  # tiny tilt

    # Collect outputs
    idx = pd.RangeIndex(0, steps+1, name="t")
    out = pd.DataFrame({
        "GBM": gbm_mean_path,
        "Langevin": ou_price,
        "ML_Path": np.r_[last, ml_path[:steps]]
    }, index=idx)

    # terminal stats
    terminal = pd.DataFrame({
        "Schrodinger_mean": [sch_mean],
        "Boltzmann_mean": [boltz_mean]
    })

    return df, out, terminal
