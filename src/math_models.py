# src/math_models.py
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ---------- Brownian / GBM ----------
def simulate_gbm(S0, mu, sigma, steps, dt, n_paths=1000, seed=42):
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, steps)) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt
    paths = np.zeros((n_paths, steps+1))
    paths[:,0] = S0
    for t in range(steps):
        paths[:,t+1] = paths[:,t] * np.exp(drift + sigma*shocks[:,t])
    return paths

# ---------- Langevin (Ornstein-Uhlenbeck) on returns ----------
@dataclass
class OUParams:
    theta: float  # mean reversion speed
    mu: float     # long-run mean
    sigma: float  # vol

def estimate_ou_params(r: np.ndarray, dt: float=1.0) -> OUParams:
    # AR(1) fit r_{t+1} = a + b r_t + eps
    rt, rlag = r[1:], r[:-1]
    X = np.vstack([np.ones_like(rlag), rlag]).T
    b_hat = np.linalg.lstsq(X, rt, rcond=None)[0]
    a, b = b_hat
    theta = -np.log(max(1e-6, b)) / dt if b < 1 else 1e-6
    mu = a / (1 - b) if abs(1-b) > 1e-6 else 0.0
    # residual std
    eps = rt - (a + b*rlag)
    sigma = np.std(eps) * np.sqrt(2*theta / (1 - np.exp(-2*theta*dt) + 1e-9))
    return OUParams(theta=theta, mu=mu, sigma=float(abs(sigma)))

def simulate_ou(r0, params: OUParams, steps, dt, n_paths=1000, seed=7):
    rng = np.random.default_rng(seed)
    r = np.zeros((n_paths, steps+1))
    r[:,0] = r0
    for t in range(steps):
        mean = r[:,t] + params.theta*(params.mu - r[:,t])*dt
        std = params.sigma * np.sqrt(dt)
        r[:,t+1] = mean + std * rng.standard_normal(n_paths)
    return r

# ---------- Schrödinger (log-price PDF evolution, free particle) ----------
def schrodinger_pdf(logS0, sigma, T, N_x=1024, span=5.0, seed=0):
    """
    Evolve a Gaussian wave packet as proxy for log-price uncertainty.
    Returns grid S and probability density p(S).
    """
    np.random.seed(seed)
    x_min = logS0 - span*sigma*np.sqrt(T+1e-9)
    x_max = logS0 + span*sigma*np.sqrt(T+1e-9)
    x = np.linspace(x_min, x_max, N_x)
    dx = x[1]-x[0]
    # initial Gaussian centered at logS0
    a = 0.05
    psi0 = (a/np.sqrt(np.pi))**0.5 * np.exp(-0.5*((x-logS0)/a)**2)
    # free evolution ~ Gaussian remains Gaussian; approximate stationary here
    p_x = np.abs(psi0)**2
    p_x /= np.sum(p_x)*dx
    S = np.exp(x)
    return S, p_x

# ---------- Boltzmann / Fokker-Planck (discrete) ----------
def fokker_planck_pdf(r0, mu, sigma, steps, dr=0.0005, width=0.05, dt=1.0):
    """
    Discretize return space and evolve p(r,t+dt) = p + drift/diffusion terms.
    Simple explicit scheme; small dt recommended.
    """
    r_grid = np.arange(-width, width+dr, dr)
    p = np.exp(-0.5*((r_grid - r0)/ (sigma*np.sqrt(dt)+1e-9))**2)
    p /= np.trapz(p, r_grid)
    for _ in range(steps):
        dp = -mu*np.gradient(p, dr) + 0.5*(sigma**2)*np.gradient(np.gradient(p, dr), dr)
        p = p + dt*dp
        p[p<0] = 0
        p /= np.trapz(p, r_grid)
    return r_grid, p
