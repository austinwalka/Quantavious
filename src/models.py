# src/models.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# GBM simulation
def gbm_simulation(S0, mu, sigma, T, dt, n_paths):
    N = int(T / dt)
    t = np.linspace(0, T, N+1)
    dW = np.random.normal(scale=np.sqrt(dt), size=(n_paths, N))
    W = np.cumsum(dW, axis=1)
    exponent = (mu - 0.5 * sigma**2) * t[1:] + sigma * W
    paths = np.empty((n_paths, N+1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(exponent)
    return paths


def black_scholes_paths(S0, r, sigma, T, n_paths=1000):
    dt = 1/252
    N = int(T / dt)
    return gbm_simulation(S0, r, sigma, T, dt, n_paths)


# Quantum-inspired Schrödinger model (simplified)
import numpy.fft as fft
from scipy.fftpack import fft as sp_fft, ifft as sp_ifft

class Schrodinger:
    def __init__(self, x, psi_x0, V_x, k0=None, hbar=1, m=1, t0=0):
        self.x = x
        self.psi_x = psi_x0.astype(np.complex128)
        self.V_x = V_x
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.dk = 2 * np.pi / (self.N * self.dx)
        self.k0 = k0 if k0 is not None else -0.5 * self.N * self.dk
        self.k = self.k0 + self.dk * np.arange(self.N)

    def time_step(self, dt, steps=1):
        x_evolve_half = np.exp(-0.5j * self.V_x / self.hbar * dt)
        x_evolve = x_evolve_half ** 2
        k_evolve = np.exp(-0.5j * self.hbar * self.k**2 / self.m * dt)
        for _ in range(steps):
            self.psi_x *= x_evolve_half
            psi_k = sp_fft(self.psi_x)
            psi_k *= k_evolve
            self.psi_x = sp_ifft(psi_k)
            self.psi_x *= x_evolve
        self.psi_x *= x_evolve_half
        self.t += dt * steps

def gauss_x(x, a, x0, k0):
    return ((a / np.pi)**0.25) * np.exp(-0.5 * ((x - x0) / a) ** 2 + 1j * k0 * x)


def quantum_predict_distribution(S0, mu, sigma, horizon_days):
    """
    Run simplified Schrodinger evolution to forecast price distribution.
    Returns dict with 'mean', 'p5', 'p95'.
    """
    import numpy as np
    x_min = np.log(S0) - 5 * sigma * np.sqrt(horizon_days / 252)
    x_max = np.log(S0) + 5 * sigma * np.sqrt(horizon_days / 252)
    N_x = 1024
    x = np.linspace(x_min, x_max, N_x)
    a = 0.05
    k0 = (mu - sigma**2 / 2) / sigma
    psi0 = gauss_x(x, a, np.log(S0), k0)
    V_x = np.zeros_like(x)
    hbar = 1.0
    m = 1.0 / (2 * sigma ** 2)
    se = Schrodinger(x, psi0, V_x, hbar=hbar, m=m)
    dt = (horizon_days / 252) / 1000
    se.time_step(dt, steps=1000)
    p_x = np.abs(se.psi_x) ** 2
    dx = x[1] - x[0]
    p_x /= p_x.sum() * dx
    S_grid = np.exp(x)
    mean_price = (S_grid * p_x).sum() * dx
    cdf = np.cumsum(p_x) * dx
    p5 = np.exp(np.interp(0.05, cdf, x))
    p95 = np.exp(np.interp(0.95, cdf, x))
    return {"mean": mean_price, "p5": p5, "p95": p95}


# LSTM Model and training
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def lstm_train_predict(series, forecast_days=15, lookback=60, epochs=30, lr=0.001, device='cpu'):
    """
    Train LSTM on series and forecast future points.
    Returns predicted full path, mean forecast, 5th and 95th percentiles.
    """
    import torch
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    series = series.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, lookback)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    preds = []
    # Predict test
    X_test_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        out_test = model(X_test_t).cpu().numpy()
    # Predict forecast_days iteratively starting from last train+test
    recent_seq = scaled[-lookback:].reshape(1, lookback, 1)
    future_preds = []
    for _ in range(forecast_days):
        seq_t = torch.from_numpy(recent_seq).float().to(device)
        with torch.no_grad():
            pred = model(seq_t).cpu().numpy()[0, 0]
        future_preds.append(pred)
        recent_seq = np.roll(recent_seq, -1)
        recent_seq[0, -1, 0] = pred

    future_preds = np.array(future_preds).reshape(-1, 1)
    inv_preds = scaler.inverse_transform(future_preds).flatten()

    # Compute confidence interval using residuals std dev on test set
    residuals = y_test.flatten() - out_test.flatten()
    resid_std = np.std(residuals)

    pred_mean = inv_preds
    pred_p5 = inv_preds - 1.64 * resid_std * (inv_preds / inv_preds[0])
    pred_p95 = inv_preds + 1.64 * resid_std * (inv_preds / inv_preds[0])

    return inv_preds.flatten(), pred_mean, pred_p5, pred_p95
