"""Univariate Hawkes process with exponential kernel: lambda(t) = mu + alpha * sum_{t_j < t} exp(-beta (t - t_j)).
Branching ratio alpha/beta = expected number of follow-on events per event (clustering strength).
Fallback for the `tick` library, which may not install on Windows."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def simulate_hawkes_exp(mu: float, a: float, b: float, T: float, seed: int = 0) -> np.ndarray:
    """Ogata thinning with the exponential-kernel recursion."""
    rng = np.random.default_rng(seed)
    t, S, ev = 0.0, 0.0, []
    while True:
        lam_bar = mu + a * S
        w = rng.exponential(1 / lam_bar)
        t += w
        if t > T:
            return np.asarray(ev)
        S *= np.exp(-b * w)
        if rng.uniform() * lam_bar <= mu + a * S:
            ev.append(t)
            S += 1.0


def _loglik(mu: float, a: float, b: float, t: np.ndarray, T: float) -> float:
    A = np.zeros(len(t))
    for i in range(1, len(t)):
        A[i] = np.exp(-b * (t[i] - t[i - 1])) * (1 + A[i - 1])
    return float(np.log(mu + a * A).sum() - mu * T - (a / b) * (1 - np.exp(-b * (T - t))).sum())


def fit_hawkes_exp(times: np.ndarray, T: float) -> dict:
    t = np.sort(np.asarray(times, dtype=float))

    def nll(x):
        mu, a, b = np.exp(x)
        if a >= b:
            return 1e12
        return -_loglik(mu, a, b, t, T)

    x0 = np.log([max(len(t) / T * 0.5, 1e-6), 0.5, 1.0])
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    mu, a, b = np.exp(res.x)
    return {"mu": float(mu), "alpha": float(a), "beta": float(b), "branching": float(a / b),
            "loglik": float(-res.fun)}


def hawkes_intensity(times: np.ndarray, params: dict, at: np.ndarray) -> np.ndarray:
    t = np.sort(np.asarray(times, dtype=float))
    out = np.empty(len(at))
    for i, s in enumerate(np.asarray(at, dtype=float)):
        past = t[t < s]
        out[i] = params["mu"] + params["alpha"] * np.exp(-params["beta"] * (s - past)).sum()
    return out
