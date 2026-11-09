"""Adaptive conformal inference (Gibbs & Candes 2021): prediction intervals whose coverage
self-corrects under distribution shift. Interval for t uses residuals up to t-1 only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aci_intervals(y: pd.Series, yhat: pd.Series, alpha: float = 0.1, gamma: float = 0.005,
                  window: int = 250) -> pd.DataFrame:
    yv, fv = y.to_numpy(dtype=float), yhat.reindex(y.index).to_numpy(dtype=float)
    score = np.abs(yv - fv)
    n = len(yv)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    cov, at = np.full(n, np.nan), np.full(n, np.nan)
    a_t = alpha
    for t in range(window, n):
        past = score[t - window:t]
        if a_t <= 0:
            q = np.inf
        elif a_t >= 1:
            q = 0.0
        else:
            q = float(np.quantile(past, 1 - a_t))
        lo[t], hi[t], at[t] = fv[t] - q, fv[t] + q, a_t
        err = float(not (lo[t] <= yv[t] <= hi[t]))
        cov[t] = 1 - err
        a_t = a_t + gamma * (alpha - err)
    return pd.DataFrame({"lo": lo, "hi": hi, "covered": cov, "alpha_t": at}, index=y.index)
