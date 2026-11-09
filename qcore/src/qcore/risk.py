"""Risk-model checks: VaR backtests (Kupiec 1995, Christoffersen 1998) and a GARCH vol forecast
that only ever uses data up to the forecast date."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits: pd.Series, alpha: float) -> dict:
    """Proportion-of-failures test. hits: True on days the loss exceeded VaR; alpha: expected breach rate."""
    h = np.asarray(hits, dtype=bool)
    n, x = len(h), int(h.sum())
    pi = x / n
    lr = -2 * (_xlogy(n - x, 1 - alpha) + _xlogy(x, alpha) - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    return {"lr": float(lr), "p": float(stats.chi2.sf(lr, 1)), "n": n, "breaches": x}


def christoffersen_ind(hits: pd.Series) -> dict:
    """Independence test: is a breach more likely the day after a breach?"""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return {"lr": float(lr), "p": float(stats.chi2.sf(lr, 1))}


def garch_vol_forecast(r: pd.Series, refit_every: int = 21, min_obs: int = 750, o: int = 1,
                       dist: str = "t") -> pd.Series:
    """GJR-GARCH(1,o,1) one-step-ahead daily vol. Value at t = forecast for t+1, made at close t.
    Parameters are refit every `refit_every` days on data strictly before the block start."""
    from arch import arch_model

    y = r.dropna() * 100
    out = pd.Series(np.nan, index=y.index, dtype=float)
    n = len(y)
    for start in range(min_obs, n, refit_every):
        end = min(start + refit_every, n)
        am = arch_model(y.iloc[:end], mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
        res = am.fit(last_obs=y.index[start], disp="off")
        f = res.forecast(horizon=1, start=y.index[start - 1], reindex=False)
        rows = y.index[start - 1:end - 1] if end < n else y.index[start - 1:]
        out.loc[rows] = np.sqrt(f.variance.loc[rows, "h.1"].to_numpy()) / 100
    return out.reindex(r.index)
