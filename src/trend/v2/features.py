"""P7 v2 Task 7 features (exploratory E1/E2). Features at t use data up to the close of t; label y uses t+1..t+21."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tstat(lr: pd.DataFrame, h: int, sd: pd.DataFrame) -> pd.DataFrame:
    return lr.rolling(h, min_periods=h).sum() / (sd * np.sqrt(h))


def feature_panel(R: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    lr = np.log1p(R)
    sd = lr.ewm(com=60, min_periods=60).std()
    feats = {f"t{h}": _tstat(lr, h, sd) for h in (21, 63, 252)}
    feats["vratio"] = lr.ewm(com=20, min_periods=20).std() / lr.ewm(com=250, min_periods=60).std()
    fwd = lr[::-1].rolling(21, min_periods=21).sum()[::-1].shift(-1)
    feats["y"] = fwd / (sd * np.sqrt(21))
    panel = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    panel.index.names = ["date", "ticker"]
    dates = R.index[::step]
    return panel.loc[panel.index.get_level_values(0).isin(dates)]


def aci_delayed(y: pd.Series, yhat: pd.Series, alpha: float = 0.1, gamma: float = 0.005, window: int = 250,
                delay: int = 5) -> pd.DataFrame:
    """Adaptive conformal intervals (Gibbs & Candes 2021) for an overlapping-horizon label. The label of row s is only
    known `delay` rows later (21-day label on a weekly panel -> delay 5), so the interval at row t uses scores of rows
    <= t - delay and the alpha update uses the error of row t - delay. qcore.conformal.aci_intervals (delay 0) would
    leak the unrealised overlap."""
    yv, fv = y.to_numpy(dtype=float), yhat.reindex(y.index).to_numpy(dtype=float)
    score, n = np.abs(yv - fv), len(yv)
    lo, hi, cov, at = (np.full(n, np.nan) for _ in range(4))
    a_t = alpha
    for t in range(window + delay, n):
        s = t - delay
        if np.isfinite(cov[s]):
            a_t = a_t + gamma * (alpha - (1 - cov[s]))
        past = score[s - window + 1:s + 1]
        q = np.inf if a_t <= 0 else (0.0 if a_t >= 1 else float(np.quantile(past, 1 - a_t)))
        lo[t], hi[t], at[t] = fv[t] - q, fv[t] + q, a_t
        cov[t] = float(lo[t] <= yv[t] <= hi[t]) if np.isfinite(yv[t]) else np.nan
    return pd.DataFrame({"lo": lo, "hi": hi, "covered": cov, "alpha_t": at}, index=y.index)
