"""GARCH vs EWMA risk scaling (H1). Every forecast at t uses returns up to and including the close of t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qcore.risk import garch_vol_forecast
from trend import model

TD = 252


def garch_vol_panel(R: pd.DataFrame, refit_every: int = 63, min_obs: int = 750) -> pd.DataFrame:
    ewma = R.ewm(com=60, min_periods=60).std() * np.sqrt(TD)
    out = {}
    for c in R.columns:
        g = garch_vol_forecast(R[c].dropna(), refit_every=refit_every, min_obs=min_obs) * np.sqrt(TD)
        out[c] = g.reindex(R.index).fillna(ewma[c])
    return pd.DataFrame(out).where(ewma.notna())


def garch_forecast_guarded(r: pd.Series, refit_every: int = 63, min_obs: int = 750,
                           max_persistence: float = 0.9999) -> pd.Series:
    """Same forecast as qcore.risk.garch_vol_forecast (GJR-GARCH(1,1,1)-t, refit on data strictly before each block) but a
    block is REJECTED (NaN) when the optimiser did not converge or alpha + beta + gamma/2 >= max_persistence (explosive or
    integrated fit). Uses only the fit result, never future data. Daily vol, forecast for t+1 made at close t."""
    import warnings

    from arch import arch_model

    y = r.dropna() * 100
    out = pd.Series(np.nan, index=y.index, dtype=float)
    n = len(y)
    for start in range(min_obs, n, refit_every):
        end = min(start + refit_every, n)
        am = arch_model(y.iloc[:end], mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(last_obs=y.index[start], disp="off")
        p = res.params
        persistence = float(p["alpha[1]"] + p["beta[1]"] + 0.5 * p["gamma[1]"])
        if res.convergence_flag != 0 or not persistence < max_persistence:
            continue
        f = res.forecast(horizon=1, start=y.index[start - 1], reindex=False)
        rows = y.index[start - 1:end - 1] if end < n else y.index[start - 1:]
        v = np.sqrt(f.variance.loc[rows, "h.1"].to_numpy()) / 100
        out.loc[rows] = np.where(np.isfinite(v), v, np.nan)
    return out.reindex(r.index)


def garch_vol_panel_guarded(R: pd.DataFrame, refit_every: int = 63, min_obs: int = 750,
                            max_persistence: float = 0.9999) -> pd.DataFrame:
    """Like garch_vol_panel, but rejected blocks (see garch_forecast_guarded) fall back to EWMA com 60."""
    ewma = R.ewm(com=60, min_periods=60).std() * np.sqrt(TD)
    out = {}
    for c in R.columns:
        g = garch_forecast_guarded(R[c].dropna(), refit_every, min_obs, max_persistence) * np.sqrt(TD)
        out[c] = g.reindex(R.index).fillna(ewma[c])
    return pd.DataFrame(out).where(ewma.notna())


def tracking_dev(R: pd.DataFrame, vol: pd.DataFrame, target: float = 0.40, window: int = 63) -> pd.DataFrame:
    """Daily deviation (realised 63-day vol of the unit-scaled return minus target). Unit-scaled return =
    R[t+1] * target / vol[t]."""
    scaled = R * (target / vol.reindex_like(R)).shift(1)
    return scaled.rolling(window).std() * np.sqrt(TD) - target


def tracking_rmse(R: pd.DataFrame, vol: pd.DataFrame, target: float = 0.40, window: int = 63,
                  start=None, end=None) -> pd.Series:
    dev = tracking_dev(R, vol, target, window).loc[start:end]
    return np.sqrt((dev ** 2).mean())


def backtest_with_vol(R: pd.DataFrame, inst: pd.DataFrame, cfg: model.TrendConfig, vol: pd.DataFrame) -> dict:
    """model.backtest with the instrument vol forecast replaced by `vol` (annualised, known at t)."""
    R = model.prepare(R)
    W0, unit = model.raw_positions(R, inst["sector"], cfg, vol=vol)
    k = model.ewma_cov_scale(R, W0, cfg)
    W = model.apply_buffer(W0.mul(k, axis=0).fillna(0.0), unit.mul(k, axis=0).fillna(0.0), cfg.buffer)
    Rx = R.reindex(columns=W.columns).fillna(0.0)
    held = W.shift(cfg.lag).fillna(0.0)
    gross = (held * Rx).sum(axis=1)
    cost = model.costs(W, inst).shift(cfg.lag).fillna(0.0)
    started = k.notna().shift(cfg.lag, fill_value=False)
    return {"gross": gross[started], "net": (gross - cost)[started], "cost": cost[started],
            "weights": W, "contrib": (held * Rx)[started]}


def exante_vol(R: pd.DataFrame, W: pd.DataFrame, halflife: int = 126, vols: pd.DataFrame | None = None,
               warm: int = 126) -> pd.Series:
    """Ex-ante DAILY portfolio vol of weights W[t] (decided at close t): sqrt(w' S_t w) with S_t the EWMA second-moment
    matrix of returns up to t. If `vols` (annualised, known at t) is given, S_t = D C_t D with D = vols/sqrt(252) and C_t the
    EWMA correlation (so the GARCH variance forecasts drive the scale and the EWMA supplies the correlations)."""
    X = R.reindex(columns=W.columns).fillna(0.0).to_numpy()
    Wn = W.reindex(R.index).fillna(0.0).to_numpy()
    V = None if vols is None else (vols.reindex(index=R.index, columns=W.columns).to_numpy() / np.sqrt(TD))
    lam = 0.5 ** (1 / halflife)
    S = np.zeros((X.shape[1], X.shape[1]))
    out = np.full(len(X), np.nan)
    for t in range(len(X)):
        S = lam * S + (1 - lam) * np.outer(X[t], X[t])
        if t + 1 < warm:
            continue
        if V is None:
            v = Wn[t] @ S @ Wn[t]
        else:
            d = np.sqrt(np.clip(np.diag(S), 1e-18, None))
            D = np.nan_to_num(V[t], nan=0.0)
            Cw = (Wn[t] * D)
            v = Cw @ (S / np.outer(d, d)) @ Cw
        out[t] = np.sqrt(v) if v > 0 else np.nan
    return pd.Series(out, index=R.index)
