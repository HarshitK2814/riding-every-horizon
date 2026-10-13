"""Multi-horizon trend model: signals, risk budgeting, vol targeting, costs.

Timing convention (checked by tests/test_lookahead.py): every quantity indexed t uses returns up to
and including the close of day t. Weights decided at the close of t earn the return of day t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class TrendConfig:
    horizons: dict = field(default_factory=lambda: {"short": 21, "medium": 63, "long": 252})
    vol_com: int = 60
    instrument_vol: float = 0.40
    target_vol: float = 0.10
    cov_halflife: int = 126
    max_leverage: float = 6.0
    lag: int = 1
    buffer: float = 0.10              # trade an instrument only if it is >10% of a unit position off target
    use: tuple | None = None          # subset of horizon names; None = all (the blend)
    sectors: tuple | None = None      # subset of sectors; None = all six
    sector_weights: dict | None = None  # risk budget per sector; None = equal (1/6 each)

    @classmethod
    def from_yaml(cls, d: dict, **kw) -> "TrendConfig":
        keep = {k: d[k] for k in ("horizons", "vol_com", "instrument_vol", "target_vol",
                                  "cov_halflife", "max_leverage", "lag", "buffer") if k in d}
        return cls(**keep, **kw)


def prepare(R: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill holidays after each instrument's first observation; NaN before it."""
    started = R.notna().cumsum() > 0
    return R.fillna(0.0).where(started)


def ewma_vol(R: pd.DataFrame, com: int) -> pd.DataFrame:
    return R.ewm(com=com, min_periods=com).std() * np.sqrt(TRADING_DAYS)


def horizon_signal(R: pd.DataFrame, h: int, com: int = 60) -> pd.DataFrame:
    """Trend strength in [-1, 1]: tanh of the t-statistic of the trailing h-day log return,
    tanh(sum_h log r / (sigma_daily * sqrt(h))). Known at the close of t. A plain sign() flips on
    noise when the trend is near zero; the t-statistic version trades ~4x less (EXP-002)."""
    lr = np.log1p(R)
    cum = lr.rolling(h, min_periods=h).sum()
    sd = lr.ewm(com=com, min_periods=com).std()
    return np.tanh(cum / (sd * np.sqrt(h)))


def raw_positions(R: pd.DataFrame, sector: pd.Series, cfg: TrendConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Notional weights before the portfolio vol target: blend of horizon signals, each scaled to
    `instrument_vol`, then equal risk across sectors and equal within a sector.
    Returns (weights, unit) where unit is the weight a full-strength signal (|s| = 1) would get."""
    vol = ewma_vol(R, cfg.vol_com)
    names = cfg.use or tuple(cfg.horizons)
    sig = sum(horizon_signal(R, cfg.horizons[n], cfg.vol_com) for n in names) / len(names)
    unit = cfg.instrument_vol / vol
    if cfg.sectors is not None:
        keep = sector.reindex(unit.columns).isin(cfg.sectors)
        unit, sig = unit.loc[:, keep], sig.loc[:, keep]
    unit = unit.where(sig.notna())
    sec = sector.reindex(unit.columns)
    live = unit.notna()
    n_in_sector = live.T.groupby(sec).transform("sum").T
    n_sectors = live.T.groupby(sec).any().T.sum(axis=1)
    if cfg.sector_weights is None:
        unit = (unit / n_in_sector.replace(0, np.nan)).div(n_sectors.replace(0, np.nan), axis=0)
    else:
        sw = sec.map(cfg.sector_weights).astype(float)
        live_w = (live.T.groupby(sec).any().T * pd.Series(cfg.sector_weights)).sum(axis=1)
        unit = (unit / n_in_sector.replace(0, np.nan)).mul(sw, axis=1).div(live_w.replace(0, np.nan), axis=0)
    return (sig * unit).fillna(0.0), unit.fillna(0.0)


def ewma_cov_scale(R: pd.DataFrame, W: pd.DataFrame, cfg: TrendConfig) -> pd.Series:
    """Scalar k_t so that k_t * W_t has ex-ante vol `target_vol`, using an EWMA covariance of
    returns up to t. Capped so gross leverage stays below `max_leverage`."""
    X = R.reindex(columns=W.columns).fillna(0.0).to_numpy()
    Wn = W.to_numpy()
    lam = 0.5 ** (1 / cfg.cov_halflife)
    S = np.zeros((X.shape[1], X.shape[1]))
    k = np.full(len(X), np.nan)
    warm = 0
    for t in range(len(X)):
        x = X[t]
        S = lam * S + (1 - lam) * np.outer(x, x)
        warm += 1
        if warm < 126:
            continue
        v = Wn[t] @ S @ Wn[t] * TRADING_DAYS
        if v > 0:
            gross = np.abs(Wn[t]).sum()
            k[t] = min(cfg.target_vol / np.sqrt(v), cfg.max_leverage / gross if gross > 0 else np.inf)
    return pd.Series(k, index=W.index)


def apply_buffer(W: pd.DataFrame, unit: pd.DataFrame, width: float) -> pd.DataFrame:
    """No-trade band: keep yesterday's position unless the target moved more than `width` x the
    instrument's full-signal position size. Uses only information at t."""
    if width <= 0:
        return W
    T, U = W.to_numpy(), np.nan_to_num(unit.reindex_like(W).to_numpy())
    out = np.zeros_like(T)
    prev = np.zeros(T.shape[1])
    for t in range(len(T)):
        move = np.abs(T[t] - prev) > width * np.abs(U[t])
        prev = np.where(move | (U[t] == 0), T[t], prev)
        out[t] = prev
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def costs(W: pd.DataFrame, inst: pd.DataFrame) -> pd.Series:
    """Daily cost: one-way cost on turnover plus a roll cost (two trades per roll) on |position|."""
    c = inst["cost_bp"].reindex(W.columns).to_numpy() / 1e4
    rolls = inst["rolls"].reindex(W.columns).to_numpy() / TRADING_DAYS
    turnover = W.diff().abs().fillna(W.abs())
    return (turnover * c).sum(axis=1) + (W.abs() * 2 * c * rolls).sum(axis=1)


def backtest(R: pd.DataFrame, inst: pd.DataFrame, cfg: TrendConfig) -> dict:
    R = prepare(R)
    W0, unit = raw_positions(R, inst["sector"], cfg)
    k = ewma_cov_scale(R, W0, cfg)
    W = apply_buffer(W0.mul(k, axis=0).fillna(0.0), unit.mul(k, axis=0).fillna(0.0), cfg.buffer)
    Rx = R.reindex(columns=W.columns).fillna(0.0)
    held = W.shift(cfg.lag).fillna(0.0)               # decided at close t, earns day t+1
    gross = (held * Rx).sum(axis=1)
    cost = costs(W, inst).shift(cfg.lag).fillna(0.0)
    net = gross - cost
    started = k.notna().shift(cfg.lag, fill_value=False)
    return {"gross": gross[started], "net": net[started], "cost": cost[started],
            "weights": W, "contrib": (held * Rx)[started]}
