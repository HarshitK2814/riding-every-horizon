"""Performance and risk metrics. Inputs are periodic simple returns (pd.Series)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def annualised_mean(r: pd.Series, periods: int) -> float:
    return float(r.mean() * periods)


def annualised_vol(r: pd.Series, periods: int) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, periods: int, rf: pd.Series | float = 0.0) -> float:
    ex = (r - rf).dropna()
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(periods))


def sharpe_se_lo(r: pd.Series, periods: int, lags: int = 0) -> float:
    """Standard error of the annualised Sharpe ratio (Lo 2002). lags=0 is the iid case;
    lags>0 applies Lo's serial-correlation adjustment to the annualisation factor."""
    r = r.dropna()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    if lags == 0:
        eta = np.sqrt(periods)
    else:
        rho = [r.autocorr(k) for k in range(1, lags + 1)]
        eta = periods / np.sqrt(periods + 2 * sum((periods - k) * rho[k - 1] for k in range(1, lags + 1)))
    return float(eta * np.sqrt((1 + 0.5 * sr**2) / n))


def sortino(r: pd.Series, periods: int, mar: float = 0.0) -> float:
    ex = r.dropna() - mar
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(periods))


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def var_cvar(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical VaR and CVaR as positive loss numbers."""
    x = r.dropna().to_numpy()
    q = np.quantile(x, 1 - level)
    return float(-q), float(-x[x <= q].mean())


def newey_west_tstat(r: pd.Series, lags: int) -> float:
    """t-statistic of the mean with a Bartlett-kernel HAC variance."""
    x = r.dropna().to_numpy()
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n))


def summary(r: pd.Series, periods: int, rf: pd.Series | float = 0.0, nw_lags: int = 6) -> pd.Series:
    """One-line performance summary; always reports N and a Sharpe confidence interval."""
    r = r.dropna()
    sr = sharpe(r, periods, rf)
    se = sharpe_se_lo((r - rf).dropna(), periods)
    var95, cvar95 = var_cvar(r, 0.95)
    return pd.Series({
        "n_periods": len(r),
        "start": r.index.min(),
        "end": r.index.max(),
        "ann_return": annualised_mean(r, periods),
        "ann_vol": annualised_vol(r, periods),
        "sharpe": sr,
        "sharpe_ci95_lo": sr - 1.96 * se,
        "sharpe_ci95_hi": sr + 1.96 * se,
        "sortino": sortino(r, periods),
        "max_drawdown": max_drawdown(r),
        "var95": var95,
        "cvar95": cvar95,
        "skew": float(stats.skew(r)),
        "nw_tstat_mean": newey_west_tstat(r, nw_lags),
    })
