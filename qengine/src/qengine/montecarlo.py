"""Stationary block bootstrap of daily returns (Politis & Romano 1994): distribution of Sharpe, CAGR and
max drawdown a strategy with these returns could have shown. Keeps volatility clustering within blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qcore.bootstrap import stationary_indices


def _maxdd(x: np.ndarray) -> float:
    eq = np.cumprod(1 + x)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def mc_summary(r: pd.Series, periods: int = 252, n_paths: int = 10000, mean_block: int = 20,
               seed: int = 0) -> pd.DataFrame:
    x = r.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    sr, cagr, mdd = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
    for i in range(n_paths):
        p = x[stationary_indices(len(x), mean_block, rng)]
        sr[i] = p.mean() / p.std(ddof=1) * np.sqrt(periods)
        cagr[i] = np.prod(1 + p) ** (periods / len(p)) - 1
        mdd[i] = _maxdd(p)
    q = lambda a: [np.quantile(a, 0.05), np.quantile(a, 0.5), np.quantile(a, 0.95)]
    out = pd.DataFrame([q(sr), q(cagr), q(mdd)], index=["sharpe", "cagr", "max_dd"], columns=["p05", "p50", "p95"])
    out.loc["prob_loss"] = [np.nan, float((cagr < 0).mean()), np.nan]
    return out
