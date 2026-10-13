"""Stationary bootstrap (Politis & Romano 1994) for statistics of dependent returns."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of indices with geometric block lengths (mean `mean_block`), wrapping around."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_stat(data: pd.Series | pd.DataFrame, stat: Callable, n_boot: int = 2000,
                   mean_block: float = 6.0, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of `stat(resampled_data)`. Rows are resampled jointly, so a
    DataFrame of two strategies keeps their cross-correlation (needed for difference tests)."""
    rng = np.random.default_rng(seed)
    values = data.to_numpy()
    n = len(values)
    out = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[stationary_indices(n, mean_block, rng)]
        out[b] = stat(pd.DataFrame(sample, columns=data.columns) if values.ndim == 2 else pd.Series(sample))
    return out


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))
