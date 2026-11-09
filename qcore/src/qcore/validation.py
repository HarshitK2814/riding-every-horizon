"""Overfitting-aware performance statistics.

All Sharpe ratios here are PER PERIOD (not annualised).
References: Bailey & Lopez de Prado (2012) PSR, (2014) DSR; Bailey, Borwein, Lopez de Prado & Zhu (2017)
PBO via CSCV; Lopez de Prado (2018) ch. 7 and 12 (purged k-fold, CPCV).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329


def _clean(r) -> np.ndarray:
    x = np.asarray(r, dtype=float)
    return x[~np.isnan(x)]


def probabilistic_sharpe(r, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), adjusting for skew and kurtosis."""
    x = _clean(r)
    n, sr = len(x), x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less trials with Sharpe variance sr_var."""
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
                                    + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r, n_trials: int, sr_var: float) -> float:
    """PSR against the Sharpe the best of n_trials pure-noise trials would show."""
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, sr_var))


def _sr_cols(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0) / X.std(axis=0, ddof=1)


def pbo_cscv(M: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of backtest overfitting. M: T x N returns, one column per trial configuration."""
    X = M.to_numpy(dtype=float)
    N = X.shape[1]
    blocks = np.array_split(np.arange(X.shape[0]), n_blocks)
    logits = []
    for is_set in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in is_set])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_set])
        best = int(np.argmax(_sr_cols(X[is_idx])))
        rank = stats.rankdata(_sr_cols(X[oos_idx]))[best] / (N + 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits, "n_combinations": len(logits)}


def group_index(index: pd.Index, n_groups: int) -> list[pd.Index]:
    return [index[g] for g in np.array_split(np.arange(len(index)), n_groups)]


def cpcv_splits(n: int, n_groups: int = 6, k_test: int = 2, purge: int = 0, embargo: int = 0):
    """Combinatorial purged CV. purge: obs before each test group dropped from train (label overlap);
    embargo: obs after each test group dropped from train (serial correlation)."""
    groups = np.array_split(np.arange(n), n_groups)
    for test_groups in combinations(range(n_groups), k_test):
        test = np.concatenate([groups[g] for g in test_groups])
        keep = np.ones(n, dtype=bool)
        for g in test_groups:
            lo, hi = groups[g][0], groups[g][-1]
            keep[max(0, lo - purge): min(n, hi + 1 + embargo)] = False
        yield np.flatnonzero(keep), test, test_groups


def cpcv_paths(oos: dict, n_groups: int, k_test: int, group_index: list) -> list[pd.Series]:
    """Stitch out-of-sample pieces into C(n_groups-1, k_test-1) full backtest paths."""
    n_paths = comb(n_groups - 1, k_test - 1)
    by_group = {g: [s for s in sorted(oos) if g in s] for g in range(n_groups)}
    return [pd.concat([oos[by_group[g][j]].loc[group_index[g]] for g in range(n_groups)])
            for j in range(n_paths)]


def spa_pvalue(bench: pd.Series, models: pd.DataFrame, block: int = 10, reps: int = 2000,
               seed: int = 0) -> dict:
    """Hansen (2005) SPA: does ANY model beat the benchmark? Inputs are returns (higher is better)."""
    from arch.bootstrap import SPA
    spa = SPA(-bench.to_numpy(), -models.to_numpy(), block_size=block, reps=reps, seed=seed)
    spa.compute()
    return {k: float(spa.pvalues[k]) for k in ("lower", "consistent", "upper")}


def romano_wolf(bench: pd.Series, models: pd.DataFrame, size: float = 0.05, block: int = 10,
                reps: int = 2000, seed: int = 0) -> list[str]:
    """Romano-Wolf StepM: names of models that beat the benchmark with FWER control at `size`."""
    from arch.bootstrap import StepM
    s = StepM(-bench.to_numpy(), -models, size=size, block_size=block, reps=reps, seed=seed)
    s.compute()
    return [str(m) for m in s.superior_models]
