"""Walk-forward optimisation. Parameters are chosen on each training window by the plateau-smoothed
Sharpe (average of each grid cell and its neighbours), then frozen for the next test window.
Only test-window returns enter `oos`."""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from qcore.metrics import sharpe
from qengine.costs import CostModel
from qengine.engine import run_weights


def wf_folds(index: pd.DatetimeIndex, first_test_year: int, test_years: int = 2, last_year: int | None = None,
             anchored: bool = True, train_years: int = 8) -> list:
    last_year = last_year or index[-1].year
    folds, y = [], first_test_year
    while y <= last_year:
        te_end = min(y + test_years - 1, last_year)
        tr_start = index[0] if anchored else pd.Timestamp(f"{y - train_years}-01-01")
        folds.append((tr_start, pd.Timestamp(f"{y - 1}-12-31"), pd.Timestamp(f"{y}-01-01"),
                      pd.Timestamp(f"{te_end}-12-31")))
        y += test_years
    return folds


def plateau_select(scores: np.ndarray) -> tuple[int, ...]:
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    s = np.where(finite, s, (s[finite].min() - 1) if finite.any() else 0.0)
    smooth = uniform_filter(s, size=3, mode="nearest")   # ties -> first cell in C order
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(smooth)), smooth.shape))


def walk_forward(make_weights: Callable[[pd.DataFrame, dict], pd.DataFrame], grid: dict, R: pd.DataFrame,
                 cost: CostModel, folds: list, lag: int = 1, periods: int = 252) -> dict:
    keys = list(grid)
    shape = tuple(len(grid[k]) for k in keys)
    pieces, rows = [], []
    for i, (tr0, tr1, te0, te1) in enumerate(folds):
        Rf = R.loc[:te1]
        nets, scores = {}, np.empty(shape)
        for pos in product(*[range(n) for n in shape]):
            params = {k: grid[k][j] for k, j in zip(keys, pos)}
            net = run_weights(make_weights(Rf, params), Rf, cost, lag=lag).net
            nets[pos] = net
            scores[pos] = sharpe(net.loc[tr0:tr1], periods)
        best = plateau_select(scores)
        oos = nets[best].loc[te0:te1]
        pieces.append(oos)
        rows.append({"fold": i, "params": {k: grid[k][j] for k, j in zip(keys, best)},
                     "train_sharpe": float(scores[best]), "test_sharpe": float(sharpe(oos, periods))})
    return {"oos": pd.concat(pieces), "chosen": pd.DataFrame(rows), "n_trials": int(np.prod(shape))}
