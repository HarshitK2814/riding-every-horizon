"""Perturbation test for look-ahead: change only data AFTER a cut date; no weight at or before the cut
may change. Catches .shift(-k), centred windows, full-sample normalisation and future-fitted params."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class LookaheadError(AssertionError):
    pass


def check_lookahead(make_weights: Callable[[pd.DataFrame], pd.DataFrame], R: pd.DataFrame,
                    cut: int | None = None, seed: int = 0, tol: float = 0.0) -> None:
    cut = int(len(R) * 0.7) if cut is None else cut
    rng = np.random.default_rng(seed)
    shocked = R.copy()
    n_after, k = len(R) - cut - 1, R.shape[1]
    shocked.iloc[cut + 1:] = (shocked.iloc[cut + 1:].to_numpy() * rng.uniform(0.5, 3.0, (n_after, k))
                              + rng.normal(0, 0.02, (n_after, k)))
    end = R.index[cut]
    a, b = make_weights(R).loc[:end], make_weights(shocked).loc[:end]
    diff = ((a - b).abs() > tol) & ~(a.isna() & b.isna())
    if diff.to_numpy().any():
        row = diff.any(axis=1)
        when = row[row].index[0]
        col = diff.loc[when][diff.loc[when]].index[0]
        raise LookaheadError(f"weight {col} on {when.date()} changed when only data after {end.date()} changed")
