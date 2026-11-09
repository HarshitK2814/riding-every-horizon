"""P7 v2 Task 6 (H3): crash events and Hawkes intensity. Event at t uses sigma up to t-1; intensity at the first
trading day of a month uses events strictly before that day; parameters refit each January on past events only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qcore.hawkes import fit_hawkes_exp, hawkes_intensity


def crash_events(r: pd.Series, k: float = 2.5, com: int = 60) -> pd.Series:
    sd = r.ewm(com=com, min_periods=20).std().shift(1)
    return (r < -k * sd).fillna(False)


def monthly_intensity(events: pd.Series, min_events: int = 20) -> pd.Series:
    pos = np.arange(len(events), dtype=float)
    t_ev = pos[events.to_numpy()]
    firsts = events.groupby([events.index.year, events.index.month]).head(1).index
    out = pd.Series(np.nan, index=firsts)
    params, fitted_year = None, None
    for d in firsts:
        i = events.index.get_loc(d)
        if (d.year != fitted_year and d.month == 1) or params is None:
            past = t_ev[t_ev < i]
            if len(past) >= min_events:
                params, fitted_year = fit_hawkes_exp(past, float(i)), d.year
        if params is not None:
            out[d] = hawkes_intensity(t_ev[t_ev < i], params, np.array([float(i)]))[0]
    return out
