"""Roll-gap audit. Yahoo '=F' series splice contracts without back-adjustment, so on roll days the return
contains the calendar spread. We flag days where a future departs from a non-rolling proxy (cash index,
spot ETF) or, without a proxy, from its sector peers, by more than z robust sigmas (MAD, trailing)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def roll_gap_flags(fut: pd.Series, proxy: pd.Series, z: float = 5.0, window: int = 250) -> pd.Series:
    x = pd.concat([fut, proxy], axis=1, keys=["f", "p"]).dropna()
    beta = (x["f"].rolling(window, min_periods=60).cov(x["p"]) / x["p"].rolling(window, min_periods=60).var()).shift(1)
    resid = x["f"] - beta * x["p"]
    mad = resid.abs().rolling(window, min_periods=60).median().shift(1) * 1.4826
    flags = (resid.abs() > z * mad).fillna(False)
    return flags.reindex(fut.index, fill_value=False)


def impute_flagged(R: pd.DataFrame, flags: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    out = R.copy()
    for s in sector.unique():
        cols = [c for c in R.columns if sector.get(c) == s]
        for c in cols:
            others = [o for o in cols if o != c]
            if not others:
                continue
            med = R[others].median(axis=1)
            m = flags[c].reindex(R.index, fill_value=False)
            out.loc[m, c] = med[m]
    return out
