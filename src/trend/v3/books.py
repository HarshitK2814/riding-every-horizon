"""v3 books: turn cached signal panels into weights and net returns with the v1 machinery."""
from __future__ import annotations

import pandas as pd


def signal_frame(long: pd.DataFrame, index: pd.DatetimeIndex, max_hold: int = 5) -> pd.DataFrame:
    wide = long.groupby(["date", "inst"])["sig"].mean().unstack("inst")
    return wide.reindex(index).ffill(limit=max_hold - 1)


import numpy as np

from trend import model
from trend.v3.tsfm import schedule


def _v1_signal(R: pd.DataFrame, cfg: model.TrendConfig) -> pd.DataFrame:
    names = tuple(cfg.horizons)
    return sum(model.horizon_signal(R, cfg.horizons[n], cfg.vol_com) for n in names) / len(names)


def weekly_signal(R: pd.DataFrame, cfg: model.TrendConfig, every: int = 5) -> pd.DataFrame:
    """v1 blend signal refreshed only on the TimesFM forecast schedule (every `every` days) and held between."""
    s1 = _v1_signal(R, cfg)
    s = s1.copy()
    s.loc[~s1.index.isin(schedule(R.index, every))] = np.nan
    return s.ffill(limit=every - 1)


def build_v3_weights(inp: dict, tsfm_long: pd.DataFrame | None, clean_full: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    cfg = model.TrendConfig.from_yaml(inp["cfg_d"])
    R = model.prepare(inp["R"])
    out = {"v1_blend": model.backtest(R, inp["inst"], cfg)["weights"]}
    out["v1_weekly"] = model.backtest(R, inp["inst"], cfg, sig=weekly_signal(R, cfg))["weights"]
    if tsfm_long is not None:
        out["tsfm_blend"] = model.backtest(R, inp["inst"], cfg, sig=signal_frame(tsfm_long, R.index))["weights"]
    if clean_full is not None:
        out["clean_blend"] = model.backtest(R, inp["inst"], cfg, sig=clean_full.reindex(R.index))["weights"]
    return out
