"""v3 books: turn cached signal panels into weights and net returns with the v1 machinery."""
from __future__ import annotations

import pandas as pd


def signal_frame(long: pd.DataFrame, index: pd.DatetimeIndex, max_hold: int = 5) -> pd.DataFrame:
    wide = long.groupby(["date", "inst"])["sig"].mean().unstack("inst")
    return wide.reindex(index).ffill(limit=max_hold - 1)
