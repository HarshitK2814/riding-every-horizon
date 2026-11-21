"""Generate or extend the TimesFM signal cache (GPU). Idempotent: only dates not yet in the cache are forecast.
Usage: python scripts/v3/forecast_tsfm.py [--cpu]"""
import sys
from pathlib import Path

import pandas as pd

from trend import data
from trend.v3.tsfm import TimesFMForecaster, forecast_panel, schedule

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "v3" / "cache" / "tsfm_signal.parquet"
H = {"short": 21, "medium": 63, "long": 252}


def main() -> None:
    R = pd.read_parquet(data.PROC / "returns.parquet")
    dates = schedule(R.index, 5)
    old = pd.read_parquet(CACHE) if CACHE.exists() else None
    if old is not None:
        dates = dates[dates > old["date"].max()]
    if len(dates) == 0:
        print("cache up to date"); return
    fc = TimesFMForecaster()
    new = forecast_panel(R, dates, fc, H)
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    print(f"forecast dates added {len(dates)}, rows {len(out)}, last {out['date'].max().date()}")


if __name__ == "__main__":
    main()
