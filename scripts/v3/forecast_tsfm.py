"""Generate or extend the TimesFM signal cache (GPU). Idempotent: only dates not yet in the cache are forecast.
Usage: python scripts/v3/forecast_tsfm.py [--cpu]   (--cpu forces TimesFM onto CPU; default is auto: GPU if available)"""
import sys
from pathlib import Path

import pandas as pd

from trend import data
from trend.v3.tsfm import TimesFMForecaster, extend_schedule, forecast_panel, schedule

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "v3" / "cache" / "tsfm_signal.parquet"
H = {"short": 21, "medium": 63, "long": 252}


def main(cpu: bool = False, base: Path | None = None, out_path: Path | None = None) -> None:
    """base: read-only starting cache (default: the reference cache); out_path: file written (default: the reference
    cache itself). forward.py passes out_path under results/v3/forward so the reference cache is never written."""
    base = base or CACHE
    out_path = out_path or CACHE
    R = pd.read_parquet(data.PROC / "returns.parquet")
    src = out_path if out_path.exists() else base
    old = pd.read_parquet(src) if src.exists() else None
    dates = schedule(R.index, 5) if old is None else extend_schedule(R.index, old["date"].max(), 5)
    if len(dates) == 0:
        print("cache up to date"); return
    fc = TimesFMForecaster(device="cpu" if cpu else None)
    new = forecast_panel(R, dates, fc, H)
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"forecast dates added {len(dates)}, rows {len(out)}, last {out['date'].max().date()}")


if __name__ == "__main__":
    main("--cpu" in sys.argv)
