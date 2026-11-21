"""Smoke + timing probe for TimesFM on the P7 panel. Prints shapes and seconds per 512 series. No returns,
no Sharpe, nothing that reveals performance (the v3 prereg is not frozen yet)."""
import time

import pandas as pd

from trend import data
from trend.v3.tsfm import TimesFMForecaster, forecast_panel

R = pd.read_parquet(data.PROC / "returns.parquet")
fc = TimesFMForecaster()
dates = R.index[-50::5]
t0 = time.time()
horizons = {"short": 21, "medium": 63, "long": 252}
out = forecast_panel(R, dates, fc, horizons)
dt = time.time() - t0
n = len(out) // len(horizons)
print(f"series {n}, seconds {dt:.1f}, s/512 series {dt / n * 512:.2f}, cols {list(out.columns)}")
n_dates = len(R.index[::5])
print(f"projected full run: {n_dates * R.shape[1] / n * dt / 60:.1f} min")
