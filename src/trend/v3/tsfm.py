"""TimesFM wrapper and a point-in-time panel forecaster.
Context = cumulative log-return index of one instrument up to and including close t (holiday zeros kept).
mu_h = forecast index at t+h minus index at t; width_h = q90 - q10 at t+h. Signal = tanh(mu / sigma_hat),
sigma_hat = width / 2.5631 (q90 - q10 of a normal). Known at the close of t."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

Z_80 = 2.5631  # q90 - q10 of a standard normal


class Forecaster(Protocol):
    def forecast(self, contexts: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]: ...


class TimesFMForecaster:
    MODEL_ID = "google/timesfm-2.5-200m-pytorch"

    def __init__(self, max_context: int = 1024, max_horizon: int = 256, per_core_batch_size: int = 256,
                 device: str | None = None):
        """device None = auto (GPU if available); "cpu" forces CPU (set on the model before compile)."""
        import timesfm
        import torch
        torch.set_float32_matmul_precision("high")
        self.max_horizon = max_horizon
        self.per_core_batch_size = per_core_batch_size
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.MODEL_ID)
        if device == "cpu":
            # The inner model chose its device at construction. Make device and device_count consistent BEFORE
            # compile, which computes global_batch_size = per_core_batch_size * model.device_count.
            self.m.model.to(torch.device("cpu"))
            self.m.model.device = torch.device("cpu")
            self.m.model.device_count = 1
        elif device is not None:
            raise ValueError(f"device must be None or 'cpu', got {device!r}")
        self.m.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon, per_core_batch_size=per_core_batch_size,
            normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=False, fix_quantile_crossing=True))

    def forecast(self, contexts, horizon):
        point, quant = self.m.forecast(horizon=horizon, inputs=[np.asarray(c, dtype=np.float32) for c in contexts])
        return np.asarray(point), np.asarray(quant)


def tstat_signal(mu: np.ndarray, width: np.ndarray) -> np.ndarray:
    return np.tanh(mu / (width / Z_80))


def schedule(index: pd.DatetimeIndex, every: int = 5) -> pd.DatetimeIndex:
    return index[::every]


def extend_schedule(index: pd.DatetimeIndex, last_cached, every: int = 5) -> pd.DatetimeIndex:
    """New forecast dates after the last cached date: positions +every, +2*every, ... in `index` after it, so the
    spacing stays exactly `every` trading days whatever the panel's first date is."""
    after = index[index > last_cached]
    return after[every - 1::every]


def forecast_panel(R: pd.DataFrame, dates: pd.DatetimeIndex, fc: Forecaster, horizons: dict[str, int],
                   context: int = 1024, min_obs: int = 256, batch: int = 512) -> pd.DataFrame:
    lr = np.log1p(R)
    started = R.notna().cumsum() > 0
    idx_level = lr.fillna(0.0).where(started).cumsum()
    hmax = max(horizons.values())
    cap = getattr(fc, "max_horizon", None)
    if cap is not None and hmax > cap:
        raise ValueError(f"max horizon {hmax} exceeds forecaster capacity {cap}")
    jobs = []                                   # (date, inst, context array)
    for d in dates:
        hist = idx_level.loc[:d]                # point in time: nothing after close d
        for inst in R.columns:
            c = hist[inst].dropna().to_numpy()
            if len(c) >= min_obs:
                jobs.append((d, inst, c[-context:]))
    rows = []
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        point, quant = fc.forecast([c for _, _, c in chunk], hmax)
        point, quant = np.asarray(point), np.asarray(quant)
        if point.shape != (len(chunk), hmax) or quant.shape != (len(chunk), hmax, 10):
            raise ValueError(f"forecaster output shape mismatch: point {point.shape} (expected {(len(chunk), hmax)}), "
                             f"quant {quant.shape} (expected {(len(chunk), hmax, 10)}; channel 0 = mean, 1..9 = q10..q90)")
        for j, (d, inst, c) in enumerate(chunk):
            for name, h in horizons.items():
                mu = float(point[j, h - 1] - c[-1])
                width = float(quant[j, h - 1, 9] - quant[j, h - 1, 1])
                rows.append((d, inst, name, mu, width))
    out = pd.DataFrame(rows, columns=["date", "inst", "horizon", "mu", "width"])
    out["sig"] = tstat_signal(out["mu"].to_numpy(), out["width"].clip(lower=1e-8).to_numpy())
    return out.reset_index(drop=True)
