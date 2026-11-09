"""Shared book construction for the forward tracker and the seen-period re-run.
Signals are computed at the close of day t from data up to t; lag 2 trades at the next close."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from trend import data, macro, model
from trend.v2.riskmodel import backtest_with_vol, garch_vol_panel


def load_inputs(root: Path) -> dict:
    R = pd.read_parquet(data.PROC / "returns.parquet")
    inst = data.instrument_table()
    cfg_d = yaml.safe_load((root / "configs" / "strategy.yaml").read_text())
    close = pd.read_parquet(data.RAW / "close.parquet").reindex(R.index).ffill()
    volume = pd.read_parquet(data.RAW / "volume.parquet").reindex(R.index)
    fx = macro.fx_spot().reindex(R.index).ffill()
    return dict(R=R, inst=inst, cfg_d=cfg_d, close=close, volume=volume, fx=fx)


def garch_full(Rp: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Causal GARCH panel on all data; cache is reused only if it already covers the last date."""
    if cache.exists():
        g = pd.read_parquet(cache)
        if g.index.equals(Rp.index):
            return g
    g = garch_vol_panel(Rp, refit_every=63, min_obs=750)
    cache.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(cache)
    return g


def build_weights(inp: dict, garch: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    cfg = model.TrendConfig.from_yaml(inp["cfg_d"])
    v1 = model.backtest(inp["R"], inp["inst"], cfg)
    gb = backtest_with_vol(inp["R"], inp["inst"], cfg, garch)
    Rp = model.prepare(inp["R"]).reindex(columns=v1["weights"].columns).fillna(0.0)
    return {"v1_blend": v1["weights"], "garch_blend": gb["weights"]}, Rp
