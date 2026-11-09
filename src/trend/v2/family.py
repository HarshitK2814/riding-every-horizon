"""The config family v1 could plausibly have chosen from; used for PBO and deflated Sharpe."""

from __future__ import annotations

from itertools import product

import pandas as pd

from trend import model

SIGNALS, BUFFERS, USES, COMS = ("sign", "tanh"), (0.0, 0.05, 0.10, 0.20), ("blend", "short", "medium", "long"), (30, 60, 120)


def family_configs() -> list[tuple[str, dict]]:
    out = []
    for s, b, u, c in product(SIGNALS, BUFFERS, USES, COMS):
        kw = {"signal": s, "buffer": b, "vol_com": c, "use": None if u == "blend" else (u,)}
        out.append((f"{s}|b{b:.2f}|{u}|com{c}", kw))
    return out


def family_returns(R: pd.DataFrame, inst: pd.DataFrame, base_cfg: dict, period_end: str = "2017-12-31") -> pd.DataFrame:
    R = R.loc[:period_end]
    cols = {}
    for name, kw in family_configs():
        cfg = model.TrendConfig.from_yaml(base_cfg)
        for k, v in kw.items():
            setattr(cfg, k, v)
        cols[name] = model.backtest(R, inst, cfg)["net"]
    return pd.DataFrame(cols).dropna()
