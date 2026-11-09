"""Capacity: how net Sharpe decays as AUM grows, under square-root market impact."""

from __future__ import annotations

import pandas as pd

from qcore.metrics import sharpe
from qengine.costs import CostModel
from qengine.engine import run_weights


def capacity_curve(W: pd.DataFrame, R: pd.DataFrame, cost: CostModel, adv_usd: pd.DataFrame,
                   sigma_daily: pd.DataFrame, aums: list[float], lag: int = 2, periods: int = 252) -> pd.DataFrame:
    rows = []
    for aum in aums:
        res = run_weights(W, R, cost, lag=lag, aum=aum, adv_usd=adv_usd, sigma_daily=sigma_daily)
        trades = res.weights.diff().abs()
        part = (trades * aum / adv_usd.reindex_like(trades)).max().max()
        rows.append({"aum": aum, "net_sharpe": sharpe(res.net, periods),
                     "impact_cost_ann": float(res.costs["impact"].mean() * periods),
                     "linear_cost_ann": float(res.costs["linear"].mean() * periods),
                     "max_participation": float(part)})
    return pd.DataFrame(rows)


def capacity_at(curve: pd.DataFrame, max_sharpe_drop: float = 0.1) -> float:
    base = curve["net_sharpe"].iloc[0]
    ok = curve[curve["net_sharpe"] >= base - max_sharpe_drop]
    return float(ok["aum"].max())
