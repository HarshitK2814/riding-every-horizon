"""Vectorised target-weight backtest.

W[t] = target weights (fraction of NAV) decided with data up to the close of t. Held weights
W[t] earn R[t+lag]. A trade made at decision t is charged on day t+lag (same day it starts earning),
which matches the v1 projects at lag=1. R are excess returns; no financing of notional is modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qengine.costs import CostModel


@dataclass
class BacktestResult:
    gross: pd.Series
    net: pd.Series
    costs: pd.DataFrame
    weights: pd.DataFrame
    held: pd.DataFrame
    turnover: pd.Series

    @property
    def n_rebalances(self) -> int:
        return int((self.turnover > 0).sum())


def run_weights(W: pd.DataFrame, R: pd.DataFrame, cost: CostModel, lag: int = 1, aum: float | None = None,
                adv_usd: pd.DataFrame | None = None, sigma_daily: pd.DataFrame | None = None) -> BacktestResult:
    if lag < 1:
        raise ValueError("lag must be >= 1: a weight cannot earn the return of the bar it was computed on")
    W = W.reindex(index=R.index).fillna(0.0)
    Rx = R.reindex(columns=W.columns).fillna(0.0)
    held = W.shift(lag).fillna(0.0)
    gross = (held * Rx).sum(axis=1)
    trades = W.diff().abs().fillna(W.abs())
    parts = pd.DataFrame({
        "linear": cost.linear(trades).sum(axis=1),
        "roll": cost.roll(W).sum(axis=1),
        "impact": cost.impact(trades, aum, adv_usd, sigma_daily).sum(axis=1),
    }).shift(lag).fillna(0.0)
    net = gross - parts.sum(axis=1)
    return BacktestResult(gross=gross, net=net, costs=parts, weights=W, held=held,
                          turnover=trades.sum(axis=1).shift(lag).fillna(0.0))
