"""Transaction-cost model. All costs are fractions of NAV.

- linear: one-way cost (half spread + commission + fixed slippage) in bp of traded notional.
- roll: futures roll = two one-way trades per roll on the held notional, spread evenly over the year.
- impact: square-root law (Almgren et al. 2005; Toth et al. 2011): cost per $ traded =
  coef * daily_vol * sqrt(traded $ / ADV $). Off when impact_coef == 0 or aum is None.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    one_way_bp: pd.Series
    rolls_per_year: pd.Series | None = None
    impact_coef: float = 0.0
    trading_days: int = 252

    @classmethod
    def from_instruments(cls, inst: pd.DataFrame, impact_coef: float = 0.0) -> "CostModel":
        return cls(one_way_bp=inst["cost_bp"].astype(float), rolls_per_year=inst["rolls"].astype(float),
                   impact_coef=impact_coef)

    def scaled(self, factor: float) -> "CostModel":
        return replace(self, one_way_bp=self.one_way_bp * factor)

    def _c(self, cols) -> np.ndarray:
        return self.one_way_bp.reindex(cols).fillna(0.0).to_numpy() / 1e4

    def linear(self, turnover: pd.DataFrame) -> pd.DataFrame:
        return turnover * self._c(turnover.columns)

    def roll(self, W: pd.DataFrame) -> pd.DataFrame:
        if self.rolls_per_year is None:
            return W * 0.0
        r = self.rolls_per_year.reindex(W.columns).fillna(0.0).to_numpy() / self.trading_days
        return W.abs() * 2 * self._c(W.columns) * r

    def impact(self, turnover: pd.DataFrame, aum: float | None, adv_usd: pd.DataFrame | None,
               sigma_daily: pd.DataFrame | None) -> pd.DataFrame:
        if self.impact_coef == 0 or aum is None or adv_usd is None or sigma_daily is None:
            return turnover * 0.0
        adv = adv_usd.reindex(index=turnover.index, columns=turnover.columns)
        sig = sigma_daily.reindex(index=turnover.index, columns=turnover.columns)
        part = (turnover * aum / adv).clip(lower=0)
        return (self.impact_coef * sig * np.sqrt(part) * turnover).fillna(0.0)
