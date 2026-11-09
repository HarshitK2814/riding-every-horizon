"""Daily event loop with integer contracts and a compounding NAV: shows what rounding, small accounts
and per-contract notional do to a strategy that looks smooth in weight space.

Timing matches run_weights: target at decision t uses W[t]; contracts bought at the close of
t + (lag - 1) earn from the next day. Costs are charged when the trade happens."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd
import yaml

from qengine.costs import CostModel


def load_specs() -> pd.DataFrame:
    raw = yaml.safe_load(resources.files("qengine").joinpath("data/contracts.yaml").read_text())
    df = pd.DataFrame.from_dict(raw, orient="index")
    for col in ("multiplier", "divisor", "face", "adv_contracts"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["divisor"] = df["divisor"].fillna(1.0)
    return df


def notional_per_contract(prices: pd.DataFrame, R: pd.DataFrame, specs: pd.DataFrame,
                          fx_usd: pd.DataFrame | None = None) -> pd.DataFrame:
    """USD notional of one contract at each close. Priced tickers: price / divisor * multiplier.
    Synthetic tickers: face * cumulative level of R (level 1.0 on the first day).
    fx_usd: USD per unit of currency, columns = currency codes; missing -> 1.0 (USD)."""
    out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    for t in R.columns:
        s = specs.loc[t]
        if pd.notna(s["multiplier"]) and t in prices:
            n = prices[t].reindex(R.index).ffill() / s["divisor"] * s["multiplier"]
        else:
            n = s["face"] * (1 + R[t].fillna(0.0)).cumprod()
        ccy = s["currency"]
        if ccy != "USD" and fx_usd is not None and ccy in fx_usd:
            n = n * fx_usd[ccy].reindex(R.index).ffill()
        out[t] = n
    return out


@dataclass
class ContractResult:
    nav: pd.Series
    returns: pd.Series
    contracts: pd.DataFrame
    costs: pd.Series
    n_trades: int


def run_contracts(W: pd.DataFrame, R: pd.DataFrame, notional: pd.DataFrame, aum0: float, cost: CostModel,
                  lag: int = 1) -> ContractResult:
    cols = list(W.columns)
    Wv = W.reindex(index=R.index, columns=cols).fillna(0.0).shift(lag - 1).fillna(0.0).to_numpy()
    Rv = R.reindex(columns=cols).fillna(0.0).to_numpy()
    Nv = notional.reindex(index=R.index, columns=cols).ffill().to_numpy()
    c = cost.one_way_bp.reindex(cols).fillna(0.0).to_numpy() / 1e4
    rolls = (cost.rolls_per_year.reindex(cols).fillna(0.0).to_numpy() / cost.trading_days
             if cost.rolls_per_year is not None else np.zeros(len(cols)))
    T = len(R)
    nav = np.empty(T); ret = np.zeros(T); tc = np.zeros(T)
    held = np.zeros(len(cols)); n_hist = np.zeros((T, len(cols))); trades = 0
    v, prev_notional = aum0, np.nan_to_num(Nv[0])
    for t in range(T):
        pnl = float((held * prev_notional * Rv[t]).sum()) if t > 0 else 0.0
        notl = np.nan_to_num(Nv[t])
        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.where(notl > 0, np.round(Wv[t] * (v + pnl) / notl), 0.0)
        traded = np.abs(target - held) * notl
        cost_t = float((traded * c).sum() + (np.abs(held) * notl * 2 * c * rolls).sum())
        trades += int((target != held).sum())
        new_v = v + pnl - cost_t
        ret[t] = (new_v - v) / v
        v, held, prev_notional = new_v, target, notl
        nav[t], tc[t], n_hist[t] = v, cost_t / max(v, 1e-12), held
    idx = R.index
    return ContractResult(nav=pd.Series(nav, idx), returns=pd.Series(ret, idx),
                          contracts=pd.DataFrame(n_hist, idx, cols), costs=pd.Series(tc, idx), n_trades=trades)
