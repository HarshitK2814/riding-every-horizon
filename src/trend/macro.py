"""Carry-correct excess returns for bonds and currencies, rebuilt from free FRED data.

Yahoo's continuous front-month futures jump at every roll by the calendar spread. For bond and
currency futures that spread *is* the carry, so the unadjusted series silently deletes it (EXP-003).
Instead we use the textbook constructions:

* Bonds: a constant-maturity par Treasury bought at the close of t-1 and repriced at t,
  r_t = P(coupon=y_{t-1}, maturity=M - dt, yield=y_t)/100 - 1 + y_{t-1} dt, minus the 3-month
  T-bill for one day, i.e. the excess return a rolled bond future earns (ignoring roll-down).
* Currencies: a 1-month forward held by a USD investor earns
  r_t = spot return (USD per FX) + (i_foreign - i_US) dt (covered interest parity).
  OECD monthly 3-month interbank rates are lagged one month to respect publication dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW, fred

DT = 1 / 252
BONDS = {"UST2": ("DGS2", 2), "UST5": ("DGS5", 5), "UST10": ("DGS10", 10), "UST20": ("DGS20", 20)}
# name: (FRED spot series, invert?, FRED 3m rate series, optional backfill rate series)
FX = {
    "EUR": ("DEXUSEU", False, "IR3TIB01EZM156N", None),
    "JPY": ("DEXJPUS", True, "IR3TIB01JPM156N", "IRSTCI01JPM156N"),
    "GBP": ("DEXUSUK", False, "IR3TIB01GBM156N", None),
    "AUD": ("DEXUSAL", False, "IR3TIB01AUM156N", None),
    "CAD": ("DEXCAUS", True, "IR3TIB01CAM156N", None),
    "CHF": ("DEXSZUS", True, "IR3TIB01CHM156N", None),
    "NZD": ("DEXUSNZ", False, "IR3TIB01NZM156N", None),
    "MXN": ("DEXMXUS", True, "IR3TIB01MXM156N", None),
    "SEK": ("DEXSDUS", True, "IR3TIB01SEM156N", None),
    "NOK": ("DEXNOUS", True, "IR3TIB01NOM156N", None),
    "DKK": ("DEXDNUS", True, "IR3TIB01DKM156N", None),
}
US_RATE = "IR3TIB01USM156N"


def download_macro() -> None:
    ids = {"DGS3MO"} | {v[0] for v in BONDS.values()} | {US_RATE}
    for v in FX.values():
        ids |= {v[0], v[2]} | ({v[3]} if v[3] else set())
    pd.DataFrame({i: fred(i) for i in sorted(ids)}).to_parquet(RAW / "fred_macro.parquet")


def _fred() -> pd.DataFrame:
    return pd.read_parquet(RAW / "fred_macro.parquet")


def par_price(coupon: np.ndarray, maturity: float, y: np.ndarray) -> np.ndarray:
    """Price per 100 of a semiannual bond (ignores accrued-interest timing within a period)."""
    n = 2 * maturity
    c, yy = coupon / 2, y / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(yy) > 1e-10, (1 - (1 + yy) ** -n) / yy, n)
    return 100 * (c * annuity + (1 + yy) ** -n)


def bond_excess_returns() -> pd.DataFrame:
    f = _fred()
    bdays = f["DGS10"].dropna().index
    rf = (f["DGS3MO"].reindex(bdays).ffill() / 100).shift(1)
    out = {}
    for name, (sid, m) in BONDS.items():
        y = (f[sid].reindex(bdays).ffill() / 100)
        y0 = y.shift(1).to_numpy()
        p1 = par_price(y0, m - DT, y.to_numpy())
        r = pd.Series(p1 / 100 - 1 + y0 * DT, index=bdays) - rf * DT
        out[name] = r.where(y.notna() & y.shift(1).notna())
    return pd.DataFrame(out).iloc[1:]


def monthly_rate_daily(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Monthly average rate (percent) for month m becomes usable on the first day of m+1."""
    s = series.dropna() / 100
    s.index = s.index + pd.offsets.MonthBegin(1)
    return s.reindex(index.union(s.index)).ffill().reindex(index)


def fx_spot() -> pd.DataFrame:
    f = _fred()
    out = {}
    for name, (sid, inv, *_ ) in FX.items():
        s = f[sid].dropna()
        out[name] = 1 / s if inv else s
    return pd.DataFrame(out)            # USD per unit of foreign currency


def fx_rates(index: pd.DatetimeIndex) -> pd.DataFrame:
    f = _fred()
    out = {"USD": monthly_rate_daily(f[US_RATE], index)}
    for name, (_, _, rid, back) in FX.items():
        r = f[rid].dropna()
        if back:
            b = f[back].dropna()
            r = pd.concat([b[b.index < r.index.min()], r])
        out[name] = monthly_rate_daily(r, index)
    return pd.DataFrame(out)


def fx_excess_returns() -> pd.DataFrame:
    S = fx_spot()
    idx = S.index
    rates = fx_rates(idx)
    spot_ret = S.ffill(limit=5).pct_change(fill_method=None)
    carry = rates[list(FX)].sub(rates["USD"], axis=0) * DT
    return (spot_ret + carry.shift(1)).where(spot_ret.notna()).iloc[1:]
