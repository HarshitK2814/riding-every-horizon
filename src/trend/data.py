"""Free data: Yahoo continuous front-month futures, fund NAVs, and the FRED 3-month T-bill.

Every series is stored as downloaded in data/raw; cleaning happens in `clean_returns`, which logs
each value it changes so the cleaning can be audited.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def load_universe() -> dict:
    return yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text())


def instrument_table() -> pd.DataFrame:
    u = load_universe()["sectors"]
    rows = [{"ticker": t, "sector": s, "cost_bp": c, "rolls": v["rolls"]}
            for s, v in u.items() for t, c in v["tickers"].items()]
    return pd.DataFrame(rows).set_index("ticker")


def fred(series_id: str) -> pd.Series:
    txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=60).text
    df = pd.read_csv(io.StringIO(txt))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna().rename(series_id)


def download() -> None:
    import yfinance as yf

    RAW.mkdir(parents=True, exist_ok=True)
    u = load_universe()
    tickers = [t for t, row in instrument_table().iterrows() if row["sector"] not in SYNTHETIC] + u["benchmarks"]
    px = yf.download(tickers, start="1999-01-01", auto_adjust=True, progress=False)
    px["Close"].to_parquet(RAW / "close.parquet")
    px["Volume"].to_parquet(RAW / "volume.parquet")
    fred("DTB3").to_frame().to_parquet(RAW / "dtb3.parquet")
    from .macro import download_macro
    download_macro()


def tbill_daily() -> pd.Series:
    """Daily T-bill return from the FRED 3-month discount yield (percent), carried forward."""
    y = pd.read_parquet(RAW / "dtb3.parquet")["DTB3"] / 100
    return (y / 252).rename("rf")


COMMODITY_SECTORS = {"metals", "energy", "ags"}
SYNTHETIC = {"rates", "fx"}       # built from FRED in macro.py, not from Yahoo futures


def clean_returns(outlier_z: float = 8.0, peer_z: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily futures returns with data errors and roll gaps removed; genuine shocks are kept.

    z_t = r_t / (EWMA vol known at t-1). A return is zeroed, and logged with its reason, when:
      1. the previous price is not positive (crude oil, April 2020);
      2. it is a bad print: |z_t| > outlier_z and the next day reverses it with |z_t+1| > outlier_z
         (both days are zeroed; this uses t+1 only to clean the data, never in a signal);
      3. it is a commodity roll gap: |z_t| > outlier_z in a metals/energy/ags contract while no
         other contract in the same sector moved by more than `peer_z` that day.
    Financial-futures shocks (SNB 2015, Brexit, Feb 2018) and confirmed commodity shocks (oil,
    9 Mar 2020) are kept. Returns (returns, log of changes).
    """
    close = pd.read_parquet(RAW / "close.parquet")
    inst = instrument_table()
    close = close[[t for t in inst.index if t in close and inst.at[t, "sector"] not in SYNTHETIC]]
    raw, Z = {}, {}
    for t in close:
        p = close[t].dropna()
        r = p.pct_change().iloc[1:]
        nonpos = p.shift(1).reindex(r.index) <= 0
        vol = r.where(~nonpos).ewm(com=60, min_periods=60).std().shift(1)
        raw[t], Z[t] = r.where(~nonpos), (r / vol).where(~nonpos)
        raw[t].attrs["nonpos"] = r[nonpos]
    R = pd.DataFrame(raw).sort_index()
    Z = pd.DataFrame(Z).reindex(R.index)
    big = Z.abs() > outlier_z
    nxt = Z.shift(-1)
    bounce = big & (nxt.abs() > outlier_z) & (np.sign(Z) != np.sign(nxt))
    bounce = bounce | bounce.shift(1, fill_value=False)
    sector = inst["sector"].reindex(R.columns)
    moved = Z.abs() > peer_z
    peers = moved.T.groupby(sector).transform("sum").T - moved.astype(int)
    is_comm = sector.isin(COMMODITY_SECTORS).to_numpy()
    roll = big & (peers == 0) & is_comm
    log = []
    for t in R:
        for d, v in raw[t].attrs["nonpos"].items():
            log.append({"ticker": t, "date": d, "raw_return": v, "z": np.nan, "reason": "non-positive price"})
        for mask, why in ((bounce, "bad print (reverses next day)"), (roll & ~bounce, "commodity roll gap")):
            for d in R.index[mask[t]]:
                log.append({"ticker": t, "date": d, "raw_return": R.at[d, t], "z": Z.at[d, t], "reason": why})
    R = R.mask(bounce | roll, 0.0)
    for t in R:   # non-positive-price days were NaN above; set to 0 within the live range
        R.loc[raw[t].attrs["nonpos"].index, t] = 0.0
    # Calendar: union of trading days; a market closed on a given day contributes 0.
    return R, pd.DataFrame(log).sort_values(["ticker", "date"])


def build() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    from .macro import bond_excess_returns, fx_excess_returns

    R, log = clean_returns()
    syn = pd.concat([bond_excess_returns(), fx_excess_returns()], axis=1)
    inst = instrument_table()
    R = pd.concat([R, syn[[t for t in inst.index if t in syn]]], axis=1).sort_index()
    R = R[R.index.isin(pd.read_parquet(RAW / "close.parquet").index)]   # one calendar: US futures days
    R.to_parquet(PROC / "returns.parquet")
    log.to_csv(PROC / "cleaning_log.csv", index=False)
    close = pd.read_parquet(RAW / "close.parquet")
    bench = load_universe()["benchmarks"]
    close[bench].to_parquet(PROC / "benchmarks.parquet")
