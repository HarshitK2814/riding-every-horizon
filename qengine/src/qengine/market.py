"""Market inputs for impact and capacity: contract notional, ADV in USD and daily vol, all known at t-1.
Shared by P7 and P8 (moved from P7 trend.v2.execution)."""

from __future__ import annotations

import pandas as pd

from qengine.contracts import load_specs, notional_per_contract

SYNTH_TO_FUT = {"UST2": "ZT=F", "UST5": "ZF=F", "UST10": "ZN=F", "UST20": "ZB=F",
                "EUR": "6E=F", "JPY": "6J=F", "GBP": "6B=F", "AUD": "6A=F", "CAD": "6C=F",
                "CHF": "6S=F", "NZD": "6N=F", "MXN": "6M=F"}
# Yahoo reports 100-3000x too little volume for these FX futures (research/data_audit_v2.md): a data
# defect, so the contracts.yaml ADV estimate is used for them even when Yahoo volume exists.
FORCE_FALLBACK = frozenset({"EUR", "JPY", "GBP", "AUD", "CAD", "MXN"})
# Yahoo volume of these cash indices is share volume, not futures contracts: ignored (no ADV, so no impact; understated).
CASH_INDEX_NO_VOLUME = frozenset({"^STOXX50E", "^FTSE"})


def market_inputs(R: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, fx_usd: pd.DataFrame | None,
                  force_fallback: frozenset | set = FORCE_FALLBACK) -> dict:
    specs = load_specs()
    notional = notional_per_contract(close, R, specs, fx_usd)
    if fx_usd is not None:      # synthetic FX: notional is face x spot (face x cumulated excess return would double count)
        for c in R.columns:
            s = specs.loc[c]
            if pd.isna(s["multiplier"]) and s["currency"] != "USD" and s["currency"] in fx_usd:
                notional[c] = s["face"] * fx_usd[s["currency"]].reindex(R.index).ffill()
    vol_cols = {c: SYNTH_TO_FUT.get(c, c) for c in R.columns}
    v = pd.DataFrame({c: volume[f] if f in volume else pd.Series(float("nan"), index=R.index)
                      for c, f in vol_cols.items()}).reindex(R.index)
    v = v.where(v > 0)                       # a zero print means "feed missing", not "nobody traded"
    for c in CASH_INDEX_NO_VOLUME & set(v.columns):
        v[c] = float("nan")
    for c in R.columns:
        fallback = specs["adv_contracts"].get(c)
        if pd.notna(fallback):
            if c in force_fallback:
                v[c] = float(fallback)
            else:
                v[c] = v[c].fillna(fallback)
    # last known 20-day median level is carried forward through gaps (no look-ahead; leading gaps stay NaN)
    adv = (v.rolling(20, min_periods=10).median().ffill() * notional).shift(1)
    sigma = R.ewm(com=60, min_periods=20).std().shift(1)
    return {"notional": notional, "adv_usd": adv, "sigma_daily": sigma}
