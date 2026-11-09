"""P7 v2 Task 1: roll-gap audit + ADV check. Uses 2001-2017 only for every number that informs a choice.
Run: python scripts/v2/audit.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qcore.bootstrap import bootstrap_stat, percentile_ci  # noqa: E402
from trend import data, model  # noqa: E402
from trend.v2.audit import impute_flagged, roll_gap_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v2" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
PROXIES = {"ES=F": "^GSPC", "NQ=F": "^NDX", "YM=F": "^DJI", "NKD=F": "^N225", "GC=F": "GLD", "SI=F": "SLV"}
QUARTERLY = {"equity", "rates", "fx"}          # expiry months 3,6,9,12
PERIOD = ("2001-01-01", "2017-12-31")
ADV_MAP = {"UST2": "ZT=F", "UST5": "ZF=F", "UST10": "ZN=F", "UST20": "ZB=F", "EUR": "6E=F", "JPY": "6J=F",
           "GBP": "6B=F", "AUD": "6A=F", "CAD": "6C=F", "CHF": "6S=F", "NZD": "6N=F", "MXN": "6M=F"}


def sharpe_d(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def sharpe_ci(r: pd.Series) -> tuple[float, float, float]:
    d = bootstrap_stat(r, sharpe_d, n_boot=1000, mean_block=21)
    lo, hi = percentile_ci(d)
    return sharpe_d(r), lo, hi


def proxies() -> pd.DataFrame:
    p = ROOT / "data" / "raw" / "proxies_v2.parquet"
    if p.exists():
        return pd.read_parquet(p)
    px = yf.download(sorted(set(PROXIES.values())), start="2000-01-01", end="2026-10-14",
                     auto_adjust=True, progress=False)["Close"]
    px.to_parquet(p)
    return px


def main() -> None:
    R = pd.read_parquet(data.PROC / "returns.parquet")
    inst = data.instrument_table()
    sector = inst["sector"]
    cfg_d = yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text())
    Rp = R.loc[PERIOD[0]:PERIOD[1]]
    pr = proxies().pct_change().reindex(Rp.index)
    flags = pd.DataFrame(False, index=Rp.index, columns=Rp.columns)
    for c in Rp.columns:
        if c in PROXIES and PROXIES[c] in pr:
            proxy = pr[PROXIES[c]]
        else:
            others = [o for o in Rp.columns if o != c and sector.get(o) == sector.get(c)]
            if not others:
                continue
            proxy = Rp[others].median(axis=1)
        flags[c] = roll_gap_flags(Rp[c], proxy)
    rows = []
    for c in Rp.columns:
        f = flags[c]
        idx = f[f].index
        share = np.nan
        if sector.get(c) in QUARTERLY and len(idx):
            share = float(((idx.month % 3 == 0) & (idx.day >= 8) & (idx.day <= 21)).mean())
        rows.append({"ticker": c, "sector": sector.get(c), "proxy": PROXIES.get(c, "sector median"),
                     "n_flags": int(f.sum()), "mean_abs_gap": float(Rp[c][f].abs().mean()) if f.any() else np.nan,
                     "share_in_expected_roll_window": share})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "roll_gaps.csv", index=False)

    base = model.TrendConfig.from_yaml(cfg_d)
    r0 = model.backtest(R, inst, base)["net"].loc[PERIOD[0]:PERIOD[1]]
    Rfix = R.copy()
    Rfix.loc[Rp.index] = impute_flagged(Rp, flags, sector)
    r1 = model.backtest(Rfix, inst, base)["net"].loc[PERIOD[0]:PERIOD[1]]
    s0, s1 = sharpe_ci(r0), sharpe_ci(r1)
    half = (s0[2] - s0[1]) / 2
    out = {"period": PERIOD, "v1": s0, "imputed": s1, "delta": s1[0] - s0[0], "ci_half_width_v1": half,
           "use_imputed_in_v2": bool(abs(s1[0] - s0[0]) > half), "n_days": int(len(r0)),
           "n_flags_total": int(flags.values.sum())}
    # ADV check (Yahoo front-month volume, last 250 days of the 20d-median series)
    vol = pd.read_parquet(ROOT / "data" / "raw" / "volume.parquet")
    spec = yaml.safe_load((Path(__import__("qengine").__file__).parent / "data" / "contracts.yaml").read_text())
    adv = []
    for syn, fut in ADV_MAP.items():
        y = float(vol[fut].rolling(20, min_periods=10).median().dropna().iloc[-250:].median())
        a = spec[syn]["adv_contracts"]
        adv.append({"synthetic": syn, "future": fut, "yahoo_adv_20d_median": y, "yaml_adv_contracts": a,
                    "ratio_yahoo_over_yaml": y / a, "off_by_gt_2x": bool(max(y / a, a / y) > 2)})
    pd.DataFrame(adv).to_csv(OUT / "adv_check.csv", index=False)
    pd.Series(out).to_json(OUT / "roll_audit_sharpe.json", indent=2)
    print(tab.to_string()); print(out); print(pd.DataFrame(adv).to_string())


if __name__ == "__main__":
    main()
