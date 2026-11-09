"""P7 v2 Task 3 (D1): execution realism and capacity. Main table 2001-2017; seen-period table 2018-2026.
Signal at close t; v1 trades at that close (lag 1); v2 also reports lag 2 (next close).
Yahoo FX volume for EUR/JPY/GBP/AUD/CAD/MXN is a data defect: the contracts.yaml ADV is forced for them.
Run: python scripts/v2/execution.py"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.splits import Splits, select
from qengine.capacity import capacity_at, capacity_curve
from qengine.contracts import run_contracts
from qengine.costs import CostModel
from qengine.engine import run_weights
from trend import data, macro, model
from trend.v2.execution import FORCE_FALLBACK, market_inputs

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG = ROOT / "results" / "v2" / "tables", ROOT / "results" / "v2" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
PREREG = ROOT / "research" / "preregistration_v2.md"
os.environ.setdefault("PREREG_HASH", "2d7828df5093")
AUMS = [1e7, 1e8, 5e8, 1e9, 5e9]
COEFS = [0.25, 0.5, 1.0]
BOOT = 1000


def sharpe_d(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def sharpe_ci(r: pd.Series) -> tuple[float, float, float]:
    lo, hi = percentile_ci(bootstrap_stat(r, sharpe_d, n_boot=BOOT, mean_block=21))
    return sharpe_d(r), lo, hi


def load():
    R = pd.read_parquet(data.PROC / "returns.parquet")
    inst = data.instrument_table()
    cfg_d = yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text())
    close = pd.read_parquet(data.RAW / "close.parquet").reindex(R.index).ffill()
    volume = pd.read_parquet(data.RAW / "volume.parquet").reindex(R.index)
    fx = macro.fx_spot().reindex(R.index).ffill()
    return R, inst, cfg_d, close, volume, fx


def row(name, net, res, mi, aum, sl, extra=None):
    r = sl(net)
    s, lo, hi = sharpe_ci(r)
    out = {"run": name, "net_sharpe": s, "ci95_lo": lo, "ci95_hi": hi, "n_days": len(r)}
    c = sl(res.costs)
    out.update({"ann_cost_linear": c["linear"].mean() * 252, "ann_cost_roll": c["roll"].mean() * 252,
                "ann_cost_impact": c["impact"].mean() * 252,
                "turnover_per_yr": sl(res.turnover).mean() * 252,
                "n_rebalance_days": int((sl(res.turnover) > 0).sum())})
    if aum:
        tr = res.weights.diff().abs()
        part = (tr * aum / mi["adv_usd"].reindex_like(tr)).where(tr > 0)
        out["max_participation"] = float(sl(part).max().max())
    if extra:
        out.update(extra)
    return out


def run_period(label, sl, Rp, inst, W, mi, base_cost, ymax):
    rows = []
    r1 = run_weights(W, Rp, base_cost, lag=1)
    rows.append(row("v1 costs, lag1", r1.net, r1, mi, None, sl))
    r2 = run_weights(W, Rp, base_cost, lag=2)
    rows.append(row("v1 costs, lag2", r2.net, r2, mi, None, sl))
    for coef in COEFS:
        cm = CostModel.from_instruments(inst, impact_coef=coef)
        for aum in AUMS:
            rr = run_weights(W, Rp, cm, lag=2, aum=aum, adv_usd=mi["adv_usd"], sigma_daily=mi["sigma_daily"])
            rows.append(row(f"+ impact coef {coef} at AUM {aum:.0e}", rr.net, rr, mi, aum, sl,
                            {"impact_coef": coef, "aum": aum}))
    contracts_info = {}
    Wc, Rc = W.loc[:ymax], Rp.loc[:ymax]
    Nc = mi["notional"].loc[:ymax]
    for aum0 in (1e7, 5e7):
        cr = run_contracts(Wc, Rc, Nc, aum0, base_cost, lag=2)
        r = sl(cr.returns)
        s, lo, hi = sharpe_ci(r)
        n = cr.contracts.abs()
        tgt_on = Wc.shift(1).abs() > 1e-9
        small = {}
        for c in n.columns:
            on = sl(n[c][tgt_on[c]])
            if len(on):
                small[c] = {"median_abs_contracts": float(on.median()), "share_days_le1": float((on <= 1).mean())}
        tr = cr.contracts.diff().abs() * Nc.reindex_like(cr.contracts)
        part = (tr / mi["adv_usd"].loc[:ymax].reindex_like(tr)).where(tr > 0)
        trade_days = sl(cr.contracts.diff().abs().sum(axis=1))
        rows.append({"run": f"contracts ${aum0/1e6:.0f}m, lag2 (roll cost incl.)", "net_sharpe": s, "ci95_lo": lo,
                     "ci95_hi": hi, "n_days": len(r), "ann_cost_impact": 0.0,
                     "ann_cost_total": sl(cr.costs).mean() * 252, "n_rebalance_days": int((trade_days > 0).sum()),
                     "max_participation": float(sl(part).max().max())})
        contracts_info[f"{aum0:.0e}"] = small
    t = pd.DataFrame(rows)
    t.insert(0, "period", label)
    return t, contracts_info


def main():
    R, inst, cfg_d, close, volume, fx = load()
    cfg = model.TrendConfig.from_yaml(cfg_d)
    v1 = model.backtest(R, inst, cfg)
    W = v1["weights"]
    Rp = model.prepare(R).reindex(columns=W.columns).fillna(0.0)
    start = max(v1["net"].index[0], pd.Timestamp("2001-01-01"))
    mi_primary = market_inputs(Rp, close, volume, fx)
    # Yahoo volume is also implausibly low for GC/SI/HG/PL/PA (and spiky for ZL/ZM). Rule fixed once, on 2001-2017 ADV
    # only: instruments with median Yahoo ADV < USD 50m get NO impact charge. This is a lower bound; "primary" is the
    # upper bound (it trusts Yahoo volume everywhere except the six FX contracts and the two cash indices).
    med = mi_primary["adv_usd"].loc[start:"2017-12-31"].median()
    thin = sorted(med[med < 50e6].index)
    mi_lo = {**mi_primary, "adv_usd": mi_primary["adv_usd"].copy()}
    mi_lo["adv_usd"][thin] = np.nan
    variants = {"primary_yahoo_adv": mi_primary, "low_adv_instruments_impact_off": mi_lo}
    mi = mi_primary
    base_cost = CostModel.from_instruments(inst, 0.0)
    splits = Splits.from_dict(yaml.safe_load((ROOT / "configs" / "splits_v2.yaml").read_text()))
    dv = lambda x: x.loc[start:"2017-12-31"]
    seen = lambda x: pd.concat([select(x, splits, "test", PREREG), select(x, splits, "recent", PREREG)])

    # parity: the engine at lag 1 must reproduce the v1 backtest exactly
    chk = run_weights(W, Rp, base_cost, lag=1).net.loc[v1["net"].index]
    assert np.allclose(chk.to_numpy(), v1["net"].to_numpy(), atol=1e-12), "engine lag1 != v1 net"

    mains, seens = [], []
    for vn, m_ in variants.items():
        t, cinfo = run_period("2001-2017", dv, Rp, inst, W, m_, base_cost, "2017-12-31")
        mains.append(t.assign(adv_variant=vn))
        t2, cinfo_seen = run_period("2018-2026 - seen period (v1 reported it)", seen, Rp, inst, W, m_, base_cost,
                                    "2026-12-31")
        seens.append(t2.assign(adv_variant=vn))
    main_all, seen_t = pd.concat(mains, ignore_index=True), pd.concat(seens, ignore_index=True)
    main_t = main_all[main_all["adv_variant"] == "primary_yahoo_adv"].reset_index(drop=True)
    assert np.isclose(main_t.iloc[0]["net_sharpe"], sharpe_d(dv(v1["net"])))
    main_all.to_csv(TAB / "execution.csv", index=False)
    seen_t.to_csv(TAB / "execution_seen_period.csv", index=False)
    json.dump({"2001-2017": cinfo, "2018-2026 seen": cinfo_seen}, open(TAB / "execution_contract_counts.json", "w"), indent=1)

    for vn in variants:
        for coef in COEFS:
            q = main_all[(main_all["impact_coef"] == coef) & (main_all["adv_variant"] == vn)]
            s = q.sort_values("aum")["net_sharpe"].to_numpy()
            assert (np.diff(s) <= 1e-9).all(), f"non-monotone in AUM for {vn} coef {coef}: {s}"

    caps = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = [1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 2.5e9, 5e9]
    sl_idx = slice(start, "2017-12-31")
    for vn, m_ in variants.items():
        for coef in COEFS:
            cm = CostModel.from_instruments(inst, impact_coef=coef)
            cur = capacity_curve(W.loc[sl_idx], Rp.loc[sl_idx], cm, m_["adv_usd"].loc[sl_idx],
                                 m_["sigma_daily"].loc[sl_idx], grid, lag=2)
            cur["impact_coef"] = coef
            cap = capacity_at(cur, 0.10)
            caps.append(cur.assign(capacity_aum_drop0p10=cap, adv_variant=vn))
            ax.plot(cur["aum"], cur["net_sharpe"], marker="o", ls="-" if vn.startswith("primary") else "--",
                    label=f"{'Yahoo ADV' if vn.startswith('primary') else 'thin-ADV impact off'}, coef {coef} (cap ${cap/1e6:,.0f}m)")
    ax.set_xscale("log")
    ax.set_xlabel("AUM (USD)")
    ax.set_ylabel("net Sharpe, lag 2")
    ax.set_title("P7 v1 blend capacity, 2001-2017 (square-root impact)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "capacity.png", dpi=130)
    cap_df = pd.concat(caps)
    cap_df.to_csv(TAB / "capacity.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["run", "net_sharpe", "ci95_lo", "ci95_hi", "ann_cost_linear", "ann_cost_roll", "ann_cost_impact",
            "ann_cost_total", "turnover_per_yr", "n_rebalance_days", "max_participation"]
    print(main_all[["adv_variant"] + [c for c in show if c in main_all]].round(4).to_string())
    print(cap_df.round(3).to_string())
    print("\nSEEN")
    print(seen_t[["adv_variant"] + [c for c in show if c in seen_t]].round(4).to_string())
    for k, v in cinfo.items():
        low = {c: x for c, x in v.items() if x["median_abs_contracts"] <= 1}
        print(f"contracts {k}: instruments with median |contracts|<=1 when target on ({len(low)}/{len(v)}):", sorted(low))
    nan_adv = mi["adv_usd"].loc[sl_idx].isna().all()
    print("instruments with NO adv (impact=0):", list(nan_adv[nan_adv].index))
    tr = W.diff().abs().loc[sl_idx]
    miss = mi["adv_usd"].reindex_like(tr).isna() | (mi["adv_usd"].reindex_like(tr) <= 0)
    share = float((tr.where(miss, 0.0)).sum().sum() / tr.sum().sum())
    print(f"share of traded weight with no usable ADV (impact charged 0 there): {share:.3%}")
    json.dump({"share_traded_weight_without_adv": share, "no_adv_instruments": list(nan_adv[nan_adv].index),
               "forced_fallback": sorted(FORCE_FALLBACK)}, open(TAB / "execution_adv_coverage.json", "w"), indent=1)
    print("thin-ADV instruments (median ADV < $50m, impact off in lower-bound variant):", thin)
    json.dump({"thin_adv_instruments": thin}, open(TAB / "execution_thin_adv.json", "w"))
    print("forced fallback:", sorted(FORCE_FALLBACK))


if __name__ == "__main__":
    main()
