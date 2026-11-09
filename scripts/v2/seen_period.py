"""P7 v2 Task 8: seen-period re-run (2018-2026) and tearsheets. EVERY 2018-2026 number is 'seen period (v1 reported it)'.
Books: v1_blend, garch_blend. Execution: signal at close t, trade next close (lag 2); (a) v1 costs only, (b) square-root
impact coef 0.5 at AUM $100m with Yahoo ADV (primary) and with thin-ADV impact off (lower bound), as in scripts/v2/execution.py.
Also writes the 2001-2017 counterpart for the same books so the two periods sit side by side.
Run: python scripts/v2/seen_period.py"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qcore import metrics
from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.splits import Splits, select
from qengine.costs import CostModel
from qengine.engine import run_weights
from qengine.montecarlo import mc_summary
from qengine.tearsheet import write_tearsheet
from trend import model
from trend.v2.books import build_weights, garch_full, load_inputs
from trend.v2.execution import market_inputs

ROOT = Path(__file__).resolve().parents[2]
TAB, TS = ROOT / "results" / "v2" / "tables", ROOT / "results" / "v2" / "tearsheets"
PREREG = ROOT / "research" / "preregistration_v2.md"
os.environ.setdefault("PREREG_HASH", "2d7828df5093")
AUM, COEF = 1e8, 0.5
SEEN = "seen period (v1 reported it)"


def sh(r) -> float:
    r = pd.Series(np.asarray(r))
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def stats(r: pd.Series, res, sl) -> dict:
    lo, hi = percentile_ci(bootstrap_stat(r, sh, n_boot=1000, mean_block=21))
    c = sl(res.costs)
    return {"net_sharpe": sh(r), "ci95_lo": lo, "ci95_hi": hi, "n_days": len(r),
            "ann_return": float(r.mean() * 252), "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
            "max_drawdown": float(metrics.drawdown(r).min()),
            "ann_cost_linear": float(c["linear"].mean() * 252), "ann_cost_roll": float(c["roll"].mean() * 252),
            "ann_cost_impact": float(c["impact"].mean() * 252),
            "turnover_per_yr": float(sl(res.turnover).mean() * 252),
            "n_rebalance_days": int((sl(res.turnover) > 0).sum())}


def main():
    inp = load_inputs(ROOT)
    Rp0 = model.prepare(inp["R"])
    garch = garch_full(Rp0, ROOT / "results" / "v2" / "runs" / "garch_vol_full.parquet")
    W, Rp = build_weights(inp, garch)
    inst = inp["inst"]
    mi = market_inputs(Rp, inp["close"], inp["volume"], inp["fx"])
    thin = json.load(open(TAB / "execution_thin_adv.json"))["thin_adv_instruments"]
    mi_lo = {**mi, "adv_usd": mi["adv_usd"].copy()}
    mi_lo["adv_usd"][thin] = np.nan
    base = CostModel.from_instruments(inst, 0.0)
    cm = CostModel.from_instruments(inst, impact_coef=COEF)
    splits = Splits.from_dict(yaml.safe_load((ROOT / "configs" / "splits_v2.yaml").read_text()))
    start = pd.Timestamp("2001-01-01")
    periods = {
        "2001-2017": lambda x: x.loc[start:"2017-12-31"],
        f"2018-2026 - {SEEN}": lambda x: pd.concat([select(x, splits, "test", PREREG), select(x, splits, "recent", PREREG)]),
    }
    execs = {
        "v1 costs, lag2": lambda w: run_weights(w, Rp, base, lag=2),
        f"impact {COEF} @ ${AUM/1e6:.0f}m, Yahoo ADV, lag2": lambda w: run_weights(
            w, Rp, cm, lag=2, aum=AUM, adv_usd=mi["adv_usd"], sigma_daily=mi["sigma_daily"]),
        f"impact {COEF} @ ${AUM/1e6:.0f}m, thin-ADV impact off, lag2": lambda w: run_weights(
            w, Rp, cm, lag=2, aum=AUM, adv_usd=mi_lo["adv_usd"], sigma_daily=mi_lo["sigma_daily"]),
    }
    rows, nets = [], {}
    for book, w in W.items():
        for en, fn in execs.items():
            res = fn(w)
            for pn, sl in periods.items():
                r = sl(res.net)
                r = r[r.index >= start]
                rows.append({"period": pn, "book": book, "execution": en, **stats(r, res, sl)})
                nets[(book, en, pn)] = (r, sl(res.costs))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "seen_period.csv", index=False)
    print(t.round(3).to_string())

    # tearsheets: v1 blend and GARCH blend, 2001-2017 and seen period, for the realistic impact run and the v1-cost run
    for (book, en, pn), (r, c) in nets.items():
        if "thin" in en:
            continue
        tag = "v1costs" if en.startswith("v1") else "impact"
        slug = f"{book}_{'dev_val_2001_2017' if pn.startswith('2001') else 'seen_2018_2026'}_{tag}"
        title = f"P7 v2 {book} {pn.split(' - ')[0]} ({en})" + (f" - {SEEN}" if pn.startswith("2018") else "")
        notes = [f"Signal computed at close t from data up to and including t; trade at the next close (lag 2).",
                 f"Costs: {'v1 per-instrument linear bp + roll cost only, no impact' if tag == 'v1costs' else f'v1 linear + roll + square-root impact coef {COEF} at AUM ${AUM/1e6:.0f}m, Yahoo volume ADV (upper bound on impact)'}.",
                 "Data: Yahoo Finance '=F' continuous futures plus synthetic FX/rates/cash indices (v1 universe, 39 instruments); not back-adjusted.",
                 f"Roll-gap treatment: none applied; imputation was a sensitivity only (research/data_audit_v2.md).",
                 "Vol model: " + ("GJR-GARCH(1,1,1)-t, refit every 63 days, EWMA before 750 obs" if book == "garch_blend" else "EWMA com 60 (v1)"),
                 "Sharpe CIs: stationary bootstrap, mean block 21. This is a research artifact, not investment advice."]
        if pn.startswith("2018"):
            notes.insert(0, f"{SEEN}: v1 already reported 2018-2026; this is descriptive, not out-of-sample evidence.")
        write_tearsheet(r, TS / f"{slug}.html", title, costs=c, mc=mc_summary(r, n_paths=2000), notes=notes)
    print("tearsheets:", sorted(p.name for p in TS.glob("*.html")))


if __name__ == "__main__":
    main()
