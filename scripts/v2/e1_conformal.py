"""P7 v2 Task 7, E1 (EXPLORATORY, never headline): adaptive conformal intervals on next-21-day instrument returns.
Pre-registration v2 (hash 2d7828df5093). 2001-2017 only. yhat = b x v1 signal (mean of tanh t21/t63/t252), b = pooled OLS
slope of y on the signal over 2001-2008 with a 21-day purge. Intervals: delayed ACI (alpha 0.1, gamma 0.005, window 250
weekly rows, delay 5 rows because the 21-day label overlaps; qcore.aci_intervals has no delay). Coverage reported 2009-2017.
Filter book: v1 position kept on days where the (last weekly) interval excludes 0, else half position; lag 2, v1 costs.
Run: PREREG_HASH=2d7828df5093 python scripts/v2/e1_conformal.py"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.splits import file_hash
from qengine.costs import CostModel
from qengine.engine import run_weights
from trend import data, model
from trend.v2.features import aci_delayed, feature_panel

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "v2" / "tables"
PREREG = ROOT / "research" / "preregistration_v2.md"
END, FIT_END, EVAL_START = "2017-12-31", "2008-12-31", "2009-01-01"
ALPHA, STEP, DELAY = 0.1, 5, 5


def sh(r) -> float:
    r = pd.Series(np.asarray(r)); return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def sci(r: pd.Series) -> list[float]:
    lo, hi = percentile_ci(bootstrap_stat(r, sh, n_boot=1000, mean_block=21))
    return [sh(r), lo, hi]


def main() -> None:
    assert os.environ.get("PREREG_HASH") == file_hash(PREREG) == "2d7828df5093"
    R = pd.read_parquet(data.PROC / "returns.parquet").loc[:END]
    inst = data.instrument_table()
    cfg = model.TrendConfig.from_yaml(yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text()))
    Rp = model.prepare(R)
    P = feature_panel(Rp, step=STEP).dropna(subset=["t21", "t63", "t252"])
    P["sig"] = (np.tanh(P["t21"]) + np.tanh(P["t63"]) + np.tanh(P["t252"])) / 3
    P = P.loc[P.index.get_level_values(0) >= "2001-01-01"]
    d = P.index.get_level_values(0)
    cut = Rp.index[Rp.index <= FIT_END][-22]                        # label window t+1..t+21 ends on or before FIT_END
    tr = P[(d <= cut)].dropna(subset=["y"])
    b = float((tr["sig"] * tr["y"]).sum() / (tr["sig"] ** 2).sum())       # no-intercept OLS slope
    b_t = float(np.corrcoef(tr["sig"], tr["y"])[0, 1])
    P["yhat"] = b * P["sig"]
    ivs = {}
    for tk, g in P.groupby(level=1):
        g = g.droplevel(1).dropna(subset=["y"])
        if len(g) > 400:
            ivs[tk] = aci_delayed(g["y"], g["yhat"], alpha=ALPHA, window=250, delay=DELAY)
    sector = inst["sector"]
    cov_rows = []
    for tk, iv in ivs.items():
        x = iv.loc[EVAL_START:END].dropna(subset=["covered"])
        cov_rows.append({"ticker": tk, "sector": sector[tk], "coverage": x["covered"].mean(), "n": len(x),
                         "width": float((x["hi"] - x["lo"]).replace(np.inf, np.nan).mean()),
                         "excludes_zero_share": float(((x["lo"] > 0) | (x["hi"] < 0)).mean())})
    cv = pd.DataFrame(cov_rows)
    cv.to_csv(TAB / "e1_coverage_by_instrument.csv", index=False)
    by_sec = cv.groupby("sector").apply(lambda g: pd.Series({"coverage": np.average(g["coverage"], weights=g["n"]),
                                                              "n_obs": int(g["n"].sum()), "n_instruments": len(g)}), include_groups=False)
    tot = float(np.average(cv["coverage"], weights=cv["n"]))
    # pooled coverage CI: block bootstrap over weekly dates of the cross-instrument mean coverage
    allc = pd.concat({tk: iv["covered"] for tk, iv in ivs.items()}, axis=1).loc[EVAL_START:END]
    wk = allc.mean(axis=1).dropna()
    cl, ch = percentile_ci(bootstrap_stat(wk, lambda x: float(np.mean(x)), n_boot=1000, mean_block=8))

    # filter book
    keep = pd.DataFrame(1.0, index=R.index, columns=R.columns)
    for tk, iv in ivs.items():
        ex = ((iv["lo"] > 0) | (iv["hi"] < 0)).where(iv["lo"].notna())
        full = ex.reindex(R.index).ffill()                          # status known at the weekly date, carried forward
        keep[tk] = np.where(full.fillna(True).astype(bool), 1.0, 0.5)
    v1 = model.backtest(R, inst, cfg)
    W = v1["weights"]
    Rx = Rp.reindex(columns=W.columns).fillna(0.0)
    cm = CostModel.from_instruments(inst, 0.0)
    base = run_weights(W, Rx, cm, lag=2)
    filt = run_weights(W * keep.reindex(columns=W.columns), Rx, cm, lag=2)
    rows = {}
    sl = slice(EVAL_START, END)
    nb, nf = base.net.loc[sl], filt.net.loc[sl]
    D = pd.concat([nf, nb], axis=1, keys=["f", "b"]).dropna()
    dd = bootstrap_stat(D, lambda x: sh(x["f"]) - sh(x["b"]), n_boot=1000, mean_block=21)
    rows = {"base_v1_lag2_sharpe_ci": sci(nb), "filtered_sharpe_ci": sci(nf), "diff_filtered_minus_base": sh(D["f"]) - sh(D["b"]),
            "diff_ci95": list(percentile_ci(dd)), "n_days": len(D),
            "turnover_per_yr_base": float(base.turnover.loc[sl].mean() * 252), "turnover_per_yr_filtered": float(filt.turnover.loc[sl].mean() * 252),
            "rebalance_days_base": int((base.turnover.loc[sl] > 0).sum()), "rebalance_days_filtered": int((filt.turnover.loc[sl] > 0).sum()),
            "mean_keep_share_full_position": float((keep.loc[sl] == 1).to_numpy().mean())}
    out = {"label": "EXPLORATORY (E1); not a hypothesis test; never headline", "prereg_hash": file_hash(PREREG),
           "timing": "features at close t, label t+1..t+21, interval from labels realised by t (delay 5 weekly rows), position change lag 2",
           "fit": {"b": b, "corr_signal_label_2001_2008": b_t, "n_train_rows": len(tr)},
           "coverage_target": 1 - ALPHA, "coverage_2009_2017": {"overall_weighted": tot, "bootstrap_ci95_weekly_mean": [cl, ch],
                                                                    "by_sector": by_sec.to_dict("index")},
           "filter_book_2009_2017": rows}
    # optional MAPIE cross-check on ES=F
    try:
        from mapie.regression import TimeSeriesRegressor
        from mapie.subsample import BlockBootstrap
        from sklearn.linear_model import LinearRegression
        g = P.xs("ES=F", level=1).dropna(subset=["y"])
        trn, tst = g.loc[:cut], g.loc[EVAL_START:END]
        cv_ = BlockBootstrap(n_resamplings=30, length=20, overlapping=True, random_state=0)
        m = TimeSeriesRegressor(LinearRegression(), method="enbpi", cv=cv_, agg_function="mean", n_jobs=1)
        m.fit(trn[["sig"]].to_numpy(), trn["y"].to_numpy())
        _, pi = m.predict(tst[["sig"]].to_numpy(), confidence_level=1 - ALPHA, ensemble=True)
        lo_, hi_ = pi[:, 0, 0], pi[:, 1, 0]
        mc = float(((tst["y"].to_numpy() >= lo_) & (tst["y"].to_numpy() <= hi_)).mean())
        out["mapie_enbpi_crosscheck_ES"] = {"coverage": mc, "n": len(tst), "aci_delayed_coverage_ES": float(ivs["ES=F"].loc[EVAL_START:END]["covered"].mean()),
                                            "note": "static EnbPI (no re-fit of residuals after training) vs our delayed ACI; ES=F only"}
    except Exception as e:                                       # noqa: BLE001
        out["mapie_enbpi_crosscheck_ES"] = {"skipped": repr(e)[:300]}
    (TAB / "e1_conformal.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
