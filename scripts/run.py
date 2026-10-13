"""Run the trend system. Usage:

    python scripts/run.py download            # fetch free data
    python scripts/run.py build               # clean returns
    python scripts/run.py dev                 # development + validation report (always allowed)
    python scripts/run.py test                # locked: needs PREREG_HASH = hash of research/preregistration.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from trend import analysis as A
from trend import data, model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def load():
    R = pd.read_parquet(data.PROC / "returns.parquet")
    inst = data.instrument_table()
    cfg_d = yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text())
    rf = data.tbill_daily().reindex(R.index).ffill()
    bench = pd.read_parquet(data.PROC / "benchmarks.parquet")
    return R, inst, cfg_d, rf, bench


def run_books(R, inst, cfg_d) -> dict[str, dict]:
    books = {"blend": model.TrendConfig.from_yaml(cfg_d)}
    for h in cfg_d["horizons"]:
        books[h] = model.TrendConfig.from_yaml(cfg_d, use=(h,))
    books["disclosed_mix"] = model.TrendConfig.from_yaml(cfg_d, sector_weights=cfg_d["disclosed_sector_mix"])
    for s in inst["sector"].unique():
        books[f"sector_{s}"] = model.TrendConfig.from_yaml(cfg_d, sectors=(s,))
    return {k: model.backtest(R, inst, c) for k, c in books.items()}


def save_runs(res: dict) -> None:
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v["net"] for k, v in res.items()}).to_parquet(OUT / "runs" / "net_daily.parquet")
    pd.DataFrame({k: v["gross"] for k, v in res.items()}).to_parquet(OUT / "runs" / "gross_daily.parquet")
    res["blend"]["weights"].to_parquet(OUT / "runs" / "blend_weights.parquet")
    res["blend"]["contrib"].to_parquet(OUT / "runs" / "blend_contrib.parquet")


def fund_comparison(net: pd.Series, rf: pd.Series, bench: pd.DataFrame, name: str) -> pd.DataFrame:
    """Monthly correlation of (strategy excess + T-bill) with real fund NAV returns, one period."""
    ours = A.monthly(net + rf)
    rows = {}
    for f in ["EBSIX", "AQMIX", "ASFYX", "DBMF", "KMLM", "SPY", "AGG"]:
        fm = A.monthly(bench[f].dropna().pct_change().dropna())
        df = A.period(pd.concat([ours, fm], axis=1).dropna(), name)
        if len(df) >= 12:
            rows[f] = {"corr": df.corr().iloc[0, 1], "n_months": len(df)}
    return pd.DataFrame(rows).T


def report(res, rf, bench, periods: list[str], tag: str) -> dict:
    net = {k: v["net"] for k, v in res.items()}
    out = {}
    for p in periods:
        t = A.table(net, p)
        t.to_csv(OUT / "tables" / f"summary_{p}.csv")
        cost = pd.Series({k: A.period(v["cost"], p).sum() / max(len(A.period(v["cost"], p)) / 252, 1e-9)
                          for k, v in res.items()})
        fc = fund_comparison(res["blend"]["net"], rf, bench, p)
        fc.to_csv(OUT / "tables" / f"fund_corr_{p}.csv")
        print(f"\n===== {p} =====")
        print(t[["n_periods", "ann_return", "ann_vol", "sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi",
                 "max_drawdown", "skew"]].round(3).to_string())
        print("annual cost drag:", cost.round(4).to_dict())
        print(fc.round(3).to_string())
        out[p] = {"summary": t.to_dict(orient="index"), "cost_drag": cost.to_dict(), "fund_corr": fc.to_dict(orient="index")}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"summary_{tag}.json").write_text(json.dumps(out, default=str, indent=1))
    return out


def hypotheses(res, bench, name: str) -> dict:
    """H1-H3 exactly as written in research/preregistration.md."""
    m = {k: A.period(A.monthly(v["net"]), name) for k, v in res.items()}
    spy = A.period(A.monthly(bench["SPY"].dropna().pct_change().dropna()), name)
    chosen = json.loads((ROOT / "research" / "sources" / "best_single_horizon_devval.json").read_text())["best"]
    h = {
        "H1_blend_sharpe_positive": A.sharpe_positive_test(m["blend"]),
        "H2_blend_minus_best_single": {"best_single_from_dev_val": chosen, **A.sharpe_diff_test(m["blend"], m[chosen])},
        "H3_convexity_vs_SPY": A.convexity(m["blend"], spy),
    }
    print(json.dumps(h, indent=1, default=str))
    (OUT / "runs" / f"hypotheses_{name}.json").write_text(json.dumps(h, indent=1, default=str))
    return h


def main(cmd: str) -> None:
    if cmd == "download":
        data.download()
        return
    if cmd == "build":
        data.build()
        return
    R, inst, cfg_d, rf, bench = load()
    res = run_books(R, inst, cfg_d)
    save_runs(res)
    if cmd == "dev":
        report(res, rf, bench, ["development", "validation"], "devval")
        # Choice made on dev+val only and frozen for H2: best single horizon by net Sharpe.
        dv = {h: A.sharpe_m(pd.concat([A.period(A.monthly(res[h]["net"]), p) for p in ("development", "validation")]))
              for h in cfg_d["horizons"]}
        best = max(dv, key=dv.get)
        (ROOT / "research" / "sources" / "best_single_horizon_devval.json").write_text(
            json.dumps({"sharpe_devval": dv, "best": best}, indent=1))
        print("dev+val single-horizon Sharpe:", dv, "-> best", best)
        for p in ("development", "validation"):
            hypotheses(res, bench, p)
    elif cmd == "test":
        report(res, rf, bench, ["test", "recent"], "test")
        hypotheses(res, bench, "test")
        hypotheses(res, bench, "recent")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
