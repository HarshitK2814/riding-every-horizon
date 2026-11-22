"""P7 v3 Task 9: export the data the GitHub Pages site (docs/) needs.

Writes docs/data/: returns.csv.gz (snapshot/ vintage, float32, 6 dp), instruments.json, tsfm_signal.csv.gz
(TimesFM signal, blended over horizons and held <= 4 extra rows exactly as trend.v3.books.signal_frame, on the
returns index), verdicts.json (v1/v2/v3), forward.json (v2 + v3 forward CSVs) and meta.json (reference numbers).
Signal at close t, trade t+1; v1 costs (cost_bp + roll). Returns always come from snapshot/, never data/.
default_run.json is docs/trend_lite.py's own run_json output for the lab's default settings, so the page can show
real numbers before the in-browser engine has loaded (the browser then reruns it and checks it against meta.json).
Usage: python scripts/v3/export_site_data.py [--forward-only | --default-run-only]   (--forward-only rewrites only
forward.json; the daily forward workflow runs it. --default-run-only rewrites only default_run.json from docs/data)
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "data"
SNAP = ROOT / "snapshot" / "processed" / "returns.parquet"
GH = "https://github.com/HarshitK2814/riding-every-horizon/blob/main/"
V3_FIRST_READ = "2027-11-30"          # research/preregistration_v3.md line 31
V2_FIRST_READ = "2027-11"             # research/preregistration_v2.md line 45 ("Earliest read: 2027-11")
# books each tracker writes (scripts/v2/forward.py line 1, scripts/v3/forward.py line 1); listed so an empty CSV
# still renders one row per book
FORWARD_BOOKS = {"v2": ["v1_blend", "garch_blend"], "v3": ["v1_blend", "tsfm_blend", "clean_blend"]}

# v1 verdict words: research/results.md lines 11-13 (numbers come from results/runs/hypotheses_test.json)
V1_STATUS = {"H1": "not_supported", "H2": "not_supported", "H3": "supported"}
# v2 verdicts exist only as markdown: research/results_v2.md "Bottom line" lines 9-11 and the
# "Hypotheses (2001-2017 only)" table lines 32-34. Numbers copied verbatim from those lines.
V2 = [
    dict(id="H1", claim="GARCH tracks the 10% risk target better than EWMA",
         result="EWMA RMSE minus GARCH RMSE -15.117, 95% CI [-18.807, -11.032]", status="not_supported",
         explain="GARCH fits blew up on a few instruments, so the plain EWMA risk model tracked the target better."),
    dict(id="H2", claim="v1 is not an overfit pick: PBO < 0.5 and deflated Sharpe > 0.90",
         result="PBO 0.030 (passes); deflated Sharpe 0.081 (fails)", status="not_supported",
         explain="The chosen settings were stable inside the 96-config family, but after allowing for 96 tries "
                 "the Sharpe is not convincingly above zero."),
    dict(id="H3", claim="Crisis alpha follows crash clustering",
         result="intensity coef 0.179, Newey-West t 0.80 (needs > 2), 121 months", status="not_supported",
         explain="Months when crashes clustered were not the months trend following earned its crisis gains."),
]


def _gz_csv(df: pd.DataFrame, path: Path) -> None:
    buf = io.StringIO()
    df.astype("float32").to_csv(buf, float_format="%.6f", date_format="%Y-%m-%d", index_label="date")
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as f:   # mtime=0: byte-identical reruns
        f.write(buf.getvalue().encode())


def _dump(obj, name: str) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def _md_has(path: Path, *needles: str) -> None:
    """Fail loudly if a markdown verdict held in this file no longer matches its source."""
    txt = path.read_text(encoding="utf-8")
    for n in needles:
        assert n in txt, f"{path.name} no longer contains {n!r}: update export_site_data.py"


def verdicts() -> dict:
    j = json.loads((ROOT / "results/runs/hypotheses_test.json").read_text())
    _md_has(ROOT / "research/results.md", "**Not supported**", "**Supported**")
    h1, h2, h3 = j["H1_blend_sharpe_positive"], j["H2_blend_minus_best_single"], j["H3_convexity_vs_SPY"]
    v1 = [
        dict(id="H1", claim="Three-horizon blend has Sharpe > 0 (net, 2018-2024 test)",
             result=f"Sharpe {h1['sharpe']:.2f}, 95% CI [{h1['ci'][0]:.2f}, {h1['ci'][1]:.2f}], "
                    f"p = {h1['p_one_sided']:.2f}, {h1['n_months']} months", status=V1_STATUS["H1"],
             explain="The point estimate is positive, but the interval includes zero: the test years cannot show an edge."),
        dict(id="H2", claim=f"Blend beats the best single horizon ({h2['best_single_from_dev_val']}, chosen on dev+val)",
             result=f"difference {h2['diff']:.2f}, 95% CI [{h2['ci'][0]:.2f}, {h2['ci'][1]:.2f}], "
                    f"p = {h2['p_one_sided']:.2f}", status=V1_STATUS["H2"],
             explain="The 252-day horizon alone did at least as well; mixing horizons did not help (as dev+val predicted)."),
        dict(id="H3", claim="Crisis convexity: c > 0 in r = a + b m + c m² (monthly SPY)",
             result=f"c = {h3['gamma']:.2f}, HAC t = {h3['t_gamma']:.2f}, p = {h3['p_gamma_one_sided']:.4f}, "
                    f"{h3['n_months']} months", status=V1_STATUS["H3"],
             explain="Returns were higher in months when stocks moved a lot either way: the 'crisis smile' held."),
    ]
    _md_has(ROOT / "research/results_v2.md", "PBO 0.030", "0.081", "-15.117 [-18.807, -11.032]", "t 0.80")
    h = json.loads((ROOT / "results/v3/runs/h2.json").read_text())
    _md_has(ROOT / "research/results_v3.md", f"pending — first read {V3_FIRST_READ}")
    v3 = [
        dict(id="H1", claim="TimesFM blend beats v1 blend, net Sharpe, forward period (confirmatory)",
             result=f"no forward data yet; tracking from 2026-12-01, first read {V3_FIRST_READ} (≥ 240 trading days)",
             status="pending",
             explain="Only data neither the model nor the researcher has seen can answer this, so it waits a year."),
        dict(id="H2", claim="A clean model (trained per fold) beats v1 on CPCV 2001-2017",
             result=f"mean path difference {h['mean_path_diff']:.2f}, 95% CI [{h['ci_lo']:.2f}, {h['ci_hi']:.2f}]; "
                    f"deflated Sharpe {h['dsr_clean']:.2f} (bar 0.90); {h['paths']} paths, {h['days']} days",
             status="supported" if h["supported"] else "not_supported",
             explain="The clean model looked better on average, but the interval spans zero and the deflated Sharpe misses the bar."),
    ]
    return {
        "v1": dict(title="v1 · horizon blend", period="test 2018-2024 (pre-registered, monthly)",
                   source=GH + "research/results.md", rows=v1),
        "v2": dict(title="v2 · risk model & overfitting audit", period="2001-2017 (choice period)",
                   source=GH + "research/results_v2.md", rows=V2),
        "v3": dict(title="v3 · pretrained forecaster", period="CPCV 2001-2017; forward from 2026-12-01",
                   source=GH + "research/results_v3.md", rows=v3),
    }


def forward() -> dict:
    out = {"as_of": None}
    for v, start_key, first_read in (("v2", "splits_v2.yaml", V2_FIRST_READ), ("v3", "splits_v3.yaml", V3_FIRST_READ)):
        start = pd.Period(yaml.safe_load((ROOT / "configs" / start_key).read_text())["forward"][0], "M").start_time
        p = ROOT / "results" / v / "forward" / "forward.csv"
        f = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=["date", "book", "net_return"])
        books = {}
        for b in dict.fromkeys(FORWARD_BOOKS[v] + list(f["book"].unique())):
            r = f[f["book"] == b].sort_values("date").drop_duplicates("date", keep="last")
            cum = (1 + r["net_return"].astype(float)).cumprod() - 1
            books[b] = dict(dates=[f"{d:%Y-%m-%d}" for d in r["date"]], cum=[round(float(x), 6) for x in cum],
                            days=int(len(r)))
        last = max((d for b in books.values() for d in b["dates"]), default=None)
        out[v] = dict(start=f"{start:%Y-%m-%d}", first_read=first_read, books=books, last_date=last,
                      days=max((b["days"] for b in books.values()), default=0),
                      note="signal at close t, traded at the next close (lag 2); v1 costs, no impact",
                      source=GH + f"results/{v}/forward/forward.csv")
        out["as_of"] = max(filter(None, [out["as_of"], last]), default=None)
    return out


# the lab's default settings (docs/app.js DEFAULTS, target_vol in decimals as the page sends it)
LAB_DEFAULT = {"signal": "tanh", "use": ["short", "medium", "long"], "buffer": 0.10, "target_vol": 0.10,
               "cost_mult": 1.0, "start": None, "end": None, "windows": None}


def default_run() -> None:
    """Run docs/trend_lite.py on the docs/data files exactly as the browser worker does; write default_run.json."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trend_lite", ROOT / "docs" / "trend_lite.py")
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    rb = lambda n: (OUT / n).read_bytes()
    inp = tl.load_inputs(rb("returns.csv.gz"), (OUT / "instruments.json").read_text(encoding="utf-8"),
                         rb("tsfm_signal.csv.gz"), rb("tbill.csv.gz"))
    out = json.loads(tl.run_json(inp, json.dumps(LAB_DEFAULT)))
    out["equity"] = [round(v, 5) for v in out["equity"]]
    out["params"] = LAB_DEFAULT
    out["note"] = "precomputed with docs/trend_lite.py at export; the page reruns it in the browser"
    (OUT / "default_run.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n",
                                          encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--default-run-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.default_run_only:
        default_run()
        print("wrote default_run.json")
        return
    _dump(forward(), "forward.json")
    if a.forward_only:
        print("wrote forward.json")
        return
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from trend import data, model
    from trend.v3.books import signal_frame
    from trend.v3.labels import CONTAM_END, FORWARD_START, SEEN_START

    R = pd.read_parquet(SNAP)
    inst = data.instrument_table().reindex(R.columns)
    assert inst.notna().all().all(), "instrument table does not cover every snapshot column"
    _gz_csv(R, OUT / "returns.csv.gz")
    sig = signal_frame(pd.read_parquet(ROOT / "results/v3/cache/tsfm_signal.parquet"), model.prepare(R).index)
    _gz_csv(sig, OUT / "tsfm_signal.csv.gz")
    # daily T-bill return exactly as scripts/figures.py: FRED DTB3 / 100 / 252 from the snapshot vintage,
    # reindexed to the returns index, forward-filled, 0 before the first quote
    y = pd.read_parquet(ROOT / "snapshot" / "raw" / "dtb3.parquet")["DTB3"] / 100
    rf = (y / 252).reindex(R.index).ffill().fillna(0.0)
    buf = io.StringIO()
    rf.rename("rf").to_csv(buf, float_format="%.9f", date_format="%Y-%m-%d", index_label="date")
    with gzip.GzipFile(OUT / "tbill.csv.gz", "wb", compresslevel=9, mtime=0) as f:
        f.write(buf.getvalue().encode())
    _dump({t: dict(sector=r.sector, cost_bp=float(r.cost_bp), rolls=int(r.rolls)) for t, r in inst.iterrows()},
          "instruments.json")
    _dump(verdicts(), "verdicts.json")

    # reference: the paper engine on the same vintage, default settings, full period (the site must match it)
    cfg = model.TrendConfig.from_yaml(yaml.safe_load((ROOT / "configs/strategy.yaml").read_text()))
    net = model.backtest(R, inst, cfg)["net"]
    s = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
    summ = pd.read_csv(ROOT / "results/v3/tables/summary_by_period.csv")
    _dump(dict(
        vintage=re.search(r"downloaded \*\*(\d{4}-\d\d-\d\d)\*\*",
                          (ROOT / "results/v3/DATA_VINTAGE.md").read_text(encoding="utf-8")).group(1),
        first_date=f"{R.index[0]:%Y-%m-%d}", last_date=f"{R.index[-1]:%Y-%m-%d}", n_instruments=int(R.shape[1]),
        tsfm_instruments=int(sig.notna().any().sum()),
        labels=dict(seen_start=f"{SEEN_START:%Y-%m-%d}", contam_end=f"{CONTAM_END:%Y-%m-%d}",
                    forward_start=f"{FORWARD_START:%Y-%m-%d}"),
        reference=dict(book="v1_blend", period="full", start=f"{net.index[0]:%Y-%m-%d}", end=f"{net.index[-1]:%Y-%m-%d}",
                       days=int(len(net)), sharpe=round(s, 6), engine="trend.model.backtest",
                       note="default settings; the in-browser rerun must match within 0.01"),
        periods=[dict(book=r.book, period=r.period, label=r.label, start=r.start, end=r.end, days=int(r.days),
                      sharpe=round(float(r.sharpe), 4))
                 for r in summ.itertuples() if r.book in ("v1_blend", "tsfm_blend") and r.days > 0],
        bootstrap=dict(n_boot=500, mean_block=20, seed=0, paper_n_boot=2000),
        timing="signal at close t, trade t+1 (lag 1)", costs="v1 cost_bp + roll (two trades per roll), x cost multiplier; no impact",
    ), "meta.json")
    default_run()
    sizes = {p.name: p.stat().st_size for p in sorted(OUT.iterdir())}
    print("wrote", sizes, "total MB", round(sum(sizes.values()) / 1e6, 2))


if __name__ == "__main__":
    main()
