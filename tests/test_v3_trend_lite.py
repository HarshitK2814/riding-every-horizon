"""docs/trend_lite.py (the in-browser engine) must reproduce trend.model.backtest exactly."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend import model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trend_lite", ROOT / "docs/trend_lite.py")
tl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tl)

DEFAULT = {"use": ["short", "medium", "long"], "buffer": 0.10, "target_vol": 0.10, "cost_mult": 1.0,
           "signal": "tanh", "start": None, "end": None}


def test_lite_matches_full_engine():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-04", periods=700)
    R = pd.DataFrame(rng.normal(0.0002, 0.01, (700, 4)), index=idx, columns=list("ABCD"))
    inst = pd.DataFrame({"sector": ["x", "x", "y", "y"], "cost_bp": 2.0, "rolls": 4}, index=list("ABCD"))
    full = model.backtest(R, inst, model.TrendConfig())["net"]
    lite = tl.run(R, inst["sector"], inst["cost_bp"], inst["rolls"], dict(DEFAULT))["net"]
    pd.testing.assert_series_equal(full, lite, check_names=False, atol=1e-12)


def test_lite_variants_match_full_engine():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-04", periods=600)
    R = pd.DataFrame(rng.normal(0.0001, 0.012, (600, 4)), index=idx, columns=list("ABCD"))
    R.iloc[:80, 3] = np.nan                                   # late starter
    inst = pd.DataFrame({"sector": ["x", "x", "y", "y"], "cost_bp": 3.0, "rolls": 12}, index=list("ABCD"))
    cfg = model.TrendConfig(use=("short", "long"), buffer=0.2, target_vol=0.15, signal="sign")
    full = model.backtest(R, inst, cfg)["net"]
    p = dict(DEFAULT, use=["short", "long"], buffer=0.2, target_vol=0.15, signal="sign")
    lite = tl.run(R, inst["sector"], inst["cost_bp"], inst["rolls"], p)["net"]
    pd.testing.assert_series_equal(full, lite, check_names=False, atol=1e-12)
    sig = pd.DataFrame(np.tanh(rng.normal(0, 1, (600, 4))), index=idx, columns=list("ABCD"))
    full = model.backtest(R, inst, model.TrendConfig(), sig=sig)["net"]
    lite = tl.run(R, inst["sector"], inst["cost_bp"], inst["rolls"], dict(DEFAULT, signal="tsfm"), sig=sig)["net"]
    pd.testing.assert_series_equal(full, lite, check_names=False, atol=1e-12)


def test_lite_slices_point_in_time_and_scales_costs():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-04", periods=700)
    R = pd.DataFrame(rng.normal(0.0002, 0.01, (700, 4)), index=idx, columns=list("ABCD"))
    sec, c, rl = pd.Series(list("xxyy"), index=list("ABCD")), pd.Series(2.0, index=list("ABCD")), pd.Series(4, index=list("ABCD"))
    whole = tl.run(R, sec, c, rl, dict(DEFAULT))
    part = tl.run(R, sec, c, rl, dict(DEFAULT, start="2011-06-01", end="2012-01-31"))
    pd.testing.assert_series_equal(whole["net"].loc["2011-06-01":"2012-01-31"], part["net"], atol=1e-15)
    free = tl.run(R, sec, c, rl, dict(DEFAULT, cost_mult=0.0))["net"]
    dbl = tl.run(R, sec, c, rl, dict(DEFAULT, cost_mult=2.0))["net"]
    pd.testing.assert_series_equal(dbl - free, 2 * (whole["net"] - free), atol=1e-15)
    for k in ("sharpe", "sharpe_lo", "sharpe_hi", "ann_ret", "maxdd", "crisis", "days"):
        assert k in whole
    assert whole["sharpe_lo"] <= whole["sharpe"] <= whole["sharpe_hi"]


def test_lite_bootstrap_matches_qcore():
    from qcore.bootstrap import bootstrap_stat
    r = pd.Series(np.random.default_rng(5).normal(0.0003, 0.01, 400))
    ref = bootstrap_stat(r, tl.sharpe, n_boot=50, mean_block=20, seed=0)
    np.testing.assert_allclose(tl.bootstrap_sharpe(r.to_numpy(), n_boot=50, mean_block=20, seed=0), ref, atol=1e-12)


SNAP = ROOT / "snapshot" / "processed" / "returns.parquet"


@pytest.mark.skipif(not SNAP.exists(), reason="snapshot/ not present")
def test_lite_matches_engine_on_snapshot():
    import yaml
    from trend import data
    R = pd.read_parquet(SNAP)
    inst = data.instrument_table()
    cfg = model.TrendConfig.from_yaml(yaml.safe_load((ROOT / "configs/strategy.yaml").read_text()))
    full = model.backtest(R, inst, cfg)["net"]
    lite = tl.run(R, inst["sector"], inst["cost_bp"], inst["rolls"], dict(DEFAULT))
    assert abs(lite["sharpe"] - tl.sharpe(full)) < 0.01
    np.testing.assert_allclose(lite["net"].to_numpy(), full.to_numpy(), atol=1e-12)
    # every v1_blend row of the v3 summary table (same vintage) is reproduced from the same full-history run
    summ = ROOT / "results/v3/tables/summary_by_period.csv"
    if summ.exists():
        cache = {}
        for r in pd.read_csv(summ).query("book == 'v1_blend' and days > 0").itertuples():
            o = tl.run(R, inst["sector"], inst["cost_bp"], inst["rolls"], dict(DEFAULT, start=r.start, end=r.end), cache=cache)
            assert o["days"] == r.days and abs(o["sharpe"] - r.sharpe) < 0.01, (r.period, o["sharpe"], r.sharpe)


@pytest.mark.skipif(not SNAP.exists(), reason="snapshot/ not present")
def test_lite_total_crisis_returns_match_paper_table():
    """Crisis table compounds net + T-bill, as scripts/figures.py does; tolerance covers vintage revisions."""
    import yaml
    from trend import data
    dtb3 = ROOT / "snapshot" / "raw" / "dtb3.parquet"
    paper = ROOT / "results" / "tables" / "crisis_returns.csv"
    if not (dtb3.exists() and paper.exists()):
        pytest.skip("dtb3 snapshot or paper crisis table missing")
    R = pd.read_parquet(SNAP)
    inst = data.instrument_table()
    rf = (pd.read_parquet(dtb3)["DTB3"] / 100 / 252).reindex(R.index).ffill().fillna(0.0)
    o = tl.run(R, inst["sector"], inst["cost_bp"], inst["rolls"], dict(DEFAULT), rf=rf)
    ref = pd.read_csv(paper, index_col=0)["Trend blend"]
    wins = {"2008": ("2008-01-01", "2008-12-31", "2008"), "Q4 2018": ("2018-10-01", "2018-12-31", "2018Q4"),
            "Feb-Mar 2020": ("2020-02-19", "2020-03-23", "covid-2020"), "2022": ("2022-01-01", "2022-12-31", "2022")}
    for label, (a, z, key) in wins.items():
        assert (o["crisis"][key]["start"], o["crisis"][key]["end"]) == (a, z)
        assert abs(o["crisis"][key]["cum_net_return"] - ref[label]) < 0.005, (label, o["crisis"][key], ref[label])
    # Sharpe stays on excess returns; total = net + rf
    np.testing.assert_allclose((o["total"] - o["net"]).to_numpy(), rf.reindex(o["net"].index).to_numpy(), atol=1e-15)


def test_site_default_run_matches_reference():
    """docs/data/default_run.json (shown before the browser engine loads) agrees with the paper-engine reference."""
    import json
    d = ROOT / "docs" / "data"
    run = json.loads((d / "default_run.json").read_text(encoding="utf-8"))
    ref = json.loads((d / "meta.json").read_text(encoding="utf-8"))["reference"]
    assert run["days"] == ref["days"] == len(run["dates"]) == len(run["equity"])
    assert (run["dates"][0], run["dates"][-1]) == (ref["start"], ref["end"])
    assert abs(run["sharpe"] - ref["sharpe"]) < 0.01
    assert run["params"]["signal"] == "tanh" and run["params"]["cost_mult"] == 1.0
