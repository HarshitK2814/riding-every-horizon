"""P7 v2 forward paper tracker. Books: v1_blend, garch_blend (no E1/E2 book: neither met its own bar).
Signal at close t, traded at the next close (lag 2), v1 costs, no impact. Only dates >= the forward start in
configs/splits_v2.yaml are written, through qcore.splits.select (locked by the v2 prereg hash). Idempotent.
Usage: python scripts/v2/forward.py [--refresh]    (--refresh re-downloads data with the v1 functions first;
without it the tracker uses the data already on disk, so seen-period numbers cannot shift under us.)"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from qcore.splits import Splits, file_hash, select
from qengine.costs import CostModel
from qengine.engine import run_weights
from trend import data, macro, model
from trend.v2.books import build_weights, garch_full, load_inputs
from trend.v2.forward_log import append_new

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "research" / "preregistration_v2.md"
OUT = ROOT / "results" / "v2" / "forward" / "forward.csv"
os.environ.setdefault("PREREG_HASH", "2d7828df5093")


def main(refresh: bool = False) -> int:
    if refresh:
        data.download(); data.build(); macro.download_macro()
    inp = load_inputs(ROOT)
    Rp0 = inp["R"]
    garch = garch_full(model.prepare(Rp0), ROOT / "results" / "v2" / "runs" / "garch_vol_full.parquet")
    W, Rp = build_weights(inp, garch)
    cost = CostModel.from_instruments(inp["inst"], 0.0)
    splits = Splits.from_dict(yaml.safe_load((ROOT / "configs" / "splits_v2.yaml").read_text()))
    h = file_hash(PREREG)
    rows = []
    for name, w in W.items():
        net = run_weights(w, Rp, cost, lag=2).net
        f = select(net, splits, "forward", PREREG)
        rows.append(pd.DataFrame({"date": f.index, "book": name, "net_return": f.to_numpy(), "prereg_hash": h}))
    rows = pd.concat(rows, ignore_index=True)
    n_new = append_new(OUT, rows)
    log = pd.read_csv(OUT, parse_dates=["date"])
    days = log["date"].nunique()
    print(f"data ends {Rp0.index[-1].date()}; forward start {splits.forward[0]}; new rows {n_new}; "
          f"forward days tracked: {days}")
    return days


if __name__ == "__main__":
    main("--refresh" in sys.argv)
