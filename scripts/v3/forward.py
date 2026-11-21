"""P7 v3 forward tracker (H1). Books v1_blend, tsfm_blend, clean_blend. Signal at close t, traded at the next
close (lag 2, as v2), v1 costs, no impact. Writes only dates in the v3 'forward' split (locked by the v3 prereg
hash). Idempotent.
Usage: python scripts/v3/forward.py [--refresh | --refresh-signals]
--refresh: re-download data (v1 functions), then as --refresh-signals.
--refresh-signals: no download; extend the TimesFM cache (auto device: GPU if present, else CPU) and recompute clean
signals from the frozen model results/v3/cache/clean_model.pt (never retrained).
Without either, data and caches on disk are used."""
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
from trend.v2.books import load_inputs
from trend.v2.forward_log import append_new
from trend.v3.books import build_v3_weights
from trend.v3.clean_model import clean_predict

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "research" / "preregistration_v3.md"
OUT = ROOT / "results" / "v3" / "forward" / "forward.csv"
CACHE = ROOT / "results" / "v3" / "cache"
sys.path.insert(0, str(ROOT / "scripts" / "v3"))


def main(refresh: bool = False, refresh_signals: bool = False) -> int:
    if refresh:
        data.download(); data.build(); macro.download_macro()
    refresh_signals = refresh or refresh_signals
    if refresh_signals:
        import forecast_tsfm
        forecast_tsfm.main()
    os.environ.setdefault("PREREG_HASH", file_hash(PREREG))
    inp = load_inputs(ROOT)
    tsfm_long = pd.read_parquet(CACHE / "tsfm_signal.parquet")
    if refresh_signals:
        clean = clean_predict(CACHE / "clean_model.pt", model.prepare(inp["R"]))
        clean.to_parquet(CACHE / "clean_signal_full.parquet")
    else:
        clean = pd.read_parquet(CACHE / "clean_signal_full.parquet")
    W = build_v3_weights(inp, tsfm_long, clean)
    cost = CostModel.from_instruments(inp["inst"], 0.0)
    splits = Splits.from_dict(yaml.safe_load((ROOT / "configs/splits_v3.yaml").read_text()))
    h = file_hash(PREREG)
    rows = []
    for name in ("v1_blend", "tsfm_blend", "clean_blend"):
        w = W[name]
        Rp = model.prepare(inp["R"]).reindex(columns=w.columns).fillna(0.0)
        f = select(run_weights(w, Rp, cost, lag=2).net, splits, "forward", PREREG)
        rows.append(pd.DataFrame({"date": f.index, "book": name, "net_return": f.to_numpy(), "prereg_hash": h}))
    n = append_new(OUT, pd.concat(rows, ignore_index=True))
    print(f"data ends {inp['R'].index[-1].date()}; forward start {splits.forward[0]}; new rows {n}")
    return n


if __name__ == "__main__":
    main("--refresh" in sys.argv, "--refresh-signals" in sys.argv)
