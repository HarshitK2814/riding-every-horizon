"""P7 v3 forward tracker (H1). Books v1_blend, tsfm_blend, clean_blend. Signal at close t, traded at the next
close (lag 2, as v2), v1 costs, no impact. Writes only dates in the v3 'forward' split (locked by the v3 prereg
hash). Idempotent.
Usage: python scripts/v3/forward.py [--refresh | --refresh-signals]
--refresh: re-download data (v1 functions), then as --refresh-signals.
--refresh-signals: no download; extend a LIVE copy of the TimesFM signals (auto device: GPU if present, else CPU) and
recompute clean signals from the frozen model results/v3/cache/clean_model.pt (never retrained). Live files go to
results/v3/forward/ (clean_signal_live.parquet, tsfm_signal_live.parquet); the reference cache in results/v3/cache/ is
read-only here and never written.
Without either, data and the live/reference files on disk are used."""
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
CACHE = ROOT / "results" / "v3" / "cache"          # frozen reference: read-only in this script
FWD = ROOT / "results" / "v3" / "forward"          # live signals written by the daily bot
sys.path.insert(0, str(ROOT / "scripts" / "v3"))


def live_paths(fwd_dir: Path) -> tuple[Path, Path]:
    return fwd_dir / "clean_signal_live.parquet", fwd_dir / "tsfm_signal_live.parquet"


def update_signals(R: pd.DataFrame, cache_dir: Path, fwd_dir: Path, forecast=None, predict=clean_predict) -> None:
    """Write live signals under fwd_dir. cache_dir (reference) is only read. TimesFM live file starts from a copy of the
    reference cache and only appends new dates; clean signals are recomputed from the frozen clean_model.pt."""
    clean_p, tsfm_p = live_paths(fwd_dir)
    fwd_dir.mkdir(parents=True, exist_ok=True)
    if forecast is None:
        import forecast_tsfm
        forecast = forecast_tsfm.main
    forecast(base=cache_dir / "tsfm_signal.parquet", out_path=tsfm_p)
    predict(cache_dir / "clean_model.pt", R).to_parquet(clean_p)


def read_signals(cache_dir: Path, fwd_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tsfm_long, clean): live files if present, else the reference cache."""
    clean_p, tsfm_p = live_paths(fwd_dir)
    tsfm = pd.read_parquet(tsfm_p if tsfm_p.exists() else cache_dir / "tsfm_signal.parquet")
    clean = pd.read_parquet(clean_p if clean_p.exists() else cache_dir / "clean_signal_full.parquet")
    return tsfm, clean


def main(refresh: bool = False, refresh_signals: bool = False) -> int:
    if refresh:
        data.download(); data.build(); macro.download_macro()
    refresh_signals = refresh or refresh_signals
    os.environ.setdefault("PREREG_HASH", file_hash(PREREG))
    inp = load_inputs(ROOT)
    if refresh_signals:
        update_signals(model.prepare(inp["R"]), CACHE, FWD)
    tsfm_long, clean = read_signals(CACHE, FWD)
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
