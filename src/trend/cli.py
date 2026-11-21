"""One-command reproduce.  Usage: python -m trend.cli quick|full

quick: copies snapshot/ to a temp dir, re-runs the v3 tables and the H2 statistics from the committed signal
       caches (no download, no training, no TimesFM), and checks them against the committed tables. Minutes, CPU.
full:  re-downloads everything and rebuilds v1, v2, v3 (TimesFM forecasts, clean-model retrain, CPCV retrain).
       Hours, network, GPU recommended. Overwrites committed caches and results.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ["summary_by_period", "crisis", "e2_hitrate", "e3_costs", "h2_cpcv_paths"]
ATOL = 1e-6
V1_PREREG, V2_PREREG = ROOT / "research/preregistration.md", "2d7828df5093"
V2_SCRIPTS = ["audit", "risk_h1", "e1_conformal", "e2_pysr", "hawkes_h3", "overfit_audit", "execution",
              "seen_period", "make_results_v2"]          # make_results_v2 reads every other v2 output, so it is last
V3_SCRIPTS = ["forecast_tsfm", "h2_cpcv", "make_results_v3"]

FULL_WARNING = """\
WARNING: `full` rebuilds everything and OVERWRITES the committed caches and results.
  - Numbers can differ from the committed ones: vendors revise data, and GPU runs are not bit-reproducible.
  - The committed artefacts are the reference. `git diff` shows the drift; `git checkout -- results research` restores them.
  - Needs network, hours of compute, and a GPU for TimesFM (it falls back to CPU, much slower)."""


def _run(*args: str, env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env={**os.environ, **(env or {})})


def _git_show(rel: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")


def compare_frames(ref: pd.DataFrame, new: pd.DataFrame, name: str, atol: float = ATOL) -> None:
    """Numeric columns equal to atol (NaN matches NaN); every other column exactly equal."""
    assert list(ref.columns) == list(new.columns), f"{name}: columns differ {list(ref.columns)} vs {list(new.columns)}"
    assert len(ref) == len(new), f"{name}: {len(ref)} rows committed, {len(new)} reproduced"
    for c in ref.columns:
        a, b = ref[c], new[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.allclose(a.to_numpy(float), b.to_numpy(float), atol=atol, rtol=0, equal_nan=True)
        else:
            ok = a.fillna("").astype(str).equals(b.fillna("").astype(str))
        assert ok, f"{name}: column {c!r} differs from the committed version"


def compare_json(ref, new, name: str, atol: float = ATOL, path: str = "") -> None:
    if isinstance(ref, dict):
        assert isinstance(new, dict) and ref.keys() == new.keys(), f"{name}{path}: keys differ"
        for k in ref:
            compare_json(ref[k], new[k], name, atol, f"{path}.{k}")
    elif isinstance(ref, list):
        assert isinstance(new, list) and len(ref) == len(new), f"{name}{path}: list length differs"
        for i, (x, y) in enumerate(zip(ref, new)):
            compare_json(x, y, name, atol, f"{path}[{i}]")
    elif isinstance(ref, float) or isinstance(new, float):
        assert np.isclose(ref, new, atol=atol, rtol=0, equal_nan=True), f"{name}{path}: {ref} vs {new}"
    else:
        assert ref == new, f"{name}{path}: {ref!r} vs {new!r}"


def quick() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        shutil.copytree(ROOT / "snapshot", tmp / "data")
        # reference values come from git HEAD; outputs go to tmp/out, so the working tree is never touched
        env = {"TREND_DATA_DIR": str(tmp / "data"), "TREND_OUT_DIR": str(tmp / "out"), "TREND_QUICK": "1"}
        _run("scripts/v3/h2_cpcv.py", "--from-cache", env=env)
        _run("scripts/v3/make_results_v3.py", env=env)
        for name in TABLES:
            rel = f"results/v3/tables/{name}.csv"
            compare_frames(pd.read_csv(io.StringIO(_git_show(rel))), pd.read_csv(tmp / "out" / rel), name)
            print(f"match: {rel}")
        rel = "results/v3/runs/h2.json"
        compare_json(json.loads(_git_show(rel)), json.loads((tmp / "out" / rel).read_text()), "h2.json")
        print(f"match: {rel}")
    print("QUICK OK")


def full() -> None:
    print(FULL_WARNING, flush=True)
    from qcore.splits import file_hash
    _run("-c", "from trend import data; data.download(); data.build()")   # download() also fetches the FRED macro panel
    _run("scripts/run.py", "dev")
    _run("scripts/run.py", "test", env={"PREREG_HASH": file_hash(V1_PREREG)})
    for s in V2_SCRIPTS:
        _run(f"scripts/v2/{s}.py", env={"PREREG_HASH": V2_PREREG})
    _run("scripts/v3/forecast_tsfm.py")
    _run("-c", "import pandas as pd; from trend import data; from trend.v3.clean_model import save_clean_model; "
               "save_clean_model('results/v3/cache/clean_model.pt', pd.read_parquet(data.PROC/'returns.parquet'))")
    _run("scripts/v3/h2_cpcv.py")
    _run("scripts/v3/make_results_v3.py")      # not TREND_QUICK: predicts from the freshly retrained clean_model.pt
    print("FULL OK (compare with `git diff`; the committed artefacts are the reference)")


if __name__ == "__main__":
    cmds = {"quick": quick, "full": full}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
