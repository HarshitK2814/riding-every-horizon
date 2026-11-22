import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import importlib.util

from trend.cli import V2_SCRIPTS, compare_frames, compare_json, v2_scripts

ROOT = Path(__file__).resolve().parents[1]


def test_compare_frames_tolerance_and_text():
    a = pd.DataFrame({"x": [1.0, float("nan")], "t": ["a", None]})
    compare_frames(a, a.assign(x=[1.0 + 1e-8, float("nan")]), "t")
    with pytest.raises(AssertionError):
        compare_frames(a, a.assign(x=[1.001, float("nan")]), "t")
    with pytest.raises(AssertionError):
        compare_frames(a, a.assign(t=["b", None]), "t")


def test_compare_json():
    compare_json({"a": [1.0, {"b": 2.0}], "s": "x"}, {"a": [1.0 + 1e-9, {"b": 2.0}], "s": "x"}, "j")
    with pytest.raises(AssertionError):
        compare_json({"a": 1.0}, {"a": 1.1}, "j")


def test_v2_scripts_skips_pysr_when_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert "e2_pysr" not in v2_scripts(importlib.util.find_spec("pysr") is not None)
    assert v2_scripts(False) == [s for s in V2_SCRIPTS if s != "e2_pysr"]


def test_v2_scripts_keeps_pysr_when_present():
    assert v2_scripts(True) == V2_SCRIPTS


@pytest.mark.skipif(not (ROOT / "snapshot/processed/returns.parquet").exists(), reason="no snapshot")
def test_quick_runs_and_matches():
    r = subprocess.run([sys.executable, "-m", "trend.cli", "quick"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "QUICK OK" in r.stdout


def test_compare_frames_per_column_tolerance():
    import pandas as pd
    from trend.cli import compare_frames
    ref = pd.DataFrame({"n_obs": [100], "hit_rate": [0.5000]})
    compare_frames(ref, pd.DataFrame({"n_obs": [99], "hit_rate": [0.5004]}), "t", atol={"n_obs": 2, "*": 1e-3})
    with pytest.raises(AssertionError):
        compare_frames(ref, pd.DataFrame({"n_obs": [96], "hit_rate": [0.5]}), "t", atol={"n_obs": 2, "*": 1e-3})
