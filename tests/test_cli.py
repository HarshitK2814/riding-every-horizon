import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from trend.cli import compare_frames, compare_json

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


@pytest.mark.skipif(not (ROOT / "snapshot/processed/returns.parquet").exists(), reason="no snapshot")
def test_quick_runs_and_matches():
    r = subprocess.run([sys.executable, "-m", "trend.cli", "quick"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "QUICK OK" in r.stdout
