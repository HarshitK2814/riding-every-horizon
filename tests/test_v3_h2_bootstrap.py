import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("h2_cpcv", Path(__file__).resolve().parents[1] / "scripts/v3/h2_cpcv.py")
h2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(h2)


def _v1(n=1500, seed=0):
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0.0003, 0.01, n), index=idx)


def test_joint_bootstrap_null_when_clean_equals_v1():
    v1 = _v1()
    paths = pd.DataFrame({i: v1 for i in range(5)})
    dd, _, _ = h2.joint_path_bootstrap(paths, v1, n_boot=200)
    assert np.allclose(dd, 0) and np.mean(dd) == 0
    lo, hi = np.quantile(dd, [0.025, 0.975])
    assert lo <= 0 <= hi


def test_joint_bootstrap_detects_drift():
    v1 = _v1()
    rng = np.random.default_rng(1)
    paths = pd.DataFrame({i: v1 + 0.002 + rng.normal(0, 0.0005, len(v1)) for i in range(5)})
    dd, _, _ = h2.joint_path_bootstrap(paths, v1, n_boot=200)
    assert np.quantile(dd, 0.025) > 0
