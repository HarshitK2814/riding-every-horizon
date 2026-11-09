import numpy as np
import pandas as pd

from trend.v2.audit import impute_flagged, roll_gap_flags


def test_flags_injected_jumps_only():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    proxy = pd.Series(rng.normal(0, 0.01, 1500), index=idx)
    fut = proxy + rng.normal(0, 0.001, 1500)
    jumps = idx[[400, 800, 1200]]
    fut[jumps] += 0.05
    f = roll_gap_flags(fut, proxy)
    assert set(f[f].index) == set(jumps)


def test_impute_uses_sector_median():
    idx = pd.bdate_range("2020-01-01", periods=2)
    R = pd.DataFrame({"a": [0.5, 0.0], "b": [0.01, 0.0], "c": [0.03, 0.0]}, index=idx)
    flags = pd.DataFrame(False, index=idx, columns=R.columns); flags.iloc[0, 0] = True
    out = impute_flagged(R, flags, pd.Series({"a": "s", "b": "s", "c": "s"}))
    assert out.iloc[0, 0] == 0.02 and out.iloc[0, 1] == 0.01
