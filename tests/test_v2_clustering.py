import numpy as np
import pandas as pd

from trend.v2.clustering import crash_events, monthly_intensity


def test_crash_event_uses_lagged_sigma():
    idx = pd.bdate_range("2010-01-01", periods=200)
    r = pd.Series(0.001 * np.sin(np.arange(200)), index=idx)
    r.iloc[150] = -0.05
    e = crash_events(r)
    assert e.iloc[150] and e.sum() == 1


def test_intensity_ignores_future_events():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2001-01-01", periods=3000)
    ev = pd.Series(rng.uniform(size=3000) < 0.03, index=idx)
    a = monthly_intensity(ev)
    ev2 = ev.copy(); ev2.iloc[2500:] = True
    b = monthly_intensity(ev2)
    cut = idx[2500]
    pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])
