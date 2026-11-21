import numpy as np
import pandas as pd

from trend import model
from trend.v3.books import signal_frame


def _R(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=idx, columns=list("ABCD"))


def _inst():
    return pd.DataFrame({"sector": ["x", "x", "y", "y"], "cost_bp": 2.0, "rolls": 4}, index=list("ABCD"))


def test_sig_override_reproduces_v1_when_given_v1_signal():
    R = model.prepare(_R())
    cfg = model.TrendConfig()
    names = tuple(cfg.horizons)
    v1sig = sum(model.horizon_signal(R, cfg.horizons[n], cfg.vol_com) for n in names) / len(names)
    a = model.backtest(R, _inst(), cfg)["net"]
    b = model.backtest(R, _inst(), cfg, sig=v1sig)["net"]
    pd.testing.assert_series_equal(a, b)


def test_signal_frame_holds_at_most_max_hold_days():
    idx = pd.bdate_range("2020-01-01", periods=12)
    long = pd.DataFrame({"date": [idx[0]] * 2, "inst": ["A", "A"], "horizon": ["short", "long"],
                         "mu": 0.0, "width": 1.0, "sig": [0.2, 0.4]})
    S = signal_frame(long, idx, max_hold=5)
    assert np.allclose(S["A"].iloc[:5], 0.3)
    assert S["A"].iloc[5:].isna().all()
