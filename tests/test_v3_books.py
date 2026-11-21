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


from trend.v3.books import build_v3_weights


def test_v1_weekly_only_changes_signal_every_5_days():
    R = _R(600)
    inp = {"R": R, "inst": _inst(), "cfg_d": {}}
    W = build_v3_weights(inp, tsfm_long=None, clean_full=None)
    assert {"v1_blend", "v1_weekly"} <= set(W)
    assert W["v1_weekly"].shape == W["v1_blend"].shape
    assert "tsfm_blend" not in W and "clean_blend" not in W


def test_weekly_signal_is_held_between_refresh_days():
    from trend.v3.books import weekly_signal
    R = model.prepare(_R(600))
    s = weekly_signal(R, model.TrendConfig(), every=5)
    blk = s.iloc[300:305]                      # 300 % 5 == 0: refresh day, then 4 held days
    assert (blk.iloc[1:] == blk.iloc[0]).all().all()
    assert not (s.iloc[305] == s.iloc[300]).all()


def test_tsfm_and_clean_books_are_added_when_given():
    R = _R(600)
    idx = R.index
    long = pd.DataFrame({"date": [idx[300]] * 4, "inst": list("ABCD"), "horizon": "short",
                         "mu": 0.0, "width": 1.0, "sig": [0.5, -0.5, 0.3, -0.3]})
    clean = pd.DataFrame(0.2, index=idx[300:], columns=list("ABCD"))
    W = build_v3_weights({"R": R, "inst": _inst(), "cfg_d": {}}, long, clean)
    assert {"v1_blend", "v1_weekly", "tsfm_blend", "clean_blend"} == set(W)
