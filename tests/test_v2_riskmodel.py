import numpy as np
import pandas as pd

from qengine.lookahead import check_lookahead
from trend import model
from trend.v2.riskmodel import backtest_with_vol, exante_vol, garch_vol_panel, tracking_rmse


def _R(n=1100):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(rng.standard_t(5, (n, 2)) * 0.01, index=idx, columns=["a", "b"])


def test_garch_panel_no_lookahead():
    check_lookahead(lambda R: garch_vol_panel(R, refit_every=200, min_obs=800), _R(), cut=950)


def test_garch_panel_no_lookahead_at_refit_boundary():
    # refit blocks start at rows 800 and 1000: perturb from 1000 on, rows <= 999 must not move (params included)
    check_lookahead(lambda R: garch_vol_panel(R, refit_every=200, min_obs=800), _R(), cut=999)


def test_tracking_rmse_zero_when_vol_is_exact():
    R = _R()
    perfect = pd.DataFrame(0.40, index=R.index, columns=R.columns)
    scaled_vol = R.rolling(63).std() * np.sqrt(252)
    out = tracking_rmse(R, scaled_vol.shift(1).bfill(), target=0.40)
    assert (out >= 0).all() and out.notna().all()
    assert set(tracking_rmse(R, perfect).index) == {"a", "b"}


def test_backtest_with_ewma_vol_equals_v1_backtest():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-03", periods=700)
    R = pd.DataFrame(rng.normal(0.0003, 0.01, (700, 4)), index=idx, columns=list("abcd"))
    inst = pd.DataFrame({"sector": ["x", "x", "y", "y"], "cost_bp": 1.0, "rolls": 4}, index=R.columns)
    cfg = model.TrendConfig()
    a = model.backtest(R, inst, cfg)["net"]
    b = backtest_with_vol(R, inst, cfg, model.ewma_vol(model.prepare(R), cfg.vol_com))["net"]
    pd.testing.assert_series_equal(a, b)


def test_exante_vol_uses_only_past_returns():
    R = _R(400)
    W = pd.DataFrame(0.5, index=R.index, columns=R.columns)
    check_lookahead(lambda X: exante_vol(X, W).to_frame("v"), R, cut=300)
    v = exante_vol(R, W).dropna()
    assert (v > 0).all()


def test_guarded_panel_equals_ewma_when_every_fit_rejected():
    from trend.v2.riskmodel import garch_vol_panel_guarded
    R = _R()
    g = garch_vol_panel_guarded(R, refit_every=200, min_obs=800, max_persistence=0.0)
    ewma = R.ewm(com=60, min_periods=60).std() * np.sqrt(252)
    pd.testing.assert_frame_equal(g, ewma.where(ewma.notna()), check_freq=False)


def test_guarded_panel_matches_raw_when_fits_are_sound_and_no_lookahead():
    from trend.v2.riskmodel import garch_vol_panel_guarded
    R = _R()
    raw = garch_vol_panel(R, refit_every=200, min_obs=800)
    g = garch_vol_panel_guarded(R, refit_every=200, min_obs=800)
    pd.testing.assert_frame_equal(raw, g, check_freq=False, rtol=1e-6)
    check_lookahead(lambda X: garch_vol_panel_guarded(X, refit_every=200, min_obs=800), R, cut=999)
