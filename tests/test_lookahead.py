"""Changing data after date D must not change any weight decided at or before D."""

import numpy as np
import pandas as pd

from trend import model


def _toy(n=900, k=6, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    R = pd.DataFrame(rng.normal(0.0003, 0.01, (n, k)), index=idx, columns=[f"i{j}" for j in range(k)])
    inst = pd.DataFrame({"sector": ["a", "a", "b", "b", "c", "c"], "cost_bp": 1.0, "rolls": 4}, index=R.columns)
    return R, inst


def test_no_lookahead_in_weights_or_returns():
    R, inst = _toy()
    cfg = model.TrendConfig()
    base = model.backtest(R, inst, cfg)
    cut = R.index[600]
    shocked = R.copy()
    ramp = np.linspace(0.02, 0.2, (R.index > cut).sum())[:, None]
    shocked.loc[R.index > cut] += ramp                       # a ramp, so ratios also change
    alt = model.backtest(shocked, inst, cfg)
    pd.testing.assert_frame_equal(base["weights"].loc[:cut], alt["weights"].loc[:cut])
    # P&L of day cut+1 uses weights from cut (unchanged) times the shocked return: must differ
    nxt = R.index[601]
    assert base["net"].loc[:cut].equals(alt["net"].loc[:cut])
    assert base["gross"].loc[nxt] != alt["gross"].loc[nxt]


def test_signal_bounded_and_lagged():
    R, inst = _toy()
    s = model.horizon_signal(R, 21)
    assert s.abs().max().max() <= 1
    assert s.iloc[:20].isna().all().all()                     # needs a full window


def test_buffer_reduces_turnover():
    R, inst = _toy()
    a = model.backtest(R, inst, model.TrendConfig(buffer=0.0))
    b = model.backtest(R, inst, model.TrendConfig(buffer=0.10))
    assert b["weights"].diff().abs().sum().sum() < a["weights"].diff().abs().sum().sum()
