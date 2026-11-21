import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from trend.v3.tsfm import TimesFMForecaster, extend_schedule, forecast_panel, schedule, tstat_signal


class DriftForecaster:
    """Fake model: extrapolates the mean daily step of the context; quantiles = point +/- 1."""
    def forecast(self, contexts, horizon):
        pts, qs = [], []
        for c in contexts:
            step = (c[-1] - c[0]) / max(len(c) - 1, 1)
            p = c[-1] + step * np.arange(1, horizon + 1)
            q = np.stack([p] + [p + z for z in np.linspace(-1.2816, 1.2816, 9)], axis=-1)
            pts.append(p); qs.append(q)
        return np.array(pts), np.array(qs)


def _panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    R = pd.DataFrame(rng.normal(0.0005, 0.01, (n, 3)), index=idx, columns=["A", "B", "C"])
    R.iloc[:300, 2] = np.nan        # C starts late
    return R


def test_tstat_signal_bounds_and_sign():
    s = tstat_signal(np.array([1.0, -1.0, 0.0]), np.array([2.5631, 2.5631, 1.0]))
    assert np.allclose(s, np.tanh([1.0, -1.0, 0.0]))


def test_schedule_every_5():
    idx = pd.bdate_range("2020-01-01", periods=12)
    assert list(schedule(idx, 5)) == [idx[0], idx[5], idx[10]]


def test_point_in_time_truncation_invariance():
    R = _panel()
    d = R.index[400]
    full = forecast_panel(R, pd.DatetimeIndex([d]), DriftForecaster(), {"short": 21})
    trunc = forecast_panel(R.loc[:d], pd.DatetimeIndex([d]), DriftForecaster(), {"short": 21})
    pd.testing.assert_frame_equal(full, trunc)


def test_future_shock_does_not_change_past_forecast():
    R = _panel()
    d = R.index[400]
    a = forecast_panel(R, pd.DatetimeIndex([d]), DriftForecaster(), {"short": 21})
    R2 = R.copy(); R2.iloc[401:] = 0.5
    b = forecast_panel(R2, pd.DatetimeIndex([d]), DriftForecaster(), {"short": 21})
    pd.testing.assert_frame_equal(a, b)


def test_min_obs_skips_young_instrument():
    R = _panel()
    out = forecast_panel(R, pd.DatetimeIndex([R.index[400]]), DriftForecaster(), {"short": 21}, min_obs=256)
    assert set(out["inst"]) == {"A", "B"}            # C has only 101 obs at index 400
    assert list(out.columns) == ["date", "inst", "horizon", "mu", "width", "sig"]


class NineQuantiles:
    def forecast(self, contexts, horizon):
        n = len(contexts)
        return np.zeros((n, horizon)), np.zeros((n, horizon, 9))


class Capped(DriftForecaster):
    max_horizon = 32


def test_wrong_quantile_layout_raises():
    R = _panel()
    with pytest.raises(ValueError, match="shape mismatch"):
        forecast_panel(R, pd.DatetimeIndex([R.index[400]]), NineQuantiles(), {"short": 21})


def test_horizon_over_capacity_raises():
    R = _panel()
    with pytest.raises(ValueError, match="capacity"):
        forecast_panel(R, pd.DatetimeIndex([R.index[400]]), Capped(), {"long": 63})


def test_multiple_horizons_two_rows_per_date_inst():
    R = _panel()
    out = forecast_panel(R, pd.DatetimeIndex([R.index[400]]), DriftForecaster(), {"short": 21, "medium": 63})
    assert len(out) == 2 * 2                      # A, B x two horizons
    assert out.groupby(["date", "inst"]).size().eq(2).all()
    assert set(out["horizon"]) == {"short", "medium"}


def test_batch_boundary_matches_single_batch():
    R = _panel()
    R.iloc[:, 2] = R.iloc[:, 1].to_numpy()        # make all 3 instruments eligible
    d = pd.DatetimeIndex([R.index[400], R.index[450]])
    a = forecast_panel(R, d, DriftForecaster(), {"short": 21, "medium": 63}, batch=2)
    b = forecast_panel(R, d, DriftForecaster(), {"short": 21, "medium": 63}, batch=512)
    pd.testing.assert_frame_equal(a, b)


def test_extend_schedule_anchors_on_cache_not_first_date():
    old_idx = pd.bdate_range("2010-01-01", periods=100)
    last = schedule(old_idx, 5)[-1]
    new_idx = pd.bdate_range("2010-01-11", periods=120)       # different first date
    got = extend_schedule(new_idx, last, 5)
    pos = new_idx.get_loc(last)
    assert list(got) == [new_idx[pos + 5 * k] for k in range(1, len(got) + 1)]
    assert len(got) == (len(new_idx) - 1 - pos) // 5
    assert len(extend_schedule(new_idx, new_idx[-1], 5)) == 0


@pytest.mark.skipif(os.environ.get("RUN_TSFM_TESTS") != "1" or importlib.util.find_spec("timesfm") is None,
                    reason="slow: set RUN_TSFM_TESTS=1 (needs timesfm + weights)")
def test_timesfm_cpu_forecast_shapes():
    fc = TimesFMForecaster(max_context=256, max_horizon=64, per_core_batch_size=4, device="cpu")
    assert fc.m.global_batch_size == 4
    rng = np.random.default_rng(0)
    ctx = [np.cumsum(rng.normal(0, 0.01, 300)), np.cumsum(rng.normal(0, 0.01, 200))]
    point, quant = fc.forecast(ctx, 21)
    assert point.shape == (2, 21) and quant.shape == (2, 21, 10)
    assert np.isfinite(point).all() and np.isfinite(quant).all()
