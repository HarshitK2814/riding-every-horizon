import numpy as np
import pandas as pd

from trend.v3.tsfm import forecast_panel, schedule, tstat_signal


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
