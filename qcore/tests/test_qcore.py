import numpy as np
import pandas as pd
import pytest

from qcore import bootstrap, metrics, splits


@pytest.fixture
def monthly():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    return pd.Series(rng.normal(0.01, 0.04, len(idx)), index=idx)


def test_sharpe_matches_definition(monthly):
    expected = monthly.mean() / monthly.std(ddof=1) * np.sqrt(12)
    assert metrics.sharpe(monthly, 12) == pytest.approx(expected)


def test_max_drawdown_simple_path():
    r = pd.Series([0.10, -0.50, 0.20])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_cvar_is_at_least_var(monthly):
    var, cvar = metrics.var_cvar(monthly)
    assert cvar >= var


def test_newey_west_equals_plain_t_with_zero_lags(monthly):
    plain = monthly.mean() / (monthly.std(ddof=0) / np.sqrt(len(monthly)))
    assert metrics.newey_west_tstat(monthly, 0) == pytest.approx(plain)


def test_bootstrap_ci_covers_sample_mean(monthly):
    draws = bootstrap.bootstrap_stat(monthly, lambda x: x.mean(), n_boot=500)
    lo, hi = bootstrap.percentile_ci(draws)
    assert lo < monthly.mean() < hi


def test_test_period_is_locked(tmp_path, monthly, monkeypatch):
    s = splits.Splits.from_dict({"development": ["2000", "2009"], "validation": ["2010", "2014"],
                                 "test": ["2015", "2019"], "recent": ["2020", None]})
    assert len(splits.select(monthly, s, "development")) > 0
    with pytest.raises(splits.TestPeriodLocked):
        splits.select(monthly, s, "test", prereg_path=tmp_path / "missing.md")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen")
    with pytest.raises(splits.TestPeriodLocked):
        splits.select(monthly, s, "test", prereg_path=prereg)
    monkeypatch.setenv("PREREG_HASH", splits.file_hash(prereg))
    assert len(splits.select(monthly, s, "test", prereg_path=prereg)) == 60
