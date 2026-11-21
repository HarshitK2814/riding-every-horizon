import numpy as np
import pandas as pd
import pytest

from trend.v3.clean_model import clean_signal, features, fit_predict, targets

H = {"short": 21, "medium": 63, "long": 252}


def _R(n=900, seed=3, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(rng.normal(drift, 0.01, (n, 3)), index=idx, columns=["A", "B", "C"])


def test_features_are_point_in_time():
    R = _R()
    d = R.index[500]
    a = features(R).loc[d]
    R2 = R.copy(); R2.iloc[501:] = 0.3
    b = features(R2).loc[d]
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == [f"m{w}" for w in (1, 5, 10, 21, 42, 63, 126, 189, 252)] + ["logvol"]


def test_targets_use_only_future():
    R = _R()
    Y = targets(R, H)
    ti, h = 300, 21
    d = R.index[ti]
    base = Y.loc[(d, "A"), "short"]
    # (a) returns strictly after t+h must not matter
    R2 = R.copy(); R2.iloc[ti + h + 1:] = 0.3
    assert targets(R2, H).loc[(d, "A"), "short"] == pytest.approx(base, abs=1e-9)
    # (b) the return at t+1 must matter
    R3 = R.copy(); R3.iloc[ti + 1, 0] += 0.05
    assert targets(R3, H).loc[(d, "A"), "short"] != base
    # (c) past (incl. t) only moves the denominator: sign equals sign of the t+1..t+h sum
    R4 = R.copy(); R4.iloc[: ti + 1] *= 3.0
    expect = np.log1p(R).iloc[ti + 1: ti + h + 1].sum()["A"]
    assert np.sign(base) == np.sign(expect)
    assert np.sign(targets(R4, H).loc[(d, "A"), "short"]) == np.sign(expect)


def test_fit_predict_learns_obvious_trend_and_is_bounded():
    R = _R(drift=0.002)
    X, Y = features(R).dropna(), targets(R, H).dropna()
    common = X.index.intersection(Y.index)
    tr, te = common[: len(common) // 2], common[len(common) // 2:]
    P = fit_predict(X.loc[tr], Y.loc[tr], X.loc[te], epochs=50, device="cpu")
    assert P.abs().max().max() <= 1.0
    assert (P.mean() > 0).all()                 # positive drift -> long signal
    S = clean_signal(P)
    assert set(S.columns) == {"A", "B", "C"}
    P2 = fit_predict(X.loc[tr], Y.loc[tr], X.loc[te], epochs=50, device="cpu")
    pd.testing.assert_frame_equal(P, P2)        # deterministic for a given seed


def test_features_flat_series_has_no_inf():
    R = _R(); R["A"] = 0.0
    assert not np.isinf(features(R).to_numpy()).any()


def test_clean_full_training_set_uses_no_post_train_end_returns():
    from trend.v3.clean_model import _train_xy
    R = _R(900)
    te = R.index[600]
    X, Y = _train_xy(R, str(te.date()))
    R2 = R.copy(); R2.iloc[601:] = 0.3
    X2, Y2 = _train_xy(R2, str(te.date()))
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_frame_equal(Y, Y2)
    last = X.index.get_level_values(0).max()
    assert list(R.index).index(last) <= 600 - 252          # rows within 252 days of train_end dropped


def test_clean_full_signal_is_point_in_time_and_deterministic():
    from trend.v3.clean_model import clean_full_signal
    R = _R(900)
    te = str(R.index[600].date())
    a = clean_full_signal(R, train_end=te, epochs=2)
    assert a.index.min() > pd.Timestamp(te)
    R2 = R.copy(); R2.iloc[700:] = 0.2           # changes after 700 must not move predictions before 700
    b = clean_full_signal(R2, train_end=te, epochs=2)
    pd.testing.assert_frame_equal(a.loc[:R.index[699]], b.loc[:R.index[699]])
    c = clean_full_signal(R, train_end=te, epochs=2)
    pd.testing.assert_frame_equal(a, c)


def test_clean_predict_roundtrip_matches_full_signal_and_is_point_in_time(tmp_path):
    from trend.v3.clean_model import clean_predict, save_clean_model
    R = _R(900)
    te = str(R.index[600].date())
    p = tmp_path / "m.pt"
    save_clean_model(p, R, train_end=te, epochs=2, device="cpu")
    a = clean_predict(p, R)
    assert a.index.min() > pd.Timestamp(te)
    from trend.v3.clean_model import _test_features, _train_xy
    Xtr, Ytr = _train_xy(R, te)
    full = clean_signal(fit_predict(Xtr, Ytr, _test_features(R, te), epochs=2, device="cpu"))
    assert a.index.equals(full.index)
    assert float((a - full).abs().max().max()) < 1e-4
    R2 = R.copy(); R2.iloc[700:] = 0.2
    b = clean_predict(p, R2)
    pd.testing.assert_frame_equal(a.loc[:R.index[699]], b.loc[:R.index[699]])
