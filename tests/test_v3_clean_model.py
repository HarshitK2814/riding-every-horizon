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
