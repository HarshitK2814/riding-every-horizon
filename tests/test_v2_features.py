import numpy as np
import pandas as pd

from qengine.lookahead import check_lookahead
from trend.v2.features import feature_panel


def test_features_no_lookahead():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2005-01-03", periods=900)
    R = pd.DataFrame(rng.normal(0, 0.01, (900, 3)), index=idx, columns=list("abc"))
    wide = lambda R: feature_panel(R, step=1)["t63"].unstack()
    check_lookahead(wide, R, cut=700)


def test_target_is_forward():
    idx = pd.bdate_range("2005-01-03", periods=400)
    calm = np.tile([0.001, -0.001], 150)                      # 21-day sums are +-0.001: weak trend
    R = pd.DataFrame({"a": np.r_[calm, np.full(100, 0.01)]}, index=idx)
    f = feature_panel(R, step=1)
    row = f.loc[(idx[290], "a")]
    assert row["y"] > 0                                       # label sees the coming rally
    assert abs(row["t21"]) < 0.5                              # feature does not


def test_aci_delayed_only_uses_realised_labels():
    from trend.v2.features import aci_delayed
    rng = np.random.default_rng(1)
    n, d, t = 400, 5, 300
    y = pd.Series(rng.normal(size=n)); yhat = pd.Series(np.zeros(n))
    base = aci_delayed(y, yhat, delay=d, window=100)
    for k in range(1, d):                                     # labels still unrealised at row t
        y2 = y.copy(); y2.iloc[t - k] += 50.0
        pd.testing.assert_series_equal(base.loc[t, ["lo", "hi"]], aci_delayed(y2, yhat, delay=d, window=100).loc[t, ["lo", "hi"]])
    y3 = y.copy(); y3.iloc[t - d] += 50.0                     # realised label must matter
    assert aci_delayed(y3, yhat, delay=d, window=100).loc[t, "hi"] != base.loc[t, "hi"]
