import numpy as np
import pandas as pd

from trend import model
from trend.v2.family import family_configs


def test_family_has_96_unique_configs_including_v1():
    f = family_configs()
    names = [n for n, _ in f]
    assert len(f) == 96 and len(set(names)) == 96
    assert "tanh|b0.10|blend|com60" in names


def test_sign_signal_variant_and_default_unchanged():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-03", periods=700)
    R = pd.DataFrame(rng.normal(0.0003, 0.01, (700, 4)), index=idx, columns=list("abcd"))
    sector = pd.Series({"a": "x", "b": "x", "c": "y", "d": "y"})
    assert model.TrendConfig().signal == "tanh"
    w_t, _ = model.raw_positions(R, sector, model.TrendConfig())
    w_s, _ = model.raw_positions(R, sector, model.TrendConfig(signal="sign"))
    assert not np.allclose(w_t.to_numpy(), w_s.to_numpy())
    # sign positions are bounded by the unit position
    _, unit = model.raw_positions(R, sector, model.TrendConfig(signal="sign"))
    assert (w_s.abs() <= unit.abs() + 1e-12).all().all()
