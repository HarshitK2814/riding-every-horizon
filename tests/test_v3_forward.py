import pandas as pd

from trend.v2.forward_log import append_new


def test_append_new_is_idempotent(tmp_path):
    p = tmp_path / "f.csv"
    rows = pd.DataFrame({"date": pd.to_datetime(["2026-12-01", "2026-12-02"]), "book": "tsfm_blend",
                         "net_return": [0.01, -0.02], "prereg_hash": "abc"})
    assert append_new(p, rows) == 2
    assert append_new(p, rows) == 0
