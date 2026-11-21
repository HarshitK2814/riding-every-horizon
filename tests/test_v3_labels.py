import pandas as pd

from trend.v3.labels import period_label


def test_labels():
    assert period_label(pd.Timestamp("2010-05-03")) == "contaminated"
    assert period_label(pd.Timestamp("2020-03-02")) == "contaminated+seen"
    assert period_label(pd.Timestamp("2026-01-05")) == "seen"
    assert period_label(pd.Timestamp("2026-12-01")) == "forward"
