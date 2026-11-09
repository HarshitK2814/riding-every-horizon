import pandas as pd

from trend.v2.forward_log import append_new


def _rows(dates, book="v1_blend"):
    return pd.DataFrame({"date": pd.to_datetime(dates), "book": book, "net_return": 0.001, "prereg_hash": "h"})


def test_append_is_idempotent(tmp_path):
    p = tmp_path / "forward.csv"
    assert append_new(p, _rows(["2026-11-02", "2026-11-03"])) == 2
    assert append_new(p, _rows(["2026-11-02", "2026-11-03"])) == 0
    assert append_new(p, _rows(["2026-11-03", "2026-11-04"])) == 1
    df = pd.read_csv(p)
    assert len(df) == 3 and not df.duplicated(["date", "book"]).any()


def test_two_books_same_date_both_kept(tmp_path):
    p = tmp_path / "f.csv"
    append_new(p, pd.concat([_rows(["2026-11-02"]), _rows(["2026-11-02"], "garch_blend")]))
    assert len(pd.read_csv(p)) == 2


def test_empty_creates_header_only(tmp_path):
    p = tmp_path / "f.csv"
    assert append_new(p, _rows([]).iloc[0:0]) == 0
    assert list(pd.read_csv(p).columns) == ["date", "book", "net_return", "prereg_hash"]
