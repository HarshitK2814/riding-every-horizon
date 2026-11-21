import pandas as pd

from trend.v2.forward_log import append_new


def test_append_new_is_idempotent(tmp_path):
    p = tmp_path / "f.csv"
    rows = pd.DataFrame({"date": pd.to_datetime(["2026-12-01", "2026-12-02"]), "book": "tsfm_blend",
                         "net_return": [0.01, -0.02], "prereg_hash": "abc"})
    assert append_new(p, rows) == 2
    assert append_new(p, rows) == 0


def test_update_signals_never_writes_reference_cache(tmp_path, monkeypatch):
    import hashlib
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts" / "v3"))
    sys.modules.pop("forward", None)
    import forward

    ref, fwd = tmp_path / "cache", tmp_path / "forward"
    ref.mkdir()
    for n in ("tsfm_signal.parquet", "clean_signal_full.parquet"):
        pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "x": [1.0]}).to_parquet(ref / n)
    (ref / "clean_model.pt").write_bytes(b"frozen")
    snap = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in ref.iterdir()}

    def fake_forecast(base, out_path):
        old = pd.read_parquet(base)
        new = pd.DataFrame({"date": pd.to_datetime(["2026-01-09"]), "x": [2.0]})
        pd.concat([old, new], ignore_index=True).to_parquet(out_path)

    fake_predict = lambda path, R: pd.DataFrame({"a": [0.5]})
    forward.update_signals(pd.DataFrame(), ref, fwd, forecast=fake_forecast, predict=fake_predict)

    assert snap == {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in ref.iterdir()}
    assert set(p.name for p in ref.iterdir()) == set(snap)
    clean_p, tsfm_p = forward.live_paths(fwd)
    assert clean_p.exists() and len(pd.read_parquet(tsfm_p)) == 2
