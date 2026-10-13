"""Time-ordered sample splits with a guard that keeps the final test period locked.

The test period may only be read once a pre-registration file exists and its hash is
exported as PREREG_HASH. This makes "I looked at the test set before freezing the
hypothesis" a code error rather than a matter of discipline.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class TestPeriodLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Splits:
    development: tuple[str, str]
    validation: tuple[str, str]
    test: tuple[str, str]
    recent: tuple[str, str | None]

    @classmethod
    def from_dict(cls, d: dict) -> "Splits":
        return cls(**{k: tuple(v) for k, v in d.items()})

    def bounds(self, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        # "2019" or "2019-06" as an end bound means the *end* of that year/month, inclusive.
        start, end = getattr(self, period)
        hi = pd.Period(end).end_time if end else pd.Timestamp.max
        return pd.Period(start).start_time, hi


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def guard(period: str, prereg_path: str | Path | None) -> None:
    """Raise unless `period` is unlocked. Only 'test' and 'recent' are locked."""
    if period not in {"test", "recent"}:
        return
    if prereg_path is None or not Path(prereg_path).exists():
        raise TestPeriodLocked(f"'{period}' is locked: no pre-registration file found.")
    expected = os.environ.get("PREREG_HASH")
    if expected != file_hash(prereg_path):
        raise TestPeriodLocked(
            f"'{period}' is locked: set PREREG_HASH={file_hash(prereg_path)} after committing "
            "the pre-registration (and do not edit it afterwards)."
        )


def select(obj: pd.DataFrame | pd.Series, splits: Splits, period: str,
           prereg_path: str | Path | None = None) -> pd.DataFrame | pd.Series:
    """Slice a date-indexed object to one period, enforcing the test-period lock."""
    guard(period, prereg_path)
    lo, hi = splits.bounds(period)
    idx = obj.index.get_level_values(0) if isinstance(obj.index, pd.MultiIndex) else obj.index
    return obj[(idx >= lo) & (idx <= hi)]
