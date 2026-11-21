"""Period labels for v3 tables (see research/preregistration_v3.md, contamination rule)."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONTAM_END = pd.Timestamp("2025-09-30")
SEEN_START = pd.Timestamp("2018-01-01")
FORWARD_START = pd.Timestamp(yaml.safe_load((ROOT / "configs" / "splits_v3.yaml").read_text())["forward"][0])


def period_label(d: pd.Timestamp) -> str:
    if d >= FORWARD_START:
        return "forward"
    if d > CONTAM_END:
        return "seen"
    return "contaminated+seen" if d >= SEEN_START else "contaminated"
