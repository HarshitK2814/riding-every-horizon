"""Self-contained HTML tearsheet (PNG figures embedded as base64)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from qcore import metrics  # noqa: E402


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_tearsheet(net: pd.Series, path, title: str, periods: int = 252, costs: pd.DataFrame | None = None,
                    mc: pd.DataFrame | None = None, notes: list[str] | None = None) -> Path:
    r = net.dropna()
    eq = (1 + r).cumprod()
    figs = []
    fig, ax = plt.subplots(figsize=(9, 3)); ax.plot(eq.index, eq.values); ax.set_yscale("log")
    ax.set_title("Equity (log)"); figs.append(_png(fig))
    fig, ax = plt.subplots(figsize=(9, 2.5)); dd = metrics.drawdown(r); ax.fill_between(dd.index, dd.values, 0)
    ax.set_title("Drawdown"); figs.append(_png(fig))
    roll = r.rolling(periods).mean() / r.rolling(periods).std() * periods ** 0.5
    fig, ax = plt.subplots(figsize=(9, 2.5)); ax.plot(roll.index, roll.values); ax.axhline(0, lw=0.5)
    ax.set_title("Rolling 1-year Sharpe"); figs.append(_png(fig))
    summ = metrics.summary(r, periods).to_frame("value")
    summ.index = [str(i).replace("sharpe", "Sharpe") for i in summ.index]
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod().sub(1).unstack().mul(100).round(1)
    parts = [f"<h1>{title}</h1>", "<h2>Summary</h2>", summ.to_html(float_format=lambda v: f"{v:.4f}")]
    if costs is not None:
        parts += ["<h2>Costs (annualised, fraction of NAV)</h2>", (costs.mean() * periods).to_frame("ann").to_html()]
    if mc is not None:
        parts += ["<h2>Monte Carlo (stationary bootstrap)</h2>", mc.to_html(float_format=lambda v: f"{v:.3f}")]
    parts += [f'<img src="data:image/png;base64,{f}"/>' for f in figs]
    parts += ["<h2>Monthly returns (%)</h2>", monthly.to_html()]
    if notes:
        parts += ["<h2>Assumptions</h2><ul>"] + [f"<li>{n}</li>" for n in notes] + ["</ul>"]
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>" + title + "</title><style>"
            "body{font-family:system-ui;max-width:960px;margin:24px auto;padding:0 16px}"
            "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:2px 6px}"
            "</style></head><body>" + "".join(parts) + "</body></html>")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
