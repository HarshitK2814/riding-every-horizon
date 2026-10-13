"""Figures for the README and results. Reads results/runs (run `run.py test` first, with PREREG_HASH)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trend import analysis as A
from trend import data

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
BLUE, ORANGE, GREEN, RED, GREY = "#2f6fdf", "#e07b00", "#1a7f37", "#cf222e", "#8c959f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})
SPANS = {"development": ("2001-01-01", "2012-12-31"), "validation": ("2013-01-01", "2017-12-31"),
         "test": ("2018-01-01", "2024-12-31"), "recent": ("2025-01-01", "2026-12-31")}


def shade(ax):
    for name, c in (("validation", "#f6f8fa"), ("test", "#fff8c5"), ("recent", "#f6f8fa")):
        a, b = SPANS[name]
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, zorder=0)
        ax.text(pd.Timestamp(a), 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=MUTED)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    N = pd.read_parquet(ROOT / "results" / "runs" / "net_daily.parquet")
    bench = pd.read_parquet(data.PROC / "benchmarks.parquet")
    rf = data.tbill_daily().reindex(N.index).ffill().fillna(0)

    # 1. Growth of $1 (total return = excess + T-bill), vs the manager's fund and SPY
    fig, ax = plt.subplots(figsize=(9, 4.2))
    shade(ax)
    for k, c, lab in (("blend", BLUE, "Three-horizon blend"), ("long", ORANGE, "Long horizon only (252d)")):
        ax.plot((1 + N[k] + rf).cumprod(), color=c, lw=1.6, label=lab)
    eb = bench["EBSIX"].dropna()
    start = eb.index[0]
    base = float((1 + N["blend"] + rf).cumprod().loc[:start].iloc[-1])
    ax.plot(eb / eb.iloc[0] * base, color=INK, lw=1.2, ls="--", label="Manager's mutual fund (NAV, from 2013)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log), net of costs")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("Riding every horizon: 37 markets, six sectors, 10% vol target", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "growth.png")
    plt.close(fig)

    # 2. Crisis returns
    spy = bench["SPY"].pct_change(fill_method=None)
    wins = {"2008": ("2008-01-01", "2008-12-31"), "Q4 2018": ("2018-10-01", "2018-12-31"),
            "Feb-Mar 2020": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-01", "2022-12-31")}
    rows = {w: {"Trend blend": float((1 + N["blend"].loc[a:b] + rf.loc[a:b]).prod() - 1),
                "S&P 500 (SPY)": float((1 + spy.loc[a:b]).prod() - 1)} for w, (a, b) in wins.items()}
    df = pd.DataFrame(rows).T
    df.to_csv(ROOT / "results" / "tables" / "crisis_returns.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["Trend blend"] * 100, 0.4, color=BLUE, label="Trend blend")
    ax.bar(x + 0.2, df["S&P 500 (SPY)"] * 100, 0.4, color=GREY, label="S&P 500 (SPY)")
    for i, (a, b) in enumerate(df.to_numpy()):
        ax.text(i - 0.2, a * 100 + (1 if a > 0 else -3), f"{a:+.0%}", ha="center", fontsize=8.5, color=BLUE)
        ax.text(i + 0.2, b * 100 + (1 if b > 0 else -3), f"{b:+.0%}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, [f"{w}{' (dev)' if w == '2008' else ''}" for w in df.index])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Return, %")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Crisis returns: it paid when equities fell", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "crisis.png")
    plt.close(fig)

    # 3. Smile: monthly strategy vs SPY, test period
    m = A.period(pd.concat([A.monthly(N["blend"]), A.monthly(spy.dropna())], axis=1).dropna(), "test")
    m.columns = ["s", "m"]
    b = np.polyfit(m["m"], m["s"], 2)
    xs = np.linspace(m["m"].min(), m["m"].max(), 100)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.scatter(m["m"] * 100, m["s"] * 100, s=14, color=BLUE, alpha=0.7)
    ax.plot(xs * 100, np.polyval(b, xs) * 100, color=ORANGE, lw=1.8)
    ax.set_xlabel("S&P 500 monthly return, %")
    ax.set_ylabel("Trend blend monthly return, %")
    ax.set_title("The trend 'smile', test 2018-24 (convexity t = 3.6)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "smile.png")
    plt.close(fig)

    # 4. Sharpe by book, dev+val vs test
    books = ["blend", "short", "medium", "long", "disclosed_mix", "sector_equity", "sector_rates", "sector_fx",
             "sector_metals", "sector_energy", "sector_ags"]
    sh = {}
    for k in books:
        mm = A.monthly(N[k])
        sh[k] = {"dev+val": A.sharpe_m(pd.concat([A.period(mm, "development"), A.period(mm, "validation")])),
                 "test": A.sharpe_m(A.period(mm, "test"))}
    sh = pd.DataFrame(sh).T
    sh.to_csv(ROOT / "results" / "tables" / "sharpe_by_book.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(sh))
    ax.bar(x - 0.2, sh["dev+val"], 0.4, color=GREY, label="Development + validation 2001-17")
    ax.bar(x + 0.2, sh["test"], 0.4, color=BLUE, label="Locked test 2018-24")
    ax.set_xticks(x, [k.replace("sector_", "").replace("_", "\n") for k in sh.index], fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Net Sharpe (monthly)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Net Sharpe by book", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "sharpe_by_book.png")
    plt.close(fig)
    print(df.round(3), sh.round(2), sep="\n")


if __name__ == "__main__":
    main()
