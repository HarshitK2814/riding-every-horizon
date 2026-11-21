"""P7 v3 Task 6: v3 books, descriptive/exploratory E1-E3 and research/results_v3.md.
Signal at close t, trade t+1 (lag=1). Costs: v1 cost_bp + roll. Sharpe 95% stationary-bootstrap CI
(mean block 20, 2000 draws, seed 0). Everything here is descriptive/exploratory: TimesFM history before
2025-10-01 is `contaminated`, 2018-01.. is `seen`. Only the forward period is confirmatory.
TREND_QUICK=1 loads results/v3/cache/clean_signal_full.parquet; the full run predicts from the frozen clean_model.pt (never retrains).
Run: python scripts/v3/make_results_v3.py"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.splits import Splits, file_hash, select
from qengine.capacity import capacity_at, capacity_curve
from qengine.costs import CostModel
from qengine.engine import run_weights
from qengine.market import market_inputs
from trend import model
from trend.v2.books import load_inputs
from trend.v3.books import build_v3_weights
from trend.v3.clean_model import clean_predict_pred, clean_signal
from trend.v3.labels import period_label

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG, CACHE = ROOT / "results/v3/tables", ROOT / "results/v3/figures", ROOT / "results/v3/cache"
for d in (TAB, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)
PREREG = ROOT / "research" / "preregistration_v3.md"
os.environ["PREREG_HASH"] = file_hash(PREREG)
BOOKS = ["v1_blend", "v1_weekly", "tsfm_blend", "clean_blend"]
PERIODS = ["development", "validation", "test", "recent"]
CUTOFF = pd.Timestamp("2025-10-01")
BOOT, BLOCK = 2000, 20
HEADER = "Signal at close t, trade t+1; v1 costs (cost_bp + roll); Sharpe 95% stationary-bootstrap CI"
CRISES = {"2018Q4": ("2018-10-01", "2018-12-31"), "covid-2020": ("2020-02-19", "2020-03-23"),
          "2022": ("2022-01-01", "2022-12-31")}


def sh(r) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan


def sharpe_ci(r: pd.Series):
    lo, hi = percentile_ci(bootstrap_stat(r, sh, n_boot=BOOT, mean_block=BLOCK, seed=0))
    return sh(r), lo, hi


def labels_of(idx) -> str:
    out = []
    for d in idx:
        lab = period_label(d)
        if lab not in out:
            out.append(lab)
    return " / ".join(out)


def maxdd(r: pd.Series) -> float:
    c = (1 + r).cumprod()
    return float((c / c.cummax() - 1).min())


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main() -> None:
    inp = load_inputs(ROOT)
    R = model.prepare(inp["R"])
    tsfm_long = pd.read_parquet(CACHE / "tsfm_signal.parquet")
    pred_p, sig_p = CACHE / "clean_pred_full.parquet", CACHE / "clean_signal_full.parquet"
    if os.environ.get("TREND_QUICK") == "1":
        clean_full = pd.read_parquet(sig_p)
        clean_pred = pd.read_parquet(pred_p)
    else:
        # frozen committed model (never retrained): aligns the full run with results/v3/cache/clean_model.pt
        clean_pred = clean_predict_pred(CACHE / "clean_model.pt", R)
        clean_full = clean_signal(clean_pred)
        clean_full.to_parquet(sig_p)
        clean_pred.to_parquet(pred_p)
    W = build_v3_weights(inp, tsfm_long, clean_full)
    base_cost = CostModel.from_instruments(inp["inst"], 0.0)
    res, net, live = {}, {}, {}
    for b, w in W.items():
        Rp = R.reindex(columns=w.columns).fillna(0.0)
        res[b] = run_weights(w, Rp, base_cost, lag=1)
        live[b] = w.index[(w.abs().sum(axis=1) > 0).to_numpy()][0]
        net[b] = res[b].net.loc[live[b]:]
    # parity: engine lag-1 net for v1_blend must equal the v1 backtest net (as scripts/v2/execution.py)
    v1_bt = model.backtest(inp["R"], inp["inst"], model.TrendConfig.from_yaml(inp["cfg_d"]))["net"]
    cd = res["v1_blend"].net.index.intersection(v1_bt.index)
    assert len(cd) > 1000 and np.allclose(res["v1_blend"].net.loc[cd].to_numpy(), v1_bt.loc[cd].to_numpy(),
                                          atol=1e-10, rtol=0), "engine lag1 != v1 backtest net"
    splits = Splits.from_dict(yaml.safe_load((ROOT / "configs/splits_v3.yaml").read_text()))

    def sel(x, period):
        return select(x, splits, period, PREREG)

    # ---- summary_by_period
    rows = []
    extra = {"recent (to 2025-09-30)": lambda x: sel(x, "recent").loc[:CUTOFF - pd.Timedelta(days=1)],
             "recent (from 2025-10-01)": lambda x: sel(x, "recent").loc[CUTOFF:]}
    for b in BOOKS:
        for p in PERIODS + list(extra):
            r = sel(net[b], p) if p in PERIODS else extra[p](net[b])
            if len(r) == 0:
                rows.append(dict(book=b, period=p, label="", days=0,
                                 note="not computed: signal starts after this period"))
                continue
            s, lo, hi = sharpe_ci(r)
            rr = res[b]
            rows.append(dict(book=b, period=p, label=labels_of(r.index), start=r.index[0].date(),
                             end=r.index[-1].date(), days=len(r), ann_ret=float(r.mean() * 252),
                             ann_vol=float(r.std(ddof=1) * np.sqrt(252)), sharpe=s, sharpe_lo=lo, sharpe_hi=hi,
                             maxdd=maxdd(r), turnover=float(rr.turnover.reindex(r.index).mean() * 252),
                             cost_ann=float(rr.costs.sum(axis=1).reindex(r.index).mean() * 252),
                             signal_bar="close t, trade t+1", costs="v1 cost_bp + roll", note=""))
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB / "summary_by_period.csv", index=False)

    # ---- crisis
    cr = []
    for b in BOOKS:
        for cn, (a, z) in CRISES.items():
            r = net[b].loc[a:z]
            ok = len(r) > 0 and net[b].index[0] <= pd.Timestamp(a)
            cr.append(dict(book=b, window=cn, start=a, end=z, label=labels_of(pd.bdate_range(a, z)),
                           days=len(r) if ok else 0,
                           cum_net_return=float((1 + r).prod() - 1) if ok else np.nan,
                           signal_bar="close t, trade t+1", costs="v1 cost_bp + roll"))
    crisis = pd.DataFrame(cr)
    crisis.to_csv(TAB / "crisis.csv", index=False)

    # ---- E2 hit rates (sign of short-horizon signal vs realised t+1..t+21 log return)
    lr = np.log1p(R)
    fwd = lr[::-1].rolling(21, min_periods=21).sum()[::-1].shift(-1)
    fdates = pd.DatetimeIndex(sorted(tsfm_long["date"].unique()))
    sig_s = {
        "tsfm_short": tsfm_long[tsfm_long["horizon"] == "short"].set_index(["date", "inst"])["sig"].unstack("inst"),
        "clean_short": clean_pred["short"].unstack("inst"),
        "v1_sign_short": np.sign(model.horizon_signal(R, 21, 60)),
    }
    bands = {"pre-2018 (contaminated)": (None, pd.Timestamp("2017-12-31")),
             "2018-01..2025-09 (contaminated+seen)": (pd.Timestamp("2018-01-01"), pd.Timestamp("2025-09-30")),
             "2025-10..2026-09 (seen, after cutoff)": (CUTOFF, None)}
    # common observation set: a (date, inst) cell counts within a band only if tsfm, v1 and clean (where it has predictions) are all valid and non-zero
    cdates = fdates.intersection(sig_s["v1_sign_short"].index)
    cols = sig_s["tsfm_short"].columns
    F = fwd.reindex(index=cdates, columns=cols)
    Sc = {k: v.reindex(index=cdates, columns=cols) for k, v in sig_s.items()}
    okc_s = {k: (v.notna() & (v != 0) & F.notna() & (F != 0)).to_numpy() for k, v in Sc.items()}
    hr = []
    for bn, (a, z) in bands.items():
        m = pd.Series(True, index=cdates)
        if a is not None:
            m &= cdates >= a
        if z is not None:
            m &= cdates <= z
        mrow = m.to_numpy()[:, None]
        # signals with no data at all in this band (clean before 2018) drop out of the intersection
        use = [k for k in Sc if (okc_s[k] & mrow).any()]
        okc = np.logical_and.reduce([okc_s[k] for k in use])
        for sn, S in Sc.items():
            mm = (mrow & okc) if sn in use else np.zeros_like(okc)
            s_, f_ = S.to_numpy()[mm], F.to_numpy()[mm]
            ok = np.ones(len(s_), dtype=bool)
            n = int(ok.sum())
            k = int((np.sign(s_[ok]) == np.sign(f_[ok])).sum())
            lo, hi = wilson(k, n)
            lab = bn[bn.index("(") + 1:-1]
            hr.append(dict(signal=sn, band=bn, label=lab, n_obs=n, hit_rate=k / n if n else np.nan,
                           wilson_lo=lo, wilson_hi=hi, signal_bar="close t, forecast t+1..t+21",
                           note="obs overlap and are cross-correlated: CI too narrow"))
    hr = pd.DataFrame(hr)
    hr.to_csv(TAB / "e2_hitrate.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bb = list(bands)
    for i, sn in enumerate(sig_s):
        q = hr[(hr.signal == sn) & hr.band.isin(bb)].set_index("band").reindex(bb)
        x = np.arange(len(bb)) + i * 0.2
        ax.errorbar(x, q["hit_rate"], yerr=[q["hit_rate"] - q["wilson_lo"], q["wilson_hi"] - q["hit_rate"]],
                    fmt="o", capsize=3, label=sn)
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_xticks(np.arange(len(bb)) + 0.2)
    ax.set_xticklabels([b.replace(" (", "\n(") for b in bb], fontsize=7)
    ax.set_ylabel("hit rate (sign vs realised t+1..t+21)")
    ax.set_title("E2 (exploratory): hit rate before/after TimesFM cutoff; Wilson 95% (too narrow)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e2_memorisation.png", dpi=130)
    plt.close(fig)

    # ---- E3 costs x2 and capacity (2018-01..2026-10, lag 2 as v2 execution.py)
    e3 = []
    for b in ("tsfm_blend", "v1_blend"):
        Rb = R.reindex(columns=W[b].columns).fillna(0.0)
        for mult in (1, 2):
            rr = run_weights(W[b], Rb, base_cost.scaled(mult), lag=1).net.loc[live[b]:]
            for pn, r in (("development", sel(rr, "development")), ("2018-01..2026-10", rr.loc["2018-01-01":])):
                s, lo, hi = sharpe_ci(r)
                e3.append(dict(book=b, what=f"cost x{mult}", period=pn, label=labels_of(r.index), days=len(r),
                               sharpe=s, sharpe_lo=lo, sharpe_hi=hi, signal_bar="close t, trade t+1",
                               costs=f"v1 cost_bp + roll, x{mult}"))
    Rp = R.reindex(columns=W["v1_blend"].columns).fillna(0.0)
    mi_p = market_inputs(Rp, inp["close"], inp["volume"], inp["fx"])
    med = mi_p["adv_usd"].loc["2001-01-01":"2017-12-31"].median()
    thin = sorted(med[med < 50e6].index)
    mi_lo = {**mi_p, "adv_usd": mi_p["adv_usd"].copy()}
    mi_lo["adv_usd"][thin] = np.nan
    grid = [1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 2.5e9, 5e9]
    sl = slice(pd.Timestamp("2018-01-01"), None)
    cap_rows = []
    for b in ("tsfm_blend", "v1_blend"):
        for vn, m_ in {"primary_yahoo_adv": mi_p, "low_adv_instruments_impact_off": mi_lo}.items():
            for coef in (0.25, 0.5, 1.0):
                cm = CostModel.from_instruments(inp["inst"], impact_coef=coef)
                cur = capacity_curve(W[b].loc[sl], Rp.loc[sl], cm, m_["adv_usd"].loc[sl], m_["sigma_daily"].loc[sl],
                                     grid, lag=2)
                cap = capacity_at(cur, 0.10)
                for _, c in cur.iterrows():
                    cap_rows.append(dict(book=b, adv_variant=vn, impact_coef=coef, aum=c["aum"],
                                         net_sharpe_lag2=c["net_sharpe"], capacity_aum_drop0p10=cap,
                                         period="2018-01..2026-10", label="contaminated+seen / seen",
                                         days=len(W[b].loc[sl]), signal_bar="close t, trade t+2 (lag 2)",
                                         costs="v1 cost_bp + roll + sqrt impact"))
    e3 = pd.DataFrame(e3)
    cap_df = pd.DataFrame(cap_rows)
    pd.concat([e3.assign(table="cost_multiplier"), cap_df.assign(table="capacity")],
              ignore_index=True).to_csv(TAB / "e3_costs.csv", index=False)

    # ---- growth figure
    fig, ax = plt.subplots(figsize=(10, 5))
    for b in BOOKS:
        ax.plot(np.log((1 + net[b]).cumprod()), label=b, lw=1)
    end = net["v1_blend"].index[-1]
    ax.axvspan(pd.Timestamp("2001-01-01"), pd.Timestamp("2017-12-31"), color="red", alpha=0.05,
               label="contaminated (TimesFM)")
    ax.axvspan(pd.Timestamp("2018-01-01"), CUTOFF, color="orange", alpha=0.12, label="contaminated+seen")
    ax.axvspan(CUTOFF, end, color="blue", alpha=0.10, label="seen (after cutoff)")
    ax.set_ylabel("log growth of 1 (net, from each book's first live day)")
    ax.set_title("P7 v3 books: signal at close t, trade t+1; descriptive only, no forward data yet")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "growth_v3.png", dpi=130)
    plt.close(fig)

    write_md(summ, crisis, hr, e3, cap_df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.drop(columns=["signal_bar", "costs", "note"], errors="ignore").round(3).to_string())
    print(hr.drop(columns=["note", "signal_bar"]).round(3).to_string())
    print(e3.drop(columns=["signal_bar", "costs"]).round(3).to_string())


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(fmt.get(c, "{}").format(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_md(summ, crisis, hr, e3, cap):
    h2 = json.loads((ROOT / "results/v3/runs/h2.json").read_text())
    s_cols = ["book", "period", "label", "days", "ann_ret", "ann_vol", "sharpe", "sharpe_lo", "sharpe_hi",
              "maxdd", "turnover", "cost_ann"]
    sf = {"ann_ret": "{:.1%}", "ann_vol": "{:.1%}", "sharpe": "{:.2f}", "sharpe_lo": "{:.2f}", "sharpe_hi": "{:.2f}",
          "maxdd": "{:.1%}", "turnover": "{:.1f}", "cost_ann": "{:.2%}", "days": "{:d}"}
    S = summ.copy()
    S["days"] = S["days"].astype(int)
    S.loc[S["days"] == 0, "label"] = "not computed"
    tbl_s = md_table(S, s_cols, sf)
    cr = crisis.copy()
    cr["days"] = cr["days"].astype(int)
    tbl_c = md_table(cr, ["book", "window", "label", "days", "cum_net_return"], {"cum_net_return": "{:.1%}"})
    tbl_h = md_table(hr, ["signal", "band", "label", "n_obs", "hit_rate", "wilson_lo", "wilson_hi"],
                     {"hit_rate": "{:.3f}", "wilson_lo": "{:.3f}", "wilson_hi": "{:.3f}", "n_obs": "{:d}"})
    def _hr(sn, key):
        q = hr[(hr.signal == sn) & hr.band.str.startswith(key)].iloc[0]
        return q["hit_rate"], int(q["n_obs"])
    drops = {}
    for sn in ("tsfm_short", "clean_short", "v1_sign_short"):
        (h0, n0), (h1, n1) = _hr(sn, "2018-01"), _hr(sn, "2025-10")
        drops[sn] = (h0, h1, n0, n1)
    dtxt = "; ".join(f"{k} {v[0]:.3f} -> {v[1]:.3f} (n {v[2]} -> {v[3]})" for k, v in drops.items())
    fell = {k: v[1] < v[0] for k, v in drops.items()}
    if fell["tsfm_short"] and not (fell["clean_short"] or fell["v1_sign_short"]):
        concl = "TimesFM-only drop: FLAGGED as possible memorisation."
    elif fell["tsfm_short"]:
        concl = "Not flagged as memorisation; the drop is shared with the non-TimesFM signals."
    else:
        concl = "Not flagged: no TimesFM hit-rate drop after the cutoff."
    e2_concl = (f"**E2 conclusion (prereg rule: flag only a TimesFM-only drop after 2025-10-01).** Hit rate, "
                f"2018-01..2025-09 -> 2025-10..2026-09 on the common set: {dtxt}. {concl}")
    tbl_e = md_table(e3, ["book", "what", "period", "label", "days", "sharpe", "sharpe_lo", "sharpe_hi"],
                     {"sharpe": "{:.2f}", "sharpe_lo": "{:.2f}", "sharpe_hi": "{:.2f}", "days": "{:d}"})
    capm = cap.drop_duplicates(["book", "adv_variant", "impact_coef"])
    tbl_cap = md_table(capm, ["book", "adv_variant", "impact_coef", "label", "days", "capacity_aum_drop0p10"],
                       {"capacity_aum_drop0p10": "{:,.0f}", "days": "{:d}", "impact_coef": "{}"})

    def g(b, p, c):
        return summ[(summ.book == b) & (summ.period == p)].iloc[0][c]

    after = "recent (from 2025-10-01)"
    md = f"""# P7 v3 results: does a pretrained forecaster add anything to trend following?

{HEADER}. Generated by `scripts/v3/make_results_v3.py` from `results/v3/tables/*`. Pre-registration v3 hash `{file_hash(PREREG)}` (never edited). Research artifact, not investment advice, not a live-trading instruction.

**Labels.** TimesFM was pretrained on data that may include these years. Every TimesFM number dated before 2025-10-01 is `contaminated`; from 2018-01 it is also `seen` (the researcher saw v1 results for 2018-2026). `seen` after the cutoff is not contaminated but has been looked at. Nothing below is out-of-sample. The only confirmatory period for TimesFM is `forward` (from 2026-12), which has no data yet. For the non-TimesFM books (v1_blend, v1_weekly, clean_blend) the label column is kept for uniformity: `contaminated` is not a pretraining claim for them, but all of 2001-2026 is still seen by the researcher. Development and validation are where v1 parameters were chosen (in-sample for v1).

**Deviation from the preregistration.** The panel has 39 instruments; the preregistration says "37-market" (v1 wording). No book, rule or setting changed because of this.

**Costs and slippage.** v1 `cost_bp` plus roll costs, no market impact, in all PnL tables except the capacity table (square-root impact, lag 2).

## Verdicts

| hypothesis | verdict | label |
|---|---|---|
| H1 (TimesFM adds value, forward) | pending — first read 2027-11-30 | forward (no data yet) |
| H2 (clean model beats v1, CPCV 2001-2017) | **NOT SUPPORTED** | seen, not out-of-sample; {h2['days']} days |

H2 detail (EXP-v3-003 to 006): clean MLP ann. Sharpe {h2['clean_ann_sharpe']:.2f} (95% CI {h2['clean_ann_sharpe_ci'][0]:.2f} to {h2['clean_ann_sharpe_ci'][1]:.2f}, mean over 5 CPCV paths) vs v1 {h2['v1_ann_sharpe']:.2f} ({h2['v1_ann_sharpe_ci'][0]:.2f} to {h2['v1_ann_sharpe_ci'][1]:.2f}); mean path difference {h2['mean_path_diff']:.3f}, 95% CI [{h2['ci_lo']:.3f}, {h2['ci_hi']:.3f}], deflated Sharpe {h2['dsr_clean']:.3f} (bar 0.90). The CI spans zero and the DSR is below the bar. Two corrections after review did not change the verdict: EXP-v3-004 recomputed the CI on the pre-registered statistic, EXP-v3-005 put everything on one common day set; EXP-v3-006 notes the DSR uses the path-averaged series, which favours clean_blend. Signal at close t, trade t+1; v1 costs; capacity not assessed.

## Seen and contaminated periods (descriptive)

Books: `v1_blend` = v1 horizon blend (daily signal); `v1_weekly` = the same signal refreshed only every 5 days (the TimesFM schedule), to separate the effect of weekly refresh from the forecaster; `tsfm_blend` = TimesFM zero-shot (short/medium/long mean signal, held at most 5 days); `clean_blend` = MLP trained once (seed 0) on feature dates 1999-12-23 to 2017-01-09 (first full feature row to the last date whose 252-day targets end by 2017-12-29; the 252-day purge drops the last year's rows), applied from 2018 on (earlier rows are blank: not computed). Crisis and capacity tables report cumulative returns and capacity, so a Sharpe CI does not apply to them. A full (non-quick) rerun retrains on CUDA and may not be bit-identical; the committed cache in `results/v3/cache/` is the reference. Net of v1 costs. Signal at close t, trade t+1.

{tbl_s}

Net Sharpe with 95% stationary-bootstrap CI (mean block {BLOCK}, {BOOT} draws, seed 0); `days` is the sample count of each row; each book starts on its first live day. `recent (to 2025-09-30)` and `recent (from 2025-10-01)` split the recent period at the TimesFM cutoff. In the `test` period (2018-2024; `contaminated+seen` for TimesFM) Sharpe is {g('v1_blend','test','sharpe'):.2f} for v1_blend, {g('v1_weekly','test','sharpe'):.2f} for v1_weekly, {g('tsfm_blend','test','sharpe'):.2f} for tsfm_blend and {g('clean_blend','test','sharpe'):.2f} for clean_blend ({int(g('v1_blend','test','days'))} days). After the cutoff (`seen`) it is {g('v1_blend',after,'sharpe'):.2f}, {g('v1_weekly',after,'sharpe'):.2f}, {g('tsfm_blend',after,'sharpe'):.2f} and {g('clean_blend',after,'sharpe'):.2f} on {int(g('tsfm_blend',after,'days'))} days, with wide CIs.

### Crisis windows (cumulative net return, compounded)

Windows: 2018-10-01..2018-12-31, 2020-02-19..2020-03-23, 2022-01-01..2022-12-31. Signal at close t, trade t+1; v1 costs.

{tbl_c}

## Exploratory

Everything in this section is exploratory (not a pre-registered test). It cannot support a claim.

### E1 growth and drawdown

See `results/v3/figures/growth_v3.png` (log growth of the four books, shaded contaminated / seen regions) and the max-drawdown column above.

### E2 memorisation check: hit rate before vs after the TimesFM cutoff

Hit = sign of the short-horizon signal at close t equals sign of the realised log return over t+1..t+21, on the TimesFM forecast dates (every 5th day), per instrument; zero signals and zero returns are dropped. `clean_short` is the clean MLP short-horizon head (2018 on only; it has no pre-2018 out-of-fold predictions, so the pre-2018 band compares tsfm and v1 only); within each band all signals are scored on a common observation set (same date x instrument cells, all valid and non-zero); `v1_sign_short` is the sign of the v1 21-day signal. Wilson 95% CIs; observations overlap in time and are correlated across instruments, so the CIs are too narrow and are shown for scale only. Figure: `results/v3/figures/e2_memorisation.png`.

{tbl_h}

A memorised model would hit more often before the cutoff than after. Compare the TimesFM drop with the v1 sign and clean rows, which cannot have memorised these years. Differences of a point or two are inside the (understated) noise.

{e2_concl}

## Costs and capacity

Cost multiplier (v1 `cost_bp` and roll scaled by 1 or 2; signal at close t, trade t+1):

{tbl_e}

Capacity: square-root impact with coefficient 0.25 / 0.5 / 1.0, lag 2 (trade at the next close, as in v2 `execution.py`); capacity is the AUM at which net Sharpe falls 0.10 below the USD 10m value; period 2018-01..2026-10. `primary_yahoo_adv` is the upper bound on impact, `low_adv_instruments_impact_off` the lower bound; ADV is Yahoo volume with the v2 data-defect rules. This is a seen-period number, not forward evidence.

{tbl_cap}

Full AUM grid: `results/v3/tables/e3_costs.csv`.

## Data vintage

Reference vintage: data downloaded **2026-11-21** (last market date 2026-11-20); file hashes in `results/v3/DATA_VINTAGE.md`, copy in `snapshot/`. Every table, run and cache here is regenerated on it. Reason: Yahoo revised history after the first v3 run (downloaded 2026-10-13, ending 2026-10-13; e.g. MXN; the frozen clean model's signals changed on 35 of 2266 dates, max abs 0.23). The old vintage is not recoverable. Two things changed together, so old-to-new differences are not all revisions: (1) revised history; (2) 5 more weeks of data (2026-10-14 to 2026-11-20), which lengthens `recent`, `recent (from 2025-10-01)` and the 2018-01..end rows (labels still read "2026-10"; they mean "to end of data"). The clean model is the frozen `clean_model.pt` (not retrained); the earlier full run retrained it.

| item | old vintage | new vintage |
|---|---|---|
| test Sharpe v1_blend / v1_weekly / tsfm_blend / clean_blend | 0.23 / 0.23 / 0.21 / 0.78 | 0.23 / 0.23 / 0.21 / 0.78 |
| recent Sharpe (same order) | 0.38 / -0.13 / 0.32 / 1.49 | 0.23 / -0.18 / 0.16 / 0.81 |
| recent from 2025-10-01 Sharpe (same order) | 1.01 / 0.51 / 0.30 / 1.18 | 0.72 / 0.37 / 0.05 / 0.19 |
| H2 mean path difference, 95% CI, deflated Sharpe | 0.390, [-0.299, 1.075], 0.687 | 0.391, [-0.299, 1.077], 0.690 (still NOT SUPPORTED) |
| E2 hit rate 2018-01..2025-09 -> after cutoff (tsfm / clean / v1) | 0.522->0.500 / 0.525->0.510 / 0.498->0.462 | 0.522->0.504 / 0.525->0.496 / 0.498->0.468 (still not flagged) |
| E3 2018+ Sharpe cost x1 (v1 / tsfm) | 0.26 / 0.24 | 0.23 / 0.20 |
| E3 capacity, tsfm low-ADV, impact off (coef 0.25 / 0.5 / 1.0) | 500m / 100m / 50m | 1,000m / 250m / 100m |

The `recent` changes are mostly the extra five weeks (a weak stretch for the clean and TimesFM books); the development, validation and test periods end before any new data, so their small changes are pure revisions. The TimesFM cache was regenerated in full (35,878 of 153,579 old rows differ by more than 1e-6; max abs signal change 0.149). v1 and v2 results are unchanged and stay sealed on their own vintages.

## What this does not show

- Nothing here is out-of-sample. H1 cannot be read until the forward period has a full year of data (first read 2027-11-30).
- TimesFM rows before 2025-10-01 may reflect memorisation of the history; E2 is a weak check, not a proof either way.
- `clean_blend` was trained once on feature dates 1999-12-23 to 2017-01-09 (252-day purge before 2018) with fixed settings, but the researcher has seen 2018-2026 results from v1, so even its 2018+ rows are `seen`.
- Sharpe CIs are wide and differences between books are mostly inside them. No multiple-testing correction is applied to this descriptive table.
- Costs are v1 linear plus roll costs; impact appears only in the capacity table. Yahoo volume is a weak ADV source.
- The panel has 39 instruments, not the "37-market" wording in the preregistration.
"""
    (ROOT / "research/results_v3.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
