"""P7 v2 Task 6 (H3): crash clustering. Pre-registration v2 (hash 2d7828df5093, FROZEN).
H3: monthly blend net return ~ Hawkes intensity of ES=F crash events (r < -2.5 x lagged EWMA sigma) at the start of the
month + 63-day realised S&P vol (known at the previous close). Supported iff intensity coef > 0 and Newey-West t > 2,
2001-2017. Descriptive add-on: the same with the rates sector removed. Timing: events/intensity/rv known before the
first trading day of month m; month-m net return = v1 blend (lag 1) compounded over m.
Run: PREREG_HASH=2d7828df5093 python scripts/v2/hawkes_h3.py"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from qcore.hawkes import fit_hawkes_exp, simulate_hawkes_exp
from qcore.splits import file_hash
from trend import data, model
from trend.v2.clustering import crash_events, monthly_intensity

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG = ROOT / "results" / "v2" / "tables", ROOT / "results" / "v2" / "figures"
PREREG = ROOT / "research" / "preregistration_v2.md"
START, END = "2001-01-01", "2017-12-31"
N_SIM = 200
NW_LAGS = 3


def ols_nw(y: pd.Series, X: pd.DataFrame) -> dict:
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"n_months": int(m.nobs), "coef": m.params.to_dict(), "nw_t": m.tvalues.to_dict(), "nw_p": m.pvalues.to_dict(),
            "r2": float(m.rsquared)}


def monthly_net(net: pd.Series) -> pd.Series:
    return (1 + net).groupby([net.index.year, net.index.month]).prod() - 1


def main() -> None:
    assert os.environ.get("PREREG_HASH") == file_hash(PREREG) == "2d7828df5093", "set PREREG_HASH=2d7828df5093"
    R = pd.read_parquet(data.PROC / "returns.parquet").loc[:END]
    inst = data.instrument_table()
    cfg_d = yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text())
    cfg = model.TrendConfig.from_yaml(cfg_d)

    es_all = R["ES=F"].dropna()
    ev_all = crash_events(es_all)                      # sigma warm-up uses pre-2001 ES=F (from 2000-09)
    ev = ev_all.loc[START:END]
    n_days = len(ev)
    t_ev = np.arange(n_days, dtype=float)[ev.to_numpy()]
    fit = fit_hawkes_exp(t_ev, float(n_days))
    rng = np.random.default_rng(0)
    sims = []
    for s in range(N_SIM):
        tt = simulate_hawkes_exp(fit["mu"], fit["alpha"], fit["beta"], float(n_days), seed=int(rng.integers(1 << 31)))
        if len(tt) >= 5:
            f = fit_hawkes_exp(tt, float(n_days)); sims.append(f)
    sd = pd.DataFrame(sims)
    fit_ci = {k: [float(sd[k].quantile(0.025)), float(sd[k].quantile(0.975))] for k in ("mu", "alpha", "beta", "branching")}
    tick_fit = None
    try:
        from tick.hawkes import HawkesExpKern
        lrn = HawkesExpKern(decays=fit["beta"], penalty="none", max_iter=1000)   # decay fixed at the qcore estimate
        lrn.fit([t_ev], end_time=float(n_days))
        br = float(lrn.adjacency[0, 0])
        tick_fit = {"decay_fixed_at_qcore_beta": fit["beta"], "mu": float(lrn.baseline[0]), "branching": br,
                    "alpha": br * fit["beta"]}
    except Exception as e:                                   # noqa: BLE001
        tick_fit = {"skipped": repr(e)[:200]}

    # monthly table
    inten = monthly_intensity(ev)
    es2 = es_all.loc[:END]
    rv = (es2.rolling(63).std().shift(1) * np.sqrt(252))     # known at the previous close
    firsts = ev.groupby([ev.index.year, ev.index.month]).head(1).index
    be = model.backtest(R, inst, cfg)
    cfg_nr = model.TrendConfig.from_yaml(cfg_d, sectors=tuple(s for s in sorted(inst["sector"].unique()) if s != "rates"))
    bn = model.backtest(R, inst, cfg_nr)

    def table(net: pd.Series) -> pd.DataFrame:
        m = monthly_net(net.loc[START:END])
        m.index = pd.MultiIndex.from_tuples(m.index)
        key = pd.MultiIndex.from_arrays([firsts.year, firsts.month])
        df = pd.DataFrame({"ret": m.reindex(key).to_numpy(), "intensity": inten.to_numpy(), "rv": rv.reindex(firsts).to_numpy()},
                          index=firsts)
        return df.dropna()

    out = {"prereg_hash": file_hash(PREREG), "window": [START, END], "timing": "events, intensity, rv known before first trading day of month; ret = month-m v1 blend net (lag 1)",
           "events": {"n": int(ev.sum()), "n_days": n_days, "rule": "ES=F r < -2.5 x EWMA(com 60) sigma lagged one day"},
           "hawkes_full_sample": fit, "hawkes_param_ci95_parametric_200_resims": fit_ci,
           "hawkes_note": "event times are integer trading-day positions; the continuous-time fit treats days as time units",
           "tick_HawkesExpKern": tick_fit}
    for label, net in (("blend", be["net"]), ("blend_ex_rates", bn["net"])):
        df = table(net)
        sdI = float(df["intensity"].std())
        out[label] = {"first_month": str(df.index[0].date()), "intensity_sd": sdI,
                      "multivariate": ols_nw(df["ret"], df[["intensity", "rv"]]),
                      "univariate": ols_nw(df["ret"], df[["intensity"]])}
        out[label]["h3_supported"] = bool(out[label]["multivariate"]["coef"]["intensity"] > 0 and out[label]["multivariate"]["nw_t"]["intensity"] > 2)
        if label == "blend":
            df.to_csv(TAB / "h3_monthly.csv")
            dfm = df
    out["h3_supported_literal"] = out["blend"]["h3_supported"]
    out["h3_rule"] = "supported iff intensity coef > 0 and Newey-West (3 lags) t > 2 in ret ~ intensity + rv, 2001-2017 (blend, all sectors)"
    (TAB / "h3_hawkes.json").write_text(json.dumps(out, indent=1, default=float))

    px = (1 + es_all.loc[START:END]).cumprod()
    dd = px / px.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    inten.plot(ax=ax, label="Hawkes intensity (month start)", color="C1")
    ax.set_ylabel("intensity (events/day)")
    ax2 = ax.twinx()
    dd.plot(ax=ax2, color="C0", alpha=0.5, label="ES=F drawdown")
    ax.set_title("Crash-event Hawkes intensity vs S&P (ES=F) drawdown, 2001-2017")
    ax.legend(loc="upper left"); ax2.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "hawkes_intensity.png", dpi=130)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
