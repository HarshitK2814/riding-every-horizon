"""P7 v2 Task 7, E2 (EXPLORATORY, never headline): PySR symbolic signal trained on 2001-2012, judged ONCE on 2013-2017
against the v1 signal. Pre-registration v2 (hash 2d7828df5093). Training labels end by 2012-12-31 (purge 21 days).
Features at close t (t21, t63, t252, vratio); label = next-21-day return in daily-sigma units. Portfolio: equation output
clipped to [-1, 1] replaces the v1 signal (same risk scaling, cov-scale, buffer), lag 2, v1 costs, Sharpe 2013-2017.
Deflated Sharpe: n_trials = number of Pareto-front equations; sr_var = variance of the front's daily Sharpes on 2013-2017
(computed only to deflate the chosen equation; the choice was fixed by PySR's 'best' rule on training data before this).
Run: PREREG_HASH=2d7828df5093 python scripts/v2/e2_pysr.py"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.splits import file_hash
from qcore.validation import deflated_sharpe
from qengine.costs import CostModel
from qengine.engine import run_weights
from trend import data, model
from trend.v2.features import feature_panel

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "v2" / "tables"
PREREG = ROOT / "research" / "preregistration_v2.md"
TRAIN_END, TEST_START, END = "2012-12-31", "2013-01-01", "2017-12-31"
FEATS = ["t21", "t63", "t252", "vratio"]


def sh(r) -> float:
    r = pd.Series(np.asarray(r))
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def sci(r: pd.Series) -> list[float]:
    lo, hi = percentile_ci(bootstrap_stat(r, sh, n_boot=1000, mean_block=21))
    return [sh(r), lo, hi]


def signal_override(R: pd.DataFrame, sector: pd.Series, cfg: model.TrendConfig, sig: pd.DataFrame):
    """raw_positions math with the signal replaced by `sig` (in [-1, 1], known at close t)."""
    vol = model.ewma_vol(R, cfg.vol_com)
    unit = cfg.instrument_vol / vol
    sig = sig.reindex(index=R.index, columns=R.columns)
    unit = unit.where(sig.notna())
    sec = sector.reindex(unit.columns)
    live = unit.notna()
    n_in_sector = live.T.groupby(sec).transform("sum").T
    n_sectors = live.T.groupby(sec).any().T.sum(axis=1)
    unit = (unit / n_in_sector.replace(0, np.nan)).div(n_sectors.replace(0, np.nan), axis=0)
    return (sig * unit).fillna(0.0), unit.fillna(0.0)


def book(Rp, inst, cfg, sig, cm):
    W0, unit = signal_override(Rp, inst["sector"], cfg, sig)
    k = model.ewma_cov_scale(Rp, W0, cfg)
    W = model.apply_buffer(W0.mul(k, axis=0).fillna(0.0), unit.mul(k, axis=0).fillna(0.0), cfg.buffer)
    Rx = Rp.reindex(columns=W.columns).fillna(0.0)
    return run_weights(W, Rx, cm, lag=2)


def daily_ic(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(level=0).apply(lambda g: stats.spearmanr(g[col], g["y"])[0] if len(g) >= 8 else np.nan).dropna()


def nw_t(x: pd.Series, lags: int = 4) -> float:
    import statsmodels.api as sm
    m = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.tvalues[0])


def main() -> None:
    assert os.environ.get("PREREG_HASH") == file_hash(PREREG) == "2d7828df5093"
    try:
        from pysr import PySRRegressor
    except Exception as e:                                         # noqa: BLE001
        (TAB / "e2_pysr.json").write_text(json.dumps({"skipped": f"pysr not importable: {e!r}"[:300]}))
        return
    R = pd.read_parquet(data.PROC / "returns.parquet").loc[:END]
    inst = data.instrument_table()
    cfg = model.TrendConfig.from_yaml(yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text()))
    Rp = model.prepare(R)
    P = feature_panel(Rp, step=5)
    P = P.loc[P.index.get_level_values(0) >= "2001-01-01"]
    d = P.index.get_level_values(0)
    cut = Rp.index[Rp.index <= TRAIN_END][-22]                      # labels t+1..t+21 end on or before TRAIN_END
    tr = P[d <= cut].dropna(subset=FEATS + ["y"])
    if len(tr) > 20000:
        tr = tr.sample(20000, random_state=0)
    model_ = PySRRegressor(niterations=60, binary_operators=["+", "-", "*", "/"], unary_operators=["tanh", "sign", "abs"],
                           maxsize=15, populations=16, elementwise_loss="L1DistLoss()", deterministic=True,
                           parallelism="serial", random_state=0, model_selection="best", progress=False,
                           output_directory=str(ROOT / "results" / "v2" / "runs" / "pysr"))
    model_.fit(tr[FEATS], tr["y"])
    front = model_.equations_
    front.to_csv(TAB / "e2_pysr_front.csv", index=False)
    chosen = str(model_.sympy())
    # ---- judged ONCE on 2013-2017
    te = P[(d >= TEST_START) & (d <= END)].dropna(subset=FEATS + ["y"]).copy()
    te["eq"] = model_.predict(te[FEATS])
    te["v1"] = (np.tanh(te["t21"]) + np.tanh(te["t63"]) + np.tanh(te["t252"])) / 3
    te["v1_tanhmean"] = np.tanh((te["t21"] + te["t63"] + te["t252"]) / 3)
    ic = {}
    for c in ("eq", "v1", "v1_tanhmean"):
        s = daily_ic(te, c)
        ic[c] = {"mean_rank_ic": float(s.mean()), "t_naive": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))),
                 "t_newey_west_4": nw_t(s), "n_dates": int(len(s))}
    # portfolios (daily features)
    Fd = feature_panel(Rp, step=1)
    cm = CostModel.from_instruments(inst, 0.0)

    def wide(fn):
        X = Fd.dropna(subset=FEATS)
        v = pd.Series(np.asarray(fn(X)), index=X.index)
        return v.unstack().reindex(columns=R.columns)

    v1_sig = wide(lambda X: (np.tanh(X["t21"]) + np.tanh(X["t63"]) + np.tanh(X["t252"])) / 3)
    W0, _ = signal_override(Rp, inst["sector"], cfg, v1_sig)
    ref = model.raw_positions(Rp, inst["sector"], cfg)[0]
    assert np.allclose(W0.loc["2005":].to_numpy(), ref.loc["2005":].to_numpy(), atol=1e-9), "override != v1 raw positions"
    sl = slice(TEST_START, END)
    rb = book(Rp, inst, cfg, v1_sig, cm)
    eq_sig = wide(lambda X: np.clip(model_.predict(X[FEATS]), -1, 1))
    re_ = book(Rp, inst, cfg, eq_sig, cm)
    nb, ne = rb.net.loc[sl], re_.net.loc[sl]
    D = pd.concat([ne, nb], axis=1, keys=["e", "b"]).dropna()
    dd = bootstrap_stat(D, lambda x: sh(x["e"]) - sh(x["b"]), n_boot=1000, mean_block=21)
    # deflated Sharpe over the Pareto front
    sharpes = {}
    for i in range(len(front)):
        sg = wide(lambda X, i=i: np.clip(model_.predict(X[FEATS], index=i), -1, 1))
        sharpes[i] = book(Rp, inst, cfg, sg, cm).net.loc[sl]
    daily_sr = np.array([x.mean() / x.std(ddof=1) for x in sharpes.values()])
    n_tr = len(front)
    dsr = deflated_sharpe(ne.to_numpy(), n_trials=n_tr, sr_var=float(np.var(daily_sr, ddof=1))) if n_tr > 1 else None
    out = {"label": "EXPLORATORY (E2); judged once on 2013-2017; never headline", "prereg_hash": file_hash(PREREG),
           "train": {"rows": int(len(tr)), "label_window_ends_by": TRAIN_END, "last_train_date": str(cut.date())},
           "chosen_equation_sympy": chosen, "n_pareto_equations": int(n_tr),
           "ic_2013_2017_weekly": ic,
           "portfolio_2013_2017_lag2_v1costs": {"v1_signal_sharpe_ci": sci(nb), "equation_sharpe_ci": sci(ne),
                                                "diff_eq_minus_v1": sh(D["e"]) - sh(D["b"]), "diff_ci95": list(percentile_ci(dd)),
                                                "n_days": len(D), "rebalance_days_eq": int((re_.turnover.loc[sl] > 0).sum()),
                                                "rebalance_days_v1": int((rb.turnover.loc[sl] > 0).sum()),
                                                "turnover_per_yr_eq": float(re_.turnover.loc[sl].mean() * 252),
                                                "turnover_per_yr_v1": float(rb.turnover.loc[sl].mean() * 252)},
           "deflated_sharpe_chosen": dsr, "front_sharpes_2013_2017": {int(i): sh(x) for i, x in sharpes.items()}}
    (TAB / "e2_pysr.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
