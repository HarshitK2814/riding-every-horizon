"""P7 v3 H2 (preregistration_v3.md): clean MLP vs v1 blend under CPCV on 2001-2017 only.
6 groups, 2 test groups, purge 252, embargo 21. MLP trained on train groups per split. Signal at close t, trade t+1.
Costs: v1 cost_bp + rolls. Run: python scripts/v3/h2_cpcv.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.validation import cpcv_paths, cpcv_splits, deflated_sharpe, group_index
from trend import data, model
from trend.v3.clean_model import clean_signal, features, fit_predict, targets

ROOT = Path(__file__).resolve().parents[2]
OUT_T, OUT_R = ROOT / "results/v3/tables", ROOT / "results/v3/runs"
H = {"short": 21, "medium": 63, "long": 252}
START, END, N_TRIALS = "2001-01-01", "2017-12-31", 98
V2_AUDIT = ROOT / "results/v2/tables/overfit_audit.json"


def sh_d(r) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std(ddof=1))


def ann(r) -> float:
    return sh_d(r) * np.sqrt(252)


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True); OUT_R.mkdir(parents=True, exist_ok=True)
    R = model.prepare(pd.read_parquet(data.PROC / "returns.parquet")).loc[START:END]
    inst = data.instrument_table()
    cfg = model.TrendConfig.from_yaml(yaml.safe_load((ROOT / "configs/strategy.yaml").read_text()))
    X, Y = features(R), targets(R, H)
    dates = R.index
    groups = group_index(dates, 6)
    d0 = X.dropna().index.get_level_values(0).min()          # first date with a full feature row
    oos_sig = {}
    for tr_pos, te_pos, tg in cpcv_splits(len(dates), n_groups=6, k_test=2, purge=252, embargo=21):
        tr_d, te_d = dates[tr_pos], dates[te_pos]
        Xtr = X[X.index.get_level_values(0).isin(tr_d)].dropna()
        Ytr = Y[Y.index.get_level_values(0).isin(tr_d)].dropna()
        Xte = X[X.index.get_level_values(0).isin(te_d)].dropna()
        oos_sig[tg] = clean_signal(fit_predict(Xtr, Ytr, Xte, seed=len(oos_sig))).reindex(te_d)
    v1 = model.backtest(R, inst, cfg)["net"]
    oos_clean = {}
    for tg, s in oos_sig.items():
        net = model.backtest(R, inst, cfg, sig=s.reindex(dates))["net"].reindex(s.index)
        oos_clean[tg] = net.where(s.index >= d0)             # warm-up days (no features for any split) -> NaN
    paths_clean = cpcv_paths(oos_clean, 6, 2, groups)

    def pair(p):
        return pd.DataFrame({"c": p, "v": v1.reindex(p.index)}).dropna()

    diffs = [ann(pair(p)["c"]) - ann(pair(p)["v"]) for p in paths_clean]
    joined = pd.concat(paths_clean, axis=1).mean(axis=1, skipna=True)
    joined = joined[joined.index >= d0]
    d_series = pd.DataFrame({"c": joined, "v": v1.reindex(joined.index)}).dropna()
    draws = bootstrap_stat(d_series, lambda df: ann(df["c"]) - ann(df["v"]), n_boot=2000, mean_block=21)
    lo, hi = percentile_ci(draws)
    # sr_var: v2 convention = variance across the 96-config family of the per-period (daily) Sharpe
    sr_var = float(json.loads(V2_AUDIT.read_text())["sr_var_per_period"])
    sr_var_paths = float(np.var([sh_d(p) for p in paths_clean], ddof=1))
    dsr = float(deflated_sharpe(d_series["c"].to_numpy(), N_TRIALS, sr_var))
    dsr_paths = float(deflated_sharpe(d_series["c"].to_numpy(), N_TRIALS, sr_var_paths))
    md = float(np.mean(diffs))
    res = dict(mean_path_diff=md, ci_lo=float(lo), ci_hi=float(hi), dsr_clean=dsr,
               n_trials=N_TRIALS, paths=len(paths_clean), days=int(len(d_series)),
               clean_ann_sharpe=ann(d_series["c"]), v1_ann_sharpe=ann(d_series["v"]),
               path_diffs=[float(x) for x in diffs], sr_var=sr_var, sr_var_note="v2 family variance of daily Sharpe (96 configs)",
               dsr_sensitivity_sr_var_across_paths=dsr_paths, sr_var_across_paths=sr_var_paths,
               bootstrap=dict(n_boot=2000, mean_block=21, stat="ann Sharpe(mean-path clean) - ann Sharpe(v1)"),
               supported=bool(md > 0 and lo > 0 and dsr > 0.90),
               note=("2001-2017 CPCV (6 groups, 2 test, purge 252, embargo 21); signal close t, trade t+1; v1 costs and rolls; "
                     "not out-of-sample for the researcher; costs/slippage as v1, capacity not assessed. "
                     "Warm-up days before first full feature row excluded for both books. "
                     "Design bias: in each split the clean book holds zero positions outside its test groups, so entering "
                     "a test group incurs entry turnover cost while v1 trades continuously; this is a conservative bias "
                     "against the clean book."))
    pd.DataFrame({"path": range(len(diffs)), "clean_minus_v1_sharpe": diffs}).to_csv(OUT_T / "h2_cpcv_paths.csv", index=False)
    pd.concat({"-".join(map(str, k)): s.stack() for k, s in oos_sig.items()}, names=["split"]).to_frame("sig").to_parquet(
        OUT_R / "clean_signal_cpcv.parquet")
    (OUT_R / "h2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
