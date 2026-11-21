"""P7 v3 H2 (preregistration_v3.md): clean MLP vs v1 blend under CPCV on 2001-2017 only.
6 groups, 2 test groups, purge 252, embargo 21. MLP trained on train groups per split. Signal at close t, trade t+1.
Costs: v1 cost_bp + rolls. Run: python scripts/v3/h2_cpcv.py"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qcore.bootstrap import bootstrap_stat, percentile_ci, stationary_indices
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


def common_frame(paths: pd.DataFrame, v1: pd.Series) -> pd.DataFrame:
    """Rows where every path column and v1 are non-NaN; last column is v1. Single day set for point estimate and bootstrap."""
    df = paths.copy(); df["__v1"] = v1.reindex(df.index)
    return df.dropna()


def joint_path_bootstrap(paths: pd.DataFrame, v1: pd.Series, n_boot: int = 2000, mean_block: float = 21.0, seed: int = 0):
    """Pre-registered H2(a) statistic: days resampled jointly (stationary block bootstrap) across the clean path
    columns and v1; per draw, mean over paths of (ann Sharpe(path) - ann Sharpe(v1)).
    Returns (diff_draws, clean_draws, v1_draws); clean = mean over paths of ann Sharpe."""
    df = common_frame(paths, v1)
    vals = df.to_numpy(); n = len(vals); k = vals.shape[1] - 1
    rng = np.random.default_rng(seed)
    dd, cc, vv = (np.empty(n_boot) for _ in range(3))
    for b in range(n_boot):
        s = vals[stationary_indices(n, mean_block, rng)]
        m, sd = np.nanmean(s, axis=0), np.nanstd(s, axis=0, ddof=1)
        sr = m / sd * np.sqrt(252)
        cc[b], vv[b] = sr[:k].mean(), sr[k]
        dd[b] = (sr[:k] - sr[k]).mean()
    return dd, cc, vv


def main(from_cache: bool = False) -> None:
    OUT_T.mkdir(parents=True, exist_ok=True); OUT_R.mkdir(parents=True, exist_ok=True)
    R = model.prepare(pd.read_parquet(data.PROC / "returns.parquet")).loc[START:END]
    inst = data.instrument_table()
    cfg = model.TrendConfig.from_yaml(yaml.safe_load((ROOT / "configs/strategy.yaml").read_text()))
    X, Y = features(R), targets(R, H)
    dates = R.index
    groups = group_index(dates, 6)
    d0 = X.dropna().index.get_level_values(0).min()          # first date with a full feature row
    oos_sig = {}
    cached = pd.read_parquet(OUT_R / "clean_signal_cpcv.parquet")["sig"] if from_cache else None
    for tr_pos, te_pos, tg in cpcv_splits(len(dates), n_groups=6, k_test=2, purge=252, embargo=21):
        tr_d, te_d = dates[tr_pos], dates[te_pos]
        if from_cache:
            sub = cached.xs("-".join(map(str, tg)), level="split").unstack("inst")
            sub.index = pd.to_datetime(sub.index)
            oos_sig[tg] = sub.reindex(te_d)
            continue
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

    pdf = pd.concat([p.rename(i) for i, p in enumerate(paths_clean)], axis=1)
    pdf = pdf[pdf.index >= d0]
    common = common_frame(pdf, v1)                           # one day set: all 5 paths and v1 non-NaN
    k = common.shape[1] - 1
    cv = common["__v1"]
    path_c = [ann(common[i]) for i in range(k)]
    v1_sr = ann(cv)
    diffs = [c - v1_sr for c in path_c]
    d_series = pd.DataFrame({"c": common.iloc[:, :k].mean(axis=1), "v": cv})
    n_days = len(common)
    n_v1 = int(pdf.index.isin(v1.dropna().index).sum())
    dd, cc, vv = joint_path_bootstrap(common.iloc[:, :k], cv, n_boot=2000, mean_block=21)
    lo, hi = percentile_ci(dd)
    draws = bootstrap_stat(d_series, lambda df: ann(df["c"]) - ann(df["v"]), n_boot=2000, mean_block=21)
    old_lo, old_hi = percentile_ci(draws)
    c_lo, c_hi = percentile_ci(cc); v_lo, v_hi = percentile_ci(vv)
    # sr_var: v2 convention = variance across the 96-config family of the per-period (daily) Sharpe
    sr_var = float(json.loads(V2_AUDIT.read_text())["sr_var_per_period"])
    sr_var_paths = float(np.var([sh_d(common[i]) for i in range(k)], ddof=1))
    dsr = float(deflated_sharpe(d_series["c"].to_numpy(), N_TRIALS, sr_var))
    dsr_paths = float(deflated_sharpe(d_series["c"].to_numpy(), N_TRIALS, sr_var_paths))
    md = float(np.mean(diffs))
    res = dict(mean_path_diff=md, ci_lo=float(lo), ci_hi=float(hi), dsr_clean=dsr,
               ci_avg_series_lo=float(old_lo), ci_avg_series_hi=float(old_hi),
               ci_avg_series_note="superseded: bootstrap of averaged-path series",
               dsr_clean_across_path_var=dsr_paths,
               clean_ann_sharpe_ci=[float(c_lo), float(c_hi)], v1_ann_sharpe_ci=[float(v_lo), float(v_hi)],
               n_trials=N_TRIALS, paths=len(paths_clean), days=int(n_days), bootstrap_n=int(len(common)),
               days_note=(f"common day set = days >= first full feature row where all 5 CPCV paths and v1 are non-NaN ({n_days}); "
                          f"{n_v1 - n_days} days with v1 present but some path NaN dropped (days not covered by every path); "
                          "previous run used per-path day sets (4129 each) for the point estimate and 4133 for the bootstrap"),
               clean_ann_sharpe=float(np.mean(path_c)), v1_ann_sharpe=float(v1_sr),
               headline_note="clean_ann_sharpe = mean over paths of per-path annualised Sharpe (matches its CI); v1 Sharpe on the common day set",
               clean_ann_sharpe_avg_series=ann(d_series["c"]),
               clean_ann_sharpe_avg_series_note="Sharpe of the 5-path averaged return series; diversified, not the prereg statistic",
               path_diffs=[float(x) for x in diffs], sr_var=sr_var, sr_var_note="v2 family variance of daily Sharpe (96 configs)",
               dsr_sensitivity_sr_var_across_paths=dsr_paths, sr_var_across_paths=sr_var_paths,
               bootstrap=dict(n_boot=2000, mean_block=21, stat="mean over paths of (ann Sharpe(clean path) - ann Sharpe(v1)); joint day resampling"),
               supported=bool(md > 0 and lo > 0 and dsr > 0.90),
               note=("2001-2017 CPCV (6 groups, 2 test, purge 252, embargo 21); signal close t, trade t+1; v1 costs and rolls; "
                     "not out-of-sample for the researcher; costs/slippage as v1, capacity not assessed. "
                     "Warm-up days before first full feature row excluded for both books. "
                     "Design bias: in each split the clean book holds zero positions outside its test groups, so entering "
                     "a test group incurs entry turnover cost while v1 trades continuously; this is a conservative bias "
                     "against the clean book."))
    tbl = pd.DataFrame({"path": range(len(diffs)), "period": "2001-2017 CPCV",
                        "label": "contaminated-free model; seen period for researcher",
                        "signal_bar": "close t, trade t+1", "costs": "v1 cost_bp + roll",
                        "days": n_days,
                        "clean_ann_sharpe": path_c,
                        "v1_ann_sharpe": v1_sr,
                        "clean_minus_v1_sharpe": diffs})
    tbl.to_csv(OUT_T / "h2_cpcv_paths.csv", index=False)
    if not from_cache:
        pd.concat({"-".join(map(str, k)): s.stack() for k, s in oos_sig.items()}, names=["split"]).to_frame("sig").to_parquet(
            OUT_R / "clean_signal_cpcv.parquet")
    (OUT_R / "h2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true", help="recompute statistics from cached signals, no training")
    main(ap.parse_args().from_cache)
