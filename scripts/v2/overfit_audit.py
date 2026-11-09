"""P7 v2 Task 4 (H2): overfitting audit of the v1 blend over a 96-config family, 2001-2017 only.
H2 (preregistration_v2.md): PBO < 0.5 AND deflated Sharpe of the v1 blend (n_trials = 96) > 0.90.
Run: python scripts/v2/overfit_audit.py"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.validation import (cpcv_paths, cpcv_splits, deflated_sharpe, expected_max_sharpe, group_index,
                              pbo_cscv, probabilistic_sharpe)
from trend import data, model
from trend.v2.family import family_configs, family_returns

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG, RUNS = ROOT / "results" / "v2" / "tables", ROOT / "results" / "v2" / "figures", ROOT / "results" / "v2" / "runs"
for d in (TAB, FIG, RUNS):
    d.mkdir(parents=True, exist_ok=True)
V1 = "tanh|b0.10|blend|com60"
N_TRIALS = 96
START, END = "2001-01-01", "2017-12-31"


def sh_d(r) -> float:
    r = pd.Series(r)
    return float(r.mean() / r.std(ddof=1))


def ann(r) -> float:
    return sh_d(r) * np.sqrt(252)


def ci_ann(r: pd.Series, n_boot: int = 1000) -> tuple[float, float, float]:
    d = bootstrap_stat(r, lambda x: sh_d(x) * np.sqrt(252), n_boot=n_boot, mean_block=21)
    lo, hi = percentile_ci(d)
    return ann(r), lo, hi


def main() -> None:
    R = pd.read_parquet(data.PROC / "returns.parquet")
    inst = data.instrument_table()
    cfg_d = yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text())
    cache = RUNS / "family_2001_2017.parquet"
    if cache.exists():
        M = pd.read_parquet(cache)
    else:
        M = family_returns(R, inst, cfg_d, END)
        M.to_parquet(cache)
    M = M.loc[START:END]
    assert M.shape[1] == N_TRIALS and not M.isna().any().any()
    names = [n for n, _ in family_configs()]
    assert list(M.columns) == names

    # the v1 column must equal the plain v1 backtest
    v1 = model.backtest(R.loc[:END], inst, model.TrendConfig.from_yaml(cfg_d))["net"].loc[START:END]
    assert np.allclose(v1.reindex(M.index).to_numpy(), M[V1].to_numpy(), atol=1e-12), "family v1 column != v1 backtest"

    sr = M.apply(sh_d)                                  # per-period (daily) Sharpe of every config
    sr_var = float(sr.var(ddof=1))
    pbo = pbo_cscv(M, n_blocks=16)
    dsr = deflated_sharpe(M[V1], N_TRIALS, sr_var)
    sr_star = expected_max_sharpe(N_TRIALS, sr_var)
    psr0 = probabilistic_sharpe(M[V1], 0.0)
    v1_ann = ci_ann(M[V1])
    best = sr.idxmax()

    # CPCV: pick the best-train-Sharpe config in each split, keep its test returns
    n = len(M)
    G, K = 8, 2
    gi = group_index(M.index, G)
    oos, picks = {}, []
    X = M.to_numpy()
    for tr, te, tg in cpcv_splits(n, n_groups=G, k_test=K, purge=0, embargo=21):
        sr_tr = X[tr].mean(axis=0) / X[tr].std(axis=0, ddof=1)
        j = int(np.argmax(sr_tr))
        picks.append(names[j])
        oos[tg] = pd.Series(X[te, j], index=M.index[te])
    paths = cpcv_paths(oos, G, K, gi)
    path_rows = []
    for i, p in enumerate(paths):
        s, lo, hi = ci_ann(p, 500)
        path_rows.append({"path": i, "ann_sharpe": s, "ci95_lo": lo, "ci95_hi": hi, "n_days": len(p)})
    path_sh = np.array([r["ann_sharpe"] for r in path_rows])
    # same paths for the fixed v1 config (no selection), for comparison
    v1_paths = [pd.concat([M[V1].loc[gi[g]] for g in range(G)])]
    pick_counts = pd.Series(picks).value_counts()

    # descriptive add-on: barbell (21 + 252) vs three-horizon blend, Sharpe difference with bootstrap CI
    base = model.TrendConfig.from_yaml(cfg_d)
    bar = model.backtest(R.loc[:END], inst, model.TrendConfig.from_yaml(cfg_d, use=("short", "long")))["net"].loc[START:END]
    D = pd.concat([bar.reindex(M.index), M[V1]], axis=1, keys=["barbell", "blend"])
    diff = bootstrap_stat(D, lambda x: (sh_d(x["barbell"]) - sh_d(x["blend"])) * np.sqrt(252), n_boot=1000, mean_block=21)
    d_lo, d_hi = percentile_ci(diff)
    d_pt = (sh_d(D["barbell"]) - sh_d(D["blend"])) * np.sqrt(252)
    bar_ci = ci_ann(D["barbell"])

    h2_pbo = bool(pbo["pbo"] < 0.5)
    h2_dsr = bool(dsr > 0.90)
    out = {
        "period": [START, END], "n_days": int(n), "n_trials": N_TRIALS,
        "pbo": pbo["pbo"], "pbo_n_combinations": pbo["n_combinations"],
        "pbo_logit_median": float(np.median(pbo["logits"])),
        "sr_var_per_period": sr_var, "sr_var_note": "variance across the 96 configs of the per-period (daily) Sharpe, ddof=1",
        "expected_max_sharpe_per_period": sr_star, "expected_max_sharpe_annualised": sr_star * np.sqrt(252),
        "v1_blend_ann_sharpe": v1_ann[0], "v1_blend_ci95": [v1_ann[1], v1_ann[2]],
        "v1_psr_vs_0": psr0, "v1_deflated_sharpe": dsr,
        "best_in_sample_config": best, "best_in_sample_ann_sharpe": float(sr[best] * np.sqrt(252)),
        "family_ann_sharpe_quantiles": {str(q): float(sr.quantile(q) * np.sqrt(252)) for q in (0, .25, .5, .75, 1)},
        "n_configs_sharpe_gt_0": int((sr > 0).sum()),
        "cpcv": {"n_groups": G, "k_test": K, "embargo": 21, "purge": 0, "n_splits": len(picks), "paths": path_rows,
                 "median_path_ann_sharpe": float(np.median(path_sh)), "mean_path_ann_sharpe": float(path_sh.mean()),
                 "min_path": float(path_sh.min()), "max_path": float(path_sh.max()),
                 "selected_config_counts": pick_counts.to_dict()},
        "barbell_vs_blend": {"barbell_ann_sharpe": bar_ci[0], "barbell_ci95": [bar_ci[1], bar_ci[2]],
                             "diff_ann_sharpe_barbell_minus_blend": d_pt, "diff_ci95": [d_lo, d_hi],
                             "ci_excludes_zero": bool(d_lo > 0 or d_hi < 0)},
        "H2": {"pbo_lt_0.5": h2_pbo, "dsr_gt_0.90": h2_dsr, "supported": bool(h2_pbo and h2_dsr)},
    }
    (TAB / "overfit_audit.json").write_text(json.dumps(out, indent=1, default=float))
    pd.DataFrame({"ann_sharpe": sr * np.sqrt(252)}).to_csv(TAB / "overfit_family_sharpe.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(pbo["logits"], bins=40, color="#4477aa")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("logit of out-of-sample rank of the in-sample best config")
    ax.set_ylabel("CSCV combinations")
    ax.set_title(f"P7 v1 family (96 configs), 2001-2017: PBO = {pbo['pbo']:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "pbo_logits.png", dpi=130)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
