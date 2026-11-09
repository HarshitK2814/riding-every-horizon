"""P7 v2 Task 5 (H1): GJR-GARCH(1,1,1)-t vs EWMA (com 60) risk scaling. 2001-2017 data only; H1 window 2004-2017.
Pre-registration v2 (hash 2d7828df5093, FROZEN) H1: mean over instruments of RMSE(realised 63d vol of unit-scaled
return vs 0.40). Supported if EWMA-minus-GARCH > 0 with a 95% stationary-bootstrap CI (mean block 21) excluding 0.
Secondary: Kupiec and Christoffersen p > 0.05 for 99% one-day portfolio VaR under GARCH.
Run: PREREG_HASH=2d7828df5093 python scripts/v2/risk_h1.py"""

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

from qcore.bootstrap import bootstrap_stat, percentile_ci, stationary_indices
from qcore.risk import christoffersen_ind, kupiec_pof
from qcore.splits import file_hash
from trend import data, model
from trend.v2.riskmodel import (backtest_with_vol, exante_vol, garch_vol_panel, garch_vol_panel_guarded,
                                tracking_dev)

ROOT = Path(__file__).resolve().parents[2]
TAB, FIG, RUNS = ROOT / "results" / "v2" / "tables", ROOT / "results" / "v2" / "figures", ROOT / "results" / "v2" / "runs"
PREREG = ROOT / "research" / "preregistration_v2.md"
END, H1_START = "2017-12-31", "2004-01-01"
TARGET = 0.40
N_BOOT = 2000


def sh_d(r) -> float:
    r = pd.Series(np.asarray(r))
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def ci(r: pd.Series) -> list[float]:
    lo, hi = percentile_ci(bootstrap_stat(r, sh_d, n_boot=1000, mean_block=21))
    return [sh_d(r), lo, hi]


def rmse_diff_stat(E2: np.ndarray, G2: np.ndarray, min_days: int = 250) -> tuple[float, np.ndarray]:
    """E2, G2: T x N squared deviations (NaN = not available). Returns mean over instruments of RMSE_ewma - RMSE_garch,
    and the per-instrument differences. Only days where BOTH are available count for an instrument."""
    ok = np.isfinite(E2) & np.isfinite(G2)
    n = ok.sum(axis=0)
    re = np.sqrt(np.where(ok, E2, 0.0).sum(axis=0) / np.maximum(n, 1))
    rg = np.sqrt(np.where(ok, G2, 0.0).sum(axis=0) / np.maximum(n, 1))
    d = re - rg
    keep = n >= min_days
    return float(d[keep].mean()), np.where(keep, d, np.nan)


def boot_rmse_diff(E2: np.ndarray, G2: np.ndarray, n_boot: int = N_BOOT, mean_block: float = 21, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = E2.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_indices(T, mean_block, rng)
        out[b] = rmse_diff_stat(E2[idx], G2[idx])[0]
    return out


def evaluate(label, garch, R, Rp, ewma, be, inst, cfg, sector) -> dict:
    avail = ((garch - ewma).abs() > 0)                        # False where GARCH fell back to EWMA
    dev_e = tracking_dev(Rp, ewma, TARGET).loc[H1_START:END]
    dev_g = tracking_dev(Rp, garch, TARGET).loc[H1_START:END]
    E2, G2 = (dev_e ** 2).to_numpy(), (dev_g ** 2).to_numpy()
    stat, per_inst = rmse_diff_stat(E2, G2)
    lo, hi = percentile_ci(boot_rmse_diff(E2, G2))
    # plan's alternative CI: bootstrap of the daily cross-instrument mean squared-error difference series
    dser = pd.Series(np.nanmean(E2 - G2, axis=1), index=dev_e.index).dropna()
    d_lo, d_hi = percentile_ci(bootstrap_stat(dser, lambda x: float(np.mean(x)), n_boot=N_BOOT, mean_block=21))
    av = avail.loc[H1_START:END].rolling(63).min().fillna(0).astype(bool).to_numpy()
    stat_a, _ = rmse_diff_stat(np.where(av, E2, np.nan), np.where(av, G2, np.nan))
    lo_a, hi_a = percentile_ci(boot_rmse_diff(np.where(av, E2, np.nan), np.where(av, G2, np.nan), n_boot=1000, seed=1))
    pi = pd.Series(per_inst, index=dev_e.columns)
    sec, sec_n = pi.groupby(sector).mean(), pi.groupby(sector).count()
    per_tab = pd.DataFrame({"sector": sector, "rmse_diff_ewma_minus_garch": per_inst,
                            "rmse_ewma": np.sqrt(np.nanmean(E2, axis=0)), "rmse_garch": np.sqrt(np.nanmean(G2, axis=0)),
                            "garch_share_of_days": avail.loc[H1_START:END].mean().to_numpy()}, index=dev_e.columns)
    per_tab.to_csv(TAB / f"h1_per_instrument_{label}.csv")
    worst = pi.sort_values().head(3)
    # portfolios
    bg = backtest_with_vol(R, inst, cfg, garch)
    rows = {}
    for lab, sl in (("2001-2017", slice("2001-01-01", END)), ("2004-2017", slice(H1_START, END))):
        ne, ng = be["net"].loc[sl], bg["net"].loc[sl]
        D = pd.concat([ng, ne], axis=1, keys=["g", "e"]).dropna()
        dd = bootstrap_stat(D, lambda x: (sh_d(x["g"]) - sh_d(x["e"])), n_boot=1000, mean_block=21)
        rows[lab] = {"ewma_sharpe_ci": ci(ne), "garch_sharpe_ci": ci(ng), "n_days": int(len(D)),
                     "garch_minus_ewma": sh_d(D["g"]) - sh_d(D["e"]), "diff_ci95": list(percentile_ci(dd)),
                     "ewma_rebalance_days": int((be["weights"].diff().abs().sum(axis=1).loc[sl] > 0).sum()),
                     "garch_rebalance_days": int((bg["weights"].diff().abs().sum(axis=1).loc[sl] > 0).sum())}
    var_rows = {}
    for name, res, vols in (("ewma_book_ewma_cov", be, None), ("garch_book_ewma_cov", bg, None),
                            ("garch_book_garch_cov", bg, garch)):
        pv = exante_vol(Rp, res["weights"], vols=vols)
        var = (2.326 * pv).shift(1)
        df = pd.concat([(-res["gross"]).rename("loss"), var.rename("var")], axis=1).loc[H1_START:END].dropna()
        hits = df["loss"] > df["var"]
        k, c = kupiec_pof(hits, 0.01), christoffersen_ind(hits)
        var_rows[name] = {"n": k["n"], "breaches": k["breaches"], "breach_rate": k["breaches"] / k["n"],
                          "kupiec_p": k["p"], "christoffersen_p": c["p"], "mean_var_pct": float(df["var"].mean() * 100),
                          "pass_both_gt_0.05": bool(k["p"] > 0.05 and c["p"] > 0.05)}
    good = pi.dropna()
    return {"n_instruments": int(len(good)),
            "h1_statistic_mean_rmse_ewma_minus_garch": stat, "h1_ci95": [lo, hi],
            "h1_supported_primary": bool(stat > 0 and lo > 0),
            "alt_ci_mean_sq_error_diff_series": {"mean": float(dser.mean()), "ci95": [d_lo, d_hi]},
            "mean_rmse_ewma": float(per_tab["rmse_ewma"].mean()), "mean_rmse_garch": float(per_tab["rmse_garch"].mean()),
            "median_instrument_rmse_diff": float(good.median()),
            "n_instruments_garch_better": int((good > 0).sum()), "n_instruments_ewma_better": int((good < 0).sum()),
            "worst_instruments_for_garch": {k: float(v) for k, v in worst.items()},
            "sensitivity_garch_available_only": {"stat": stat_a, "ci95": [lo_a, hi_a]},
            "per_sector_rmse_diff": {k: {"mean": float(v), "n_instruments": int(sec_n[k])} for k, v in sec.items()},
            "portfolio": rows, "var99": var_rows,
            "h1_secondary_garch_var_pass": {"ewma_cov_scaling": var_rows["garch_book_ewma_cov"]["pass_both_gt_0.05"],
                                            "garch_cov_scaling": var_rows["garch_book_garch_cov"]["pass_both_gt_0.05"]}}


def main() -> None:
    assert os.environ.get("PREREG_HASH") == file_hash(PREREG) == "2d7828df5093", "set PREREG_HASH=2d7828df5093"
    RUNS.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(data.PROC / "returns.parquet").loc[:END]
    inst = data.instrument_table()
    sector = inst["sector"].reindex(R.columns)
    cfg_d = yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text())
    cfg = model.TrendConfig.from_yaml(cfg_d)
    Rp = model.prepare(R)

    ewma = model.ewma_vol(Rp, cfg.vol_com)
    panels = {}
    for label, fn, fname in (("literal", garch_vol_panel, "garch_vol.parquet"),
                             ("guarded", garch_vol_panel_guarded, "garch_vol_guarded.parquet")):
        cache = RUNS / fname
        if cache.exists():
            panels[label] = pd.read_parquet(cache)
        else:
            panels[label] = fn(Rp, refit_every=63, min_obs=750)
            panels[label].to_parquet(cache)
    be = model.backtest(R, inst, cfg)
    results = {}
    for label, garch in panels.items():
        results[label] = evaluate(label, garch, R, Rp, ewma, be, inst, cfg, sector)
    lit = results["literal"]
    out = {"prereg_hash": file_hash(PREREG), "h1_window": [H1_START, END],
           "verdict_rule": "LITERAL: H1 supported iff mean-over-instruments RMSE(EWMA)-RMSE(GARCH) > 0 with 95% bootstrap CI excluding 0, "
                           "GARCH exactly as qcore.risk.garch_vol_forecast (refit every 63d, min 750 obs, EWMA only before 750 obs)",
           "h1_supported_literal": lit["h1_supported_primary"], "literal": lit, "guarded_sensitivity": results["guarded"]}
    (TAB / "h1_risk.json").write_text(json.dumps(out, indent=1, default=float))
    per_tab = pd.read_csv(TAB / "h1_per_instrument_guarded.csv", index_col=0)
    dev_e = tracking_dev(Rp, ewma, TARGET).loc[H1_START:END]
    E2 = (dev_e ** 2).to_numpy()
    dev_g = tracking_dev(Rp, panels["guarded"], TARGET).loc[H1_START:END]
    G2 = (dev_g ** 2).to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    m_e = pd.Series(np.sqrt(np.nanmean(E2, axis=1)), index=dev_e.index)
    m_g = pd.Series(np.sqrt(np.nanmean(G2, axis=1)), index=dev_e.index)
    m_e.rolling(126).mean().plot(ax=ax[0], label="EWMA com 60")
    m_g.rolling(126).mean().plot(ax=ax[0], label="GJR-GARCH-t (guarded; literal version blows up on PA, PL, UST2)")
    ax[0].set_title("cross-instrument RMS tracking error vs 40% target (126d avg)")
    ax[0].legend()
    per_tab["rmse_diff_ewma_minus_garch"].sort_values().plot.barh(ax=ax[1], fontsize=5)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_title("RMSE EWMA - guarded GARCH by instrument (>0: GARCH better)")
    fig.tight_layout()
    fig.savefig(FIG / "vol_tracking.png", dpi=130)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
