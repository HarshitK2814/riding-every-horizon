"""Period summaries, hypothesis tests and the fund comparison. All through the qcore test lock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from qcore.bootstrap import bootstrap_stat, percentile_ci
from qcore.metrics import summary
from qcore.splits import Splits, select

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "research" / "preregistration.md"


def splits() -> Splits:
    return Splits.from_dict(yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text()))


def monthly(r: pd.Series) -> pd.Series:
    return (1 + r).resample("ME").prod() - 1


def period(obj, name: str):
    return select(obj, splits(), name, PREREG)


def table(series: dict[str, pd.Series], name: str, periods: int = 12) -> pd.DataFrame:
    """Monthly-return summary (Sharpe with Lo 95% CI and N) for one period."""
    return pd.DataFrame({k: summary(period(monthly(v), name), periods) for k, v in series.items()}).T


def sharpe_m(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def sharpe_diff_test(a: pd.Series, b: pd.Series, n_boot: int = 5000, seed: int = 0) -> dict:
    """Stationary-bootstrap CI and one-sided p-value for Sharpe(a) - Sharpe(b), monthly data."""
    df = pd.concat([a, b], axis=1).dropna()
    df.columns = ["a", "b"]
    obs = sharpe_m(df["a"]) - sharpe_m(df["b"])
    draws = bootstrap_stat(df, lambda d: sharpe_m(d["a"]) - sharpe_m(d["b"]), n_boot=n_boot, seed=seed)
    centred = draws - draws.mean()
    return {"diff": obs, "ci": percentile_ci(draws), "p_one_sided": float((centred >= obs).mean()),
            "n_months": len(df)}


def sharpe_positive_test(a: pd.Series, n_boot: int = 5000, seed: int = 0) -> dict:
    a = a.dropna()
    obs = sharpe_m(a)
    draws = bootstrap_stat(a, sharpe_m, n_boot=n_boot, seed=seed)
    return {"sharpe": obs, "ci": percentile_ci(draws), "p_one_sided": float(((draws - draws.mean()) >= obs).mean()),
            "n_months": len(a)}


def convexity(strategy: pd.Series, market: pd.Series) -> dict:
    """Henriksson-Merton style 'smile': r = a + b*m + c*m^2, HAC (6 lags) t-stat on c."""
    df = pd.concat([strategy, market], axis=1).dropna()
    df.columns = ["r", "m"]
    X = sm.add_constant(pd.DataFrame({"m": df["m"], "m2": df["m"] ** 2}))
    fit = sm.OLS(df["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {"alpha_m": fit.params["const"], "beta": fit.params["m"], "gamma": fit.params["m2"],
            "t_gamma": fit.tvalues["m2"], "p_gamma_one_sided": float(fit.pvalues["m2"] / 2 if fit.params["m2"] > 0
                                                                     else 1 - fit.pvalues["m2"] / 2),
            "n_months": len(df)}
