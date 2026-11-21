"""trend_lite: the v1 multi-horizon trend backtest, slimmed to run in the browser (Pyodide).

Same algorithm as src/trend/model.py (prepare, horizon_signal, horizon_signal_sign, raw_positions,
ewma_cov_scale, apply_buffer, costs, backtest), copied with the config object replaced by explicit
arguments. No imports from `trend`: only numpy and pandas. tests/test_v3_trend_lite.py checks that it
matches trend.model.backtest to 1e-12.

Timing: every quantity indexed t uses returns up to and including the close of day t; weights decided at
the close of t earn the return of day t+1 (lag 1). Signals are computed on the full history and only
then sliced to start..end, so a date range never changes what was known on a given day.
Costs: v1 cost_bp on turnover plus roll costs (two trades per roll), times `cost_mult`. No impact.
Sharpe CI: stationary bootstrap, mean block 20, 500 draws, seed 0 (the paper tables use 2000 draws).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = {"short": 21, "medium": 63, "long": 252}
VOL_COM = 60
INSTRUMENT_VOL = 0.40
COV_HALFLIFE = 126
MAX_LEVERAGE = 6.0
LAG = 1
N_BOOT, MEAN_BLOCK = 500, 20
# date-defined windows (the v3 crisis table plus calendar 2008); cumulative net return, no Sharpe
CRISES = {"2008": ("2008-01-01", "2008-12-31"), "2018Q4": ("2018-10-01", "2018-12-31"),
          "covid-2020": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-01", "2022-12-31")}


def prepare(R: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill holidays after each instrument's first observation; NaN before it."""
    started = R.notna().cumsum() > 0
    return R.fillna(0.0).where(started)


def ewma_vol(R: pd.DataFrame, com: int) -> pd.DataFrame:
    return R.ewm(com=com, min_periods=com).std() * np.sqrt(TRADING_DAYS)


def horizon_signal(R: pd.DataFrame, h: int, com: int = 60) -> pd.DataFrame:
    """tanh of the t-statistic of the trailing h-day log return. Known at the close of t."""
    lr = np.log1p(R)
    cum = lr.rolling(h, min_periods=h).sum()
    sd = lr.ewm(com=com, min_periods=com).std()
    return np.tanh(cum / (sd * np.sqrt(h)))


def horizon_signal_sign(R: pd.DataFrame, h: int, com: int = 60) -> pd.DataFrame:
    """v1 EXP-001 variant: sign of the trailing h-day log return."""
    cum = np.log1p(R).rolling(h, min_periods=h).sum()
    return np.sign(cum)


def raw_positions(R: pd.DataFrame, sector: pd.Series, use, signal: str = "tanh",
                  sig: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend of horizon signals, each scaled to INSTRUMENT_VOL, equal risk across sectors and equal
    within a sector. Returns (weights, unit); unit is the weight of a full-strength signal."""
    vol = ewma_vol(R, VOL_COM)
    names = tuple(use) if use else tuple(HORIZONS)
    sig_fn = horizon_signal if signal == "tanh" else horizon_signal_sign
    if sig is None:
        sig = sum(sig_fn(R, HORIZONS[n], VOL_COM) for n in names) / len(names)
    else:
        sig = sig.reindex(index=R.index, columns=R.columns)
    unit = INSTRUMENT_VOL / vol
    unit = unit.where(sig.notna())
    sec = sector.reindex(unit.columns)
    live = unit.notna()
    n_in_sector = live.T.groupby(sec).transform("sum").T
    n_sectors = live.T.groupby(sec).any().T.sum(axis=1)
    unit = (unit / n_in_sector.replace(0, np.nan)).div(n_sectors.replace(0, np.nan), axis=0)
    return (sig * unit).fillna(0.0), unit.fillna(0.0)


def ewma_cov_scale(R: pd.DataFrame, W: pd.DataFrame, target_vol: float) -> pd.Series:
    """Scalar k_t so that k_t * W_t has ex-ante vol `target_vol` (EWMA covariance up to t), capped at
    MAX_LEVERAGE gross."""
    X = R.reindex(columns=W.columns).fillna(0.0).to_numpy()
    Wn = W.to_numpy()
    lam = 0.5 ** (1 / COV_HALFLIFE)
    S = np.zeros((X.shape[1], X.shape[1]))
    k = np.full(len(X), np.nan)
    warm = 0
    for t in range(len(X)):
        x = X[t]
        S = lam * S + (1 - lam) * np.outer(x, x)
        warm += 1
        if warm < 126:
            continue
        v = Wn[t] @ S @ Wn[t] * TRADING_DAYS
        if v > 0:
            gross = np.abs(Wn[t]).sum()
            k[t] = min(target_vol / np.sqrt(v), MAX_LEVERAGE / gross if gross > 0 else np.inf)
    return pd.Series(k, index=W.index)


def apply_buffer(W: pd.DataFrame, unit: pd.DataFrame, width: float) -> pd.DataFrame:
    """No-trade band: keep yesterday's position unless the target moved more than `width` x the
    full-signal position size. Uses only information at t."""
    if width <= 0:
        return W
    T, U = W.to_numpy(), np.nan_to_num(unit.reindex_like(W).to_numpy())
    out = np.zeros_like(T)
    prev = np.zeros(T.shape[1])
    for t in range(len(T)):
        move = np.abs(T[t] - prev) > width * np.abs(U[t])
        prev = np.where(move | (U[t] == 0), T[t], prev)
        out[t] = prev
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def costs(W: pd.DataFrame, cost_bp: pd.Series, rolls: pd.Series) -> pd.Series:
    """Daily cost: one-way cost on turnover plus a roll cost (two trades per roll) on |position|."""
    c = cost_bp.reindex(W.columns).to_numpy() / 1e4
    rl = rolls.reindex(W.columns).to_numpy() / TRADING_DAYS
    turnover = W.diff().abs().fillna(W.abs())
    return (turnover * c).sum(axis=1) + (W.abs() * 2 * c * rl).sum(axis=1)


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan


def bootstrap_sharpe(x: np.ndarray, n_boot: int = N_BOOT, mean_block: float = MEAN_BLOCK, seed: int = 0) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano) of the annualised Sharpe. Same random draws, in the same
    order, as qcore.bootstrap.bootstrap_stat; the index loop is vectorised."""
    rng = np.random.default_rng(seed)
    n = len(x)
    p = 1.0 / mean_block
    ar = np.arange(n)
    out = np.empty(n_boot)
    for b in range(n_boot):
        first = rng.integers(n)
        new_block = rng.random(n) < p
        starts = rng.integers(n, size=n)
        new_block[0], starts[0] = True, first
        bs = np.maximum.accumulate(np.where(new_block, ar, 0))
        s = x[(starts[bs] + ar - bs) % n]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    return out


def _positions(R, sector, params, sig, cache):
    """Weights before the buffer, keyed so the browser can rerun buffer/cost changes without the
    covariance loop."""
    signal = params.get("signal", "tanh")
    use = tuple(params.get("use") or HORIZONS)
    key = (signal, () if signal == "tsfm" else use, float(params.get("target_vol", 0.10)))
    if cache is not None and key in cache:
        return cache[key]
    W0, unit = raw_positions(R, sector, use, "tanh" if signal == "tsfm" else signal,
                             sig=sig if signal == "tsfm" else None)
    k = ewma_cov_scale(R, W0, key[2])
    val = (W0.mul(k, axis=0).fillna(0.0), unit.mul(k, axis=0).fillna(0.0), k)
    if cache is not None:
        if len(cache) >= 12:          # ~5 MB per entry on the full panel; keep browser memory bounded
            cache.clear()
        cache[key] = val
    return val


def run(R: pd.DataFrame, sector: pd.Series, cost_bp: pd.Series, rolls: pd.Series, params: dict,
        sig: pd.DataFrame | None = None, cache: dict | None = None, rf: pd.Series | None = None) -> dict:
    """Backtest on full history, then slice to params start..end (or the union of `windows`).
    Returns `net` (excess of cash; Sharpe is computed on it) and `total` = net + daily T-bill `rf`
    (what the paper's growth curve and crisis table compound). Without `rf`, total == net."""
    if params.get("signal") == "tsfm" and sig is None:
        raise ValueError("signal 'tsfm' needs the precomputed signal panel `sig`")
    R = prepare(R)
    Wk, Uk, k = _positions(R, sector, params, sig, cache)
    W = apply_buffer(Wk, Uk, float(params.get("buffer", 0.10)))
    Rx = R.reindex(columns=W.columns).fillna(0.0)
    held = W.shift(LAG).fillna(0.0)                       # decided at close t, earns day t+1
    gross = (held * Rx).sum(axis=1)
    cost = (costs(W, cost_bp, rolls) * float(params.get("cost_mult", 1.0))).shift(LAG).fillna(0.0)
    started = k.notna().shift(LAG, fill_value=False)
    full = (gross - cost)[started]
    cash = rf.reindex(full.index).ffill().fillna(0.0) if rf is not None else pd.Series(0.0, index=full.index)
    full_tot = full + cash
    turn = W.diff().abs().fillna(W.abs()).sum(axis=1).shift(LAG).fillna(0.0)[started]
    crisis = {}
    for name, (a, z) in CRISES.items():
        r = full_tot.loc[a:z]
        ok = len(r) > 0 and full.index[0] <= pd.Timestamp(a)
        crisis[name] = {"start": a, "end": z, "days": int(len(r)) if ok else 0,
                        "cum_net_return": float((1 + r).prod() - 1) if ok else None}
    if params.get("windows"):
        m = np.zeros(len(full), dtype=bool)
        for a, z in params["windows"]:
            m |= (full.index >= pd.Timestamp(a)) & (full.index <= pd.Timestamp(z))
        net = full[m]
    else:
        net = full.loc[params.get("start"):params.get("end")]
    out = {"net": net, "total": net + cash.reindex(net.index), "days": int(len(net)), "crisis": crisis, "sharpe": np.nan, "sharpe_lo": np.nan,
           "sharpe_hi": np.nan, "ann_ret": np.nan, "ann_vol": np.nan, "maxdd": np.nan,
           "turnover": np.nan, "cost_ann": np.nan}
    if len(net) > 2:
        x = net.to_numpy()
        draws = bootstrap_sharpe(x)
        eq = np.cumprod(1 + x)
        out.update(sharpe=sharpe(net), sharpe_lo=float(np.nanquantile(draws, 0.025)),
                   sharpe_hi=float(np.nanquantile(draws, 0.975)), ann_ret=float(x.mean() * TRADING_DAYS),
                   ann_vol=float(x.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                   maxdd=float((eq / np.maximum.accumulate(eq) - 1).min()),
                   turnover=float(turn.reindex(net.index).mean() * TRADING_DAYS),
                   cost_ann=float(cost.reindex(net.index).mean() * TRADING_DAYS))
    return out


# ---- browser glue: the Web Worker calls exactly these two functions -------------------------------------------
def load_inputs(ret_bytes: bytes, inst_json: str, sig_bytes: bytes | None = None,
                rf_bytes: bytes | None = None) -> dict:
    """Parse docs/data files (gzip or already-decoded CSV bytes, as a server may decode them)."""
    import gzip
    import io
    import json

    def frame(b):
        b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
        return pd.read_csv(io.BytesIO(b), index_col=0, parse_dates=True).astype("float64")

    R = frame(ret_bytes)
    inst = pd.DataFrame(json.loads(inst_json)).T.reindex(R.columns)
    return {"R": R, "sector": inst["sector"], "cost_bp": inst["cost_bp"].astype(float),
            "rolls": inst["rolls"].astype(float), "sig": frame(sig_bytes) if sig_bytes else None,
            "rf": frame(rf_bytes)["rf"] if rf_bytes else None}


def run_json(inp: dict, params_json: str, cache: dict | None = None) -> str:
    """run() on loaded inputs; returns JSON (dates, equity = growth of 1 incl. T-bill, stats on excess returns, crisis) for the page."""
    import json
    p = json.loads(params_json)
    res = run(inp["R"], inp["sector"], inp["cost_bp"], inp["rolls"], p, sig=inp["sig"], cache=cache, rf=inp.get("rf"))
    net = res.pop("net")
    total = res.pop("total")
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res.items()}
    clean.update(dates=[d.strftime("%Y-%m-%d") for d in net.index],
                 equity=[round(float(v), 6) for v in np.cumprod(1 + total.to_numpy())])
    return json.dumps(clean)
