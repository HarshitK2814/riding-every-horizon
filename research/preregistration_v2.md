# Pre-registration v2: Riding Every Horizon (P7)
Frozen 2026-10-29. v1 pre-registration (hash 2cca1fec34d9) and its results are unchanged.

## Evidence ranking
1. Forward paper tracking from 2026-11 (only unseen data). 2. CPCV / walk-forward on 2001-2017.
3. 2018-2026: seen period (v1 reported it); descriptive only.

Timing: every signal is computed at the close of day t from data up to and including t. v1 trades at that close (lag 1); v2 reports lag 1 and the realistic lag 2 (next close).

## Primary hypotheses
- **H1 (risk model).** Instrument-level GJR-GARCH(1,1,1)-t vol (refit every 63 days, expanding) tracks the
  40% per-instrument risk target better than EWMA (com 60, half-life about 42 days; kept as the baseline estimator). Metric: mean over instruments of RMSE between
  rolling 63-day realised vol of the unit-scaled return and 0.40, on 2004-2017. Supported if the
  EWMA-minus-GARCH RMSE difference > 0 with a 95% stationary-bootstrap CI (mean block 21) excluding 0.
  Secondary: Kupiec and Christoffersen p > 0.05 for 99% one-day portfolio VaR under GARCH.
  Descriptive breakdown (no separate hypothesis): the same RMSE difference per sector.
- **H2 (overfitting audit of v1).** Over the 96-config family (signal {sign, tanh} x buffer {0, 0.05,
  0.1, 0.2} x horizons {blend, short, medium, long} x vol_com {30, 60, 120}) on 2001-2017:
  PBO < 0.5 and the v1 blend's deflated Sharpe (n_trials = 96) > 0.90. Reported either way.
  Descriptive add-on: horizon redundancy, comparing a short+long barbell (21 + 252 day) against the three-horizon blend, reported with a 95% bootstrap CI on the Sharpe difference.
- **H3 (crash clustering).** Monthly blend net return regressed on the Hawkes intensity of equity-crash
  events (ES=F return < -2.5 x lagged EWMA sigma) at the start of the month, controlling for 63-day
  realised S&P vol, has a positive intensity coefficient with Newey-West t > 2 on 2001-2017.
  Hawkes parameters refit yearly on past events only.
  Descriptive add-on: the same regression with the rates sector removed from the blend, because 2001-2017 is a bond bull market and bonds and equities need not fall together.

## Descriptive (no hypothesis)
- D1 execution: lag 1 vs lag 2; square-root impact (coef 0.25/0.5/1.0) at AUM $10m-$5bn; integer
  contracts at $10m/$50m; capacity = largest AUM with net Sharpe within 0.10 of the $10m value.
  Also reported: turnover and cost in Sharpe units; a cost ceiling check (net cost at most one third of pre-cost Sharpe, and a turnover speed limit
  set at the v1 blend's own turnover); any horizon or sector that breaks the ceiling is flagged, not removed; margin-to-equity path against the 10-30% band
  (assumption: margin 5% of notional, stated in the table); results split by sector and by tick size (small vs large tick) for the seen period.
  Capacity is never offered as the cause of any decay.
- D2 roll-gap sensitivity (Task 1, run before this freeze, 2001-2017 only): flagged roll gaps imputed with the sector median.
  The v1 data stay in use because the Sharpe change was inside its own CI half-width (see research/data_audit_v2.md).
  Backwards-ratio adjustment with an explicit roll needs contract-level prices and is future work.

## Exploratory (labelled exploratory everywhere; never headline)
- E1 adaptive conformal intervals (alpha 0.1) on next-21-day instrument returns; coverage and a
  "trade only if the interval excludes zero" filter.
- E2 PySR symbolic signal trained on 2001-2012, judged once on 2013-2017 against the v1 signal.

## Forward test
From 2026-11 the v1 blend, the GARCH variant and (if E1/E2 look useful) one exploratory variant are
tracked daily by scripts/v2/forward.py. Earliest read: 2027-11 (12 months). Report Sharpe with CI.

## What would change our conclusions
A bug found after freezing is fixed, logged in experiment_log_v2.md, and both numbers reported.
