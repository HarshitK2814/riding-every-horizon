# Pre-registration: Riding Every Horizon (P7)

Frozen before the test period (2018-01 to 2024-12) is read. The code refuses to load test data until
`PREREG_HASH` equals the SHA-256 prefix of this file (`qcore.splits.guard`). Development 2001-2012 and
validation 2013-2017 were used for every design choice (see `experiment_log.md`, EXP-001 to EXP-004).

## Question
Does a rule built only from what a long-running trend manager discloses (six sectors, short/medium/long trend
models) earn a positive risk-adjusted return out of sample, net of realistic costs, and does it behave like a
crisis hedge?

## Frozen system
`configs/strategy.yaml` and `configs/universe.yaml` as committed with this file:
* 37 markets in six sectors: 6 equity indices, 4 US Treasury maturities, 8 currencies, 5 metals, 5 energy,
  11 agricultural/livestock. Rates and FX are carry-correct FRED reconstructions; the rest are Yahoo front-month
  futures with the logged cleaning rule.
* Signal for horizon h in {21, 63, 252} days: tanh(trailing h-day log return / (sigma_daily sqrt h)), computed on
  the close of day t; the position is held over day t+1.
* Position = signal x 40% / EWMA vol (centre of mass 60 d); equal risk across live sectors and within a sector;
  scaled to 10% ex-ante vol with an EWMA covariance (half-life 126 d); gross leverage cap 6x; 10% no-trade band.
* Costs: per-contract one-way cost (half a tick + $2 commission, x1.5-3) on every trade, plus two trades per roll.
  All results are net of these costs. Capacity is not modelled (positions are in NAV units).

## Hypotheses (monthly net excess returns, test period 2018-01 to 2024-12, 84 months)
* **H1 (it works):** Sharpe of the three-horizon blend > 0. Stationary bootstrap (mean block 6 months, 5,000
  draws), one-sided p < 0.05.
* **H2 (horizon diversification):** Sharpe(blend) - Sharpe(best single horizon) > 0, where the best single horizon
  was chosen on development + validation (it is the 252-day horizon; `sources/best_single_horizon_devval.json`).
  Paired stationary bootstrap, one-sided p < 0.05. On development + validation the blend was *worse* (0.35 vs
  0.68), so we expect H2 to fail; we test it because the manager's design claims the opposite.
* **H3 (crisis convexity):** in r = a + b m + c m^2 on monthly SPY returns m, c > 0 with HAC (6 lags) one-sided
  p < 0.05.

## Replication gate (decides how results are labelled)
Monthly correlation of the blend (plus T-bill) with the manager's own mutual fund (EBSIX, Class I) over the test
period must be >= 0.50. If not, the test results are labelled "indicative" rather than a replication.

## Reported without a hypothesis
Each single horizon, each sector book, the disclosed-mix book, cost drag, stress windows (Q4 2018, Feb-Mar 2020,
2022), and correlations with five public managed-futures funds. The 2025+ "recent" period is reported separately
and used for nothing.

## What would change our conclusions
A failed H1 means the disclosed design does not survive costs out of sample. A failed H3 means the "crisis alpha"
claim did not hold in this window. Nothing in this file will be edited after the hash is set.
