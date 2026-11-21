# P7 v3 pre-registration: a time-series foundation model as a trend signal

Frozen: 2026-11-21, before any v3 return, Sharpe or hit-rate statistic was computed. Hash = sha256[:12] of this file.
v1 (`2cca1fec34d9`) and v2 (`2d7828df5093`) stay sealed.

## Question
Does a zero-shot time-series foundation model (TimesFM 2.5, 200M, Google, released Sep 2025) produce a better trend
signal than the v1 tanh t-statistic blend, net of costs, in the same 37-market risk-budgeted book?

## Design (fixed)
- Model: `google/timesfm-2.5-200m-pytorch` via PyPI `timesfm==3.0.2`; ForecastConfig max_context 1024, max_horizon 256,
  normalize_inputs, continuous quantile head, flip invariance, infer_is_positive False, fix_quantile_crossing;
  per_core_batch_size 256, panel chunk 512 (batch composition moves forecasts by ~3e-4 in log units; pinned for
  reproducibility).
- Context: cumulative log-return index per instrument, last 1024 trading days up to close t; min 256 obs.
- Forecast every 5th trading day (anchored at the first date of the returns panel), horizons 21/63/252 days from
  one 252-step forecast. Signal_h = tanh(mu_h / ((q90_h - q10_h)/2.5631)). Blend = mean of the three. Held
  until the next forecast date. Everything else (vol scaling, equal sector risk, EWMA vol target 10%, leverage
  cap 6, buffer 0.10, costs, lag 1) identical to v1.
- Clean control ("clean_blend"): a small MLP trained only on data inside each training fold (see H2),
  same signal mapping and book.
- Cadence control: v1 blend sampled every 5th day ("v1_weekly") reported alongside, so cadence is not confused
  with the model.

## Contamination rule
TimesFM's pretraining data may contain these markets up to its release. All TimesFM results dated before
2025-10-01 are labelled `contaminated` and are descriptive only. 2018-01..2026-11 is also `seen` (reported in v1/v2).

## Hypotheses (Holm across H1, H2 at 5%)
- **H1 (confirmatory, forward).** tsfm_blend net Sharpe minus v1_blend net Sharpe > 0 over the forward period
  (from 2026-12-01; first read 2027-11-30, ≥ 240 trading days). One-sided stationary block bootstrap
  (mean block 20, 5000 draws) of the daily Sharpe difference; p < 0.05 after Holm. Forward tracker lag 2, v1 costs.
- **H2 (clean control, 2001-2017 CPCV).** CPCV with 6 groups, 2 test groups (15 splits, 5 paths), purge 252 days,
  embargo 21. The MLP is trained on the train groups only, per split. Supported iff (a) the mean over paths of
  (clean_blend − v1_blend) annualised Sharpe is > 0 with 95% block-bootstrap CI excluding 0, and (b) the deflated
  Sharpe of clean_blend with n_trials = 98 (v2's 96 + 2 v3 books) is > 0.90.

## Exploratory (no verdict)
- E1: TimesFM on 2001-2026 (contaminated/seen), all books, crisis windows (Feb-Mar 2020, 2022, Q4 2018).
- E2 (memorisation probe): 21-day directional hit rate of TimesFM vs clean model vs v1 sign, before vs after
  2025-10-01. A TimesFM-only drop after its release is flagged as possible memorisation.
- E3: cost x2 and capacity for tsfm_blend.

## Reporting
Every table: period label, signal bar (close t, trade t+1), costs, Sharpe 95% CI, day count. Failed hypotheses stay in.
