# P7 v3 experiment log (append-only)

## Setup: TimesFM install
- `.venv/Scripts/python -m pip install "timesfm[torch]>=2.0"` -> timesfm 3.0.2 from PyPI (no git install needed;
  `timesfm.TimesFM_2p5_200M_torch` exists). torch left unchanged at 2.8.0+cu128 (CUDA, RTX 4070 8 GB).
- Model: `google/timesfm-2.5-200m-pytorch` (HF hub, downloaded on first run; no commit pinned from git).

## EXP-v3-000 probe
- Script: `scripts/v3/probe_tsfm.py` (shapes and timing only; no performance statistics).
- Panel: returns.parquet 7166 dates x 39 instruments; probe = last 50 dates every 5th (10 dates), horizons 21/63/252.
- Result: 390 series in 190.1 s -> 249.6 s per 512 series (includes first-call warmup). Projected full run: 454.4 min.
- Projected > 240 min. Noted; prereg (Task 2) fixes cadence at 5 days regardless.
- Note: ForecastConfig.per_core_batch_size left at default (1); likely the main speed lever if a faster run is needed.

## EXP-v3-000b probe after batching
- Change: `per_core_batch_size` now passed to ForecastConfig (default 256). Batch-size test on 512 real series, context 1024, horizon 256, RTX 4070 8 GB: 256 -> 2.7 s/512 (peak 4.08 GiB), 128 -> 3.2 s (3.32 GiB), 64 -> 4.5 s (2.55 GiB); no OOM. Chose 256.
- Probe (same script, shapes and timing only): 390 series in 3.3 s -> 4.28 s per 512 series. Projected full run: 7.8 min (was 454.4 min).
- Batched (256) vs unbatched (1), 8 series: max abs point diff 2.8e-4 (1.7e-3 of series std); quantiles 4.2e-4 (2.3e-3 of std). Numerical noise (TF32 matmul), not a logic difference.

## EXP-v3-001 freeze
- Pre-registration frozen: `research/preregistration_v3.md`, dated 2026-11-21, committed alone as `e6f8367` (`prereg-v3: freeze`).
- PREREG_V3_HASH = `6451a846e036` (sha256[:12] of the frozen file; matches the committed blob).
- Splits: `configs/splits_v3.yaml`. Period labels: `src/trend/v3/labels.py` (forward starts 2026-12-01).
- No v3 return, Sharpe or hit-rate statistic computed.

## EXP-v3-002 tsfm cache
- Script: `scripts/v3/forecast_tsfm.py` (defaults: per_core_batch_size 256, chunk 512, context 1024, horizons 21/63/252, cadence 5 days) on RTX 4070.
- Output: `results/v3/cache/tsfm_signal.parquet`, 152994 rows, 1434 forecast dates (last 2026-10-13), 4.2 MB, run time 308 s.
- Note: panel has 39 instruments; prereg text says 37 markets (v1 wording) — the design applies to all instruments in data/processed/returns.parquet.
- Also: `raw_positions`/`backtest` gained an optional `sig` override (None = v1 behaviour, bit-identical; guarded by tests).
- No return, Sharpe or hit-rate statistic computed.

- Correction: EXP-v3-002 said 1434 forecast dates; cache holds 1383 distinct dates (first 51 schedule dates have < 256 obs).

## EXP-v3-003 H2 run
- Script: `scripts/v3/h2_cpcv.py`, run once with pre-registered settings (6 groups, 2 test, purge 252, embargo 21, n_trials 98, MLP seed = split index, 20 epochs), runtime 37.5 s, RTX 4070. No setting changed after seeing the result.
- Fix before run: `features` now maps +-inf (sd == 0) to NaN (test added).
- Choices not fixed by the prereg, made before the run: bootstrap mean block 21 (v2 convention), 2000 draws; deflated-Sharpe sr_var = v2 family variance of daily Sharpe (3.0187e-4, from overfit_audit.json); across-path variance reported as sensitivity only; warm-up days before the first full feature row (2001-12-ish) dropped for both books.
- Known bias (conservative against clean): in each split the clean book holds zero positions outside its test groups, so entering a test group incurs entry turnover cost while v1 trades continuously.
- Verdict: H2 NOT supported (CI includes 0, DSR 0.66 < 0.90; sensitivity DSR 0.83 also < 0.90).
```json
{
  "mean_path_diff": 0.38739411631762455,
  "ci_lo": -0.298716410830705,
  "ci_hi": 1.1623353664202687,
  "dsr_clean": 0.6600896889206009,
  "n_trials": 98,
  "paths": 5,
  "days": 4133,
  "clean_ann_sharpe": 0.7987898300755559,
  "v1_ann_sharpe": 0.34561626799411604,
  "path_diffs": [
    0.05646801994537498,
    0.402351025467672,
    0.32023902103329455,
    0.516011907407028,
    0.6419006077347533
  ],
  "sr_var": 0.00030187162001605157,
  "sr_var_note": "v2 family variance of daily Sharpe (96 configs)",
  "dsr_sensitivity_sr_var_across_paths": 0.8304726512496472,
  "sr_var_across_paths": 0.00019590376332900955,
  "bootstrap": {
    "n_boot": 2000,
    "mean_block": 21,
    "stat": "ann Sharpe(mean-path clean) - ann Sharpe(v1)"
  },
  "supported": false,
  "note": "2001-2017 CPCV (6 groups, 2 test, purge 252, embargo 21); signal close t, trade t+1; v1 costs and rolls; not out-of-sample for the researcher; costs/slippage as v1, capacity not assessed. Warm-up days before first full feature row excluded for both books. Design bias: in each split the clean book holds zero positions outside its test groups, so entering a test group incurs entry turnover cost while v1 trades continuously; this is a conservative bias against the clean book."
}
```

## EXP-v3-004 H2 statistic correction (2026-11-21)
- The unheaded "Correction" line after EXP-v3-002 (1434 vs 1383 forecast dates) refers to EXP-v3-002.
- Review found the EXP-v3-003 CI was computed on a different statistic than the prereg wording: it bootstrapped the Sharpe difference of the path-averaged return series, not the mean over paths of (clean - v1) Sharpe.
- Recomputed with `scripts/v3/h2_cpcv.py --from-cache` (signals from `results/v3/runs/clean_signal_cpcv.parquet`). No change to model, data or settings (2000 draws, mean block 21, seed 0). Days resampled jointly across the 5 clean paths and v1.
- Old (averaged series): CI [-0.299, 1.162]. New (prereg statistic): mean_path_diff 0.387, CI [-0.321, 1.069], DSR 0.660 (across-path-variance sensitivity 0.830).
- Verdict before: H2 NOT supported. After: H2 NOT supported (unchanged).

## EXP-v3-005 H2 day-set alignment (2026-11-21)
- Issues: (1) headline `clean_ann_sharpe` was the Sharpe of the 5-path averaged series, not the per-path mean its CI describes; (2) point estimate used per-path day sets (4129 days) while `days` said 4133 and the bootstrap used `pdf >= d0` rows.
- Fix: one common day set = days from the first full feature row where all 5 paths and v1 are non-NaN (4128 days). Point estimate, bootstrap and DSR all use it. `clean_ann_sharpe`/`v1_ann_sharpe` are now path-mean / common-set values; averaged-series Sharpe kept as `clean_ann_sharpe_avg_series` (secondary). No model, data or setting change; recomputed with `--from-cache`.
- Old: mean_path_diff 0.387, CI [-0.321, 1.069], DSR 0.660 (0.830 across-path var), days 4133, clean Sharpe 0.799 (avg series) / 0.737 (path mean), v1 0.346.
- New: mean_path_diff 0.390, CI [-0.299, 1.075], DSR 0.687 (0.854 across-path var), days = bootstrap_n = 4128, clean Sharpe 0.741 (path mean) / 0.817 (avg series), v1 0.351.
- Verdict before: H2 NOT supported. After: H2 NOT supported (unchanged; CI spans 0, DSR < 0.90).

## EXP-v3-006 DSR series note (2026-11-21)
The H2 deflated Sharpe (EXP-v3-003 to 005) is computed on the 5-path averaged daily return series over the common day set (Sharpe 0.817 ann.), not on the per-path mean (0.741). The averaged series has the higher Sharpe, so this choice favours clean_blend; it still fails the 0.90 bar (0.687). No number changed.

## EXP-v3-007 results run: books, E1-E3, results_v3.md (2026-11-21)
- Ran `scripts/v3/make_results_v3.py` once (no crash, nothing tuned). Books: v1_blend, v1_weekly (v1 signal refreshed every 5 days), tsfm_blend (cached TimesFM signal, held <= 5 days), clean_blend (MLP trained once on 2001-2017, seed 0, 20 epochs, applied from 2018; training rows within 252 days of 2017-12-31 dropped so no target uses post-2017 returns; `clean_full_signal` in `trend/v3/clean_model.py`). Signal at close t, trade t+1; v1 costs + roll.
- Descriptive/exploratory only. TimesFM rows before 2025-10-01 `contaminated` (`+seen` from 2018). H1 pending (first read 2027-11-30). Nothing is out-of-sample.
- Deviation: the panel has 39 instruments; the preregistration says "37-market" (v1 wording). Nothing changed because of it.
- Choices not in the brief: (a) `recent` period also reported split at 2025-10-01; (b) E2 "clean" hit rate uses the clean MLP short-horizon head (per-horizon predictions cached in `clean_pred_full.parquet`), not the blend; (c) E3 cost x2 and capacity use 2018-01..2026-10 (capacity: lag 2, v2 `execution.py` pattern and both ADV variants); (d) each book starts on its first non-zero-weight day.
- Re-run with TREND_QUICK=1 (cached clean signal) reproduced the four CSVs byte-for-byte.

## EXP-v3-008 results wording and E2 common-set fixes (2026-11-21)
- Wording: EXP-v3-007 and results_v3.md said clean_blend was "trained once on 2001-2017". Actual training rows (same logic as `_train_xy`/`clean_full_pred`): feature dates 1999-12-23 to 2017-01-09 (4397 dates, 156857 rows); the 252-day purge drops rows after 2017-01-09 so no target uses post-2017 returns. Not a leak; wording only.
- E2: removed the duplicate, mislabelled `clean_short` "all before 2025-10-01" row (all four "all before" rows dropped; clean has no pre-2018 predictions). Hit rates now use a common (date, inst) set across tsfm, v1 and clean within each band (pre-2018: tsfm and v1 only). New n: pre-2018 33124, 2018-01..2025-09 15600, after cutoff 1911. Pre-2018 tsfm 0.511 (was 0.511), v1 0.498 (was 0.498); other bands unchanged to 3 dp.
- E2 conclusion added: 2018-01..2025-09 -> after cutoff: tsfm 0.522 -> 0.500, clean 0.525 -> 0.510, v1 0.498 -> 0.462. All fall; not flagged as memorisation.
- Added assert: engine lag-1 net for v1_blend equals model.backtest net within 1e-10. Passed.
- Results note: crisis and capacity tables have no Sharpe CI (cumulative return / capacity); full rerun trains on CUDA, may not be bit-identical, committed cache is the reference.
- No model, signal or setting changed. Rerun with TREND_QUICK=1; summary_by_period.csv byte-identical.

## EXP-v3-009 forward pipeline live check (2026-11-21)
- Bug: `forward.py --refresh` called `forecast_tsfm.main(cpu=True)`, which set CUDA_VISIBLE_DEVICES="" after torch had already initialised CUDA. timesfm then saw `cuda.is_available()` True but `device_count()` 0, so `global_batch_size = per_core_batch_size * 0 = 0` and `forecast()` failed `assert global_batch_size > 0`.
- Fix: `TimesFMForecaster(device=...)`; "cpu" moves the inner model to CPU and sets its device/device_count before `compile`. `forecast_tsfm.main(cpu=)` passes device; no env var. `forward.py` uses auto device. New `--refresh-signals` (no download). Cache extension now anchored on the last cached date (`extend_schedule`: +5, +10, ... trading days), not R.index[0].
- Test: RUN_TSFM_TESTS=1 `test_timesfm_cpu_forecast_shapes` passed (11 passed in tests/test_v3_tsfm.py, 10.5 s); full suite 49 passed, 1 skipped.
- Live `forward.py --refresh`: "forecast dates added 5, rows 153579, last 2026-11-17"; "data ends 2026-11-20; forward start 2026-12; new rows 0". Then `forecast_tsfm.py --cpu`: "cache up to date".
- Note: the re-download changed history. clean_signal_full.parquet (frozen model, unchanged) differs from the committed copy on 35 of 2266 old dates (max abs 0.23, MXN), 2018-01-17 to 2026-10-13, so the source data (Yahoo/FRED) was revised. Old tsfm rows are unchanged (cache is append-only). No forward row is affected (forward starts 2026-12-01).
