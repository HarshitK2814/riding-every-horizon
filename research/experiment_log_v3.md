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
