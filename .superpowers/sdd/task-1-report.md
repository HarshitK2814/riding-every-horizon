# Task 1 report

Status: DONE_WITH_CONCERNS. Commit 7126dd1 "v3: TimesFM wrapper and point-in-time panel forecaster" (branch v3, not pushed).

## Built
- src/trend/v3/__init__.py, src/trend/v3/tsfm.py (verbatim from brief; installed timesfm 3.0.2 API matched, no adaptation)
- tests/test_v3_tsfm.py (5 tests), scripts/v3/probe_tsfm.py, pyproject.toml extra `v3`, research/experiment_log_v3.md

## Install
`.venv/Scripts/python -m pip install "timesfm[torch]>=2.0"` -> timesfm 3.0.2 from PyPI. Dry run first: only timesfm added, torch stays 2.8.0+cu128 (CUDA True). TimesFM_2p5_200M_torch exists, so no git install. Model google/timesfm-2.5-200m-pytorch downloaded from HF.

## TDD
- RED: ModuleNotFoundError: No module named 'trend.v3'.
- GREEN: 5 passed. Full suite: 27 passed (22 baseline + 5).

## Probe (shapes and timing only)
series 390, seconds 190.1, s/512 series 249.62, cols [date, inst, horizon, mu, width, sig]
projected full run: 454.4 min (> 240 min; noted in the log, prereg fixes 5-day cadence anyway).
Data: returns.parquet is 7166 x 39.

## Concerns
- 454 min is slow. ForecastConfig.per_core_batch_size defaults to 1; raising it is likely the main speedup (not changed, out of brief). Probe time also includes first-call warmup, and the model download happened before timing.
- Windows symlink warning from huggingface_hub (harmless).
- Git warns LF->CRLF on pyproject.toml (harmless).

## Fix round 1
Changes
- tsfm.py: `TimesFMForecaster(per_core_batch_size=256)` passed to ForecastConfig; stores `max_horizon`, `per_core_batch_size`. `forecast_panel` raises ValueError on horizon > forecaster `max_horizon`, and on point/quant shape != (n, hmax) / (n, hmax, 10).
- Timesfm source: forecast() pads the input list up to a multiple of global_batch_size (= per_core_batch_size x device_count) and decodes one global batch at a time; decode always runs max_horizon steps.
- probe_tsfm.py: removed numpy import, named `horizons` dict, `len(out) // len(horizons)`.
- tests: +4 (9-quantile raises, horizon over capacity raises, {21,63} gives 2 rows per (date, inst), batch=2 vs 512 with 3 instruments identical).

Batch-size test (512 real series, ctx 1024, horizon 256, RTX 4070): 256 ok 2.7 s, peak 4.08 GiB; 128 ok 3.2 s, 3.32 GiB; 64 ok 4.5 s, 2.55 GiB. No OOM. Chose 256.

Batched (256) vs unbatched (1), 8 series: point max abs diff 2.8e-4 (1.7e-3 of series std); quantiles 4.2e-4 (2.3e-3 of std). Under 1e-3 in absolute terms, slightly above 1e-3 if scaled by series std; consistent with TF32 matmul noise.

Tests: `.venv/Scripts/python -m pytest tests/test_v3_tsfm.py -q` -> 9 passed. Full suite: see commit summary below.

Probe after fix: 390 series in 3.3 s, 4.28 s/512, projected full run 7.8 min (was 454.4). Logged as EXP-v3-000b.
Full suite: 31 passed.
