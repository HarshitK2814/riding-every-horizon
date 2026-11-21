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
