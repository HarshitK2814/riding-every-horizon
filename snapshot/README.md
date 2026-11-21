# snapshot/

Frozen copy of the data vintage used for all P7 v3 results.

- Mirrors `data/processed/` and `data/raw/` (only the files the v3 pipeline reads).
- Sources: Yahoo Finance via yfinance (continuous futures, volume, benchmarks); FRED (DTB3, macro series).
- Vintage: downloaded 2026-11-21, last market date 2026-11-20.
- SHA256 of every file: `results/v3/DATA_VINTAGE.md`.
- For reproducibility only. Not a data product; vendor terms apply.
