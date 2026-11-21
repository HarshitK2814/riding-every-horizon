# Data vintage for P7 v3 results

- Reference vintage: downloaded **2026-11-21** (Yahoo Finance via yfinance, FRED).
- Last market date: **2026-11-20** (close, volume, returns); last FRED date: 2026-11-19 (DTB3, fred_macro).
- Replaces the 2026-10-13 download (ending 2026-10-13), which Yahoo has since revised (e.g. MXN) and which is not recoverable.
- All v3 tables, runs, caches and research/results_v3.md are regenerated on this vintage. v1/v2 results stay sealed on their own vintages.
- A copy of these files is committed under `snapshot/` (reproducibility only).
- Not read by the v3 pipeline: data/raw/proxies_v2.parquet (v2 audit only; not part of the snapshot).

| file | sha256 |
|---|---|
| data/processed/benchmarks.parquet | `df20d6bb948f6f0899b662f6e702ecd0a281c907fa90775d492fb375f492416b` |
| data/processed/cleaning_log.csv | `3a0008a8cb1075e0c5ed0e65df4f1c88357227547363b84826500d8913e6dbdc` |
| data/processed/returns.parquet | `c80dfcc1e5929c191ac410e849c58c90cb478fce2581605d6acb55505dd474be` |
| data/raw/close.parquet | `bdfa904578fb23cf05357098ad00f277eb36ee8f5e84c816387391f500854a36` |
| data/raw/dtb3.parquet | `88a8b56c92a265fa4c990ba2b26d4084c84da9d43076c86ac4ae01dc94a8edc1` |
| data/raw/fred_macro.parquet | `86f8d5a47f4c5918060a0189e4de56a94448d0beea3be342f2c7c6fe71a34c09` |
| data/raw/volume.parquet | `536619185eb5eaed2bc965d269d513cf0b59a086284ad5f8dd04b2c2902f34ac` |
