# Experiment log (append-only)

Every run that informed a design choice. Development = 2001-2012, validation = 2013-2017. The test period was
locked in code for all of these.

| ID | Date | Period | What changed | Why | Result |
|---|---|---|---|---|---|
| EXP-000 | 2026-10-13 | - | Data audit: 39 instruments, Yahoo + FRED | - | Found CL negative price (Apr 2020), 6J and PL bad prints, livestock/ag roll gaps |
| EXP-001 | 2026-10-13 | dev | First cleaning rule: zero any |z| > 8 | remove bad prints | **Rejected:** it also deleted real shocks (SNB 2015, Brexit, Feb 2018, oil 9 Mar 2020). Replaced by the 3-rule filter in `data.clean_returns` (35 values changed, all logged in `data/processed/cleaning_log.csv`). |
| EXP-001b | 2026-10-13 | dev+val | Flat 1-5 bp cost per sector | first pass | Cost drag 4.9%/yr; 2-year notes were charged ~10x their real cost because their low vol needs large notionals. Replaced with per-contract costs (half tick + $2, x1.5). Drag fell to 2.3%/yr. |
| EXP-002 | 2026-10-13 | dev+val | sign() signal -> tanh(t-stat); 10% no-trade band | turnover was ~70x position per year: the sign of a near-zero trend flips on noise | Drag 2.3% -> 1.3%/yr (dev). Signal form follows Baltas & Kosowski (2013). |
| EXP-003 | 2026-10-13 | dev+val | Bonds and FX: Yahoo front-month futures -> FRED carry-correct excess returns | The rates book lost money in 2001-2012, a bond bull market. Cause: unadjusted front-month series drop at each roll by the calendar spread, which for bond and FX futures is the carry. | Rates dev Sharpe -0.15 -> 0.28, FX -0.02 -> 0.45. Data-correctness fix; flagged because a poor result led us to it. Commodity roll gaps remain (limitation L1). |
| EXP-004 | 2026-10-13 | dev+val | Added `disclosed_mix` book (N-CSR sector weights) | Layer C comparison | dev+val Sharpe 0.41 vs 0.35 equal-risk. Reported only, no hypothesis. |
| - | 2026-10-13 | dev+val | **Frozen.** Blend 0.35, long 0.68, short 0.08, medium 0.14; corr with the manager's fund 0.69 (2013-17) | | Pre-registration written; best single horizon = long. |
