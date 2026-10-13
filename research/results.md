# Results: Riding Every Horizon (P7)

All numbers are net of costs (per-contract half tick + commission, plus roll costs; drag 1.4%/yr for the blend in
the test), monthly, excess of T-bills unless stated. Signals use the close of day t; positions earn day t+1.
Pre-registration hash `2cca1fec34d9` (frozen before the test was read). Test = 2018-01 to 2024-12 (84 months).

## Hypotheses

| | Pre-registered claim | Test result | Verdict |
|---|---|---|---|
| H1 | Blend Sharpe > 0 | 0.22, bootstrap 95% CI [-0.40, 0.75], p = 0.23 | **Not supported** |
| H2 | Blend beats the best single horizon (252d, chosen on dev+val) | Difference -0.31 [-0.75, 0.19], p = 0.91 | **Not supported** (as predicted from dev+val) |
| H3 | Crisis convexity: c > 0 in r = a + b m + c m^2 | c = 2.68, HAC t = 3.60, p = 0.0002 | **Supported** |
| Gate | Correlation with the manager's own fund >= 0.50 | 0.52 (84 months) | Passed: results count as a replication, not "indicative" |

## Books (test 2018-24)

| Book | Return/yr | Vol | Sharpe [Lo 95% CI] | Max DD |
|---|---|---|---|---|
| Three-horizon blend | 2.6% | 11.7% | 0.22 [-0.52, 0.96] | -18.6% |
| Long horizon only (252d) | 6.1% | 11.6% | 0.53 [-0.22, 1.27] | -18.0% |
| Disclosed sector mix (N-CSR weights) | 5.4% | 11.7% | 0.46 [-0.28, 1.21] | -12.7% |

Single-horizon test Sharpes: short 0.06, medium 0.00, long 0.53. By sector: rates 0.50, FX 0.26, energy 0.22,
ags -0.16, equity -0.25, metals -0.36 (`results/tables/sharpe_by_book.csv`).

## Crisis behaviour (total return, blend)

| Window | Trend blend | S&P 500 |
|---|---|---|
| 2008 (development) | +38.5% | -36.5% |
| Q4 2018 | +3.0% | -12.8% |
| 19 Feb - 23 Mar 2020 | +22.9% | -33.4% |
| 2022 | +13.2% | -17.2% |

The 2020 gain came mostly from long Treasuries (+8.6 pp; the Fed cut 150 bp) and short energy (+6.5 pp).
Test-period monthly correlation with SPY is -0.37, with AGG -0.26.

## Does it look like the real thing?

Monthly correlation of the blend (+ T-bill) with real fund NAVs over the test period: the manager's mutual fund
0.52, AQR Managed Futures 0.68, a second large managed-futures mutual fund 0.73, two managed-futures ETFs 0.63 and
0.70 (`results/tables/fund_corr_test.csv`). In validation (2013-17) it was 0.69 with the manager's fund.

## What we learned

1. **Trend following paid exactly when it was supposed to.** All four equity crises above (one in
   development, three in the test) had a positive trend return, and the convexity is highly significant out of sample. The "crisis alpha" claim held.
2. **Its average return was weak out of sample.** The Sharpe is positive but not significant over seven years. Most
   of the edge came from the slow (12-month) signal. Fast signals paid their costs (2.5%/yr for the 21-day book)
   and added little except in crashes.
3. **The multi-horizon design did not beat its simplest component in any period.** Diversifying across horizons
   lowered the Sharpe, consistent with the trend literature, where 12-month look-backs carry most of the premium.
4. **The disclosed sector mix (heavier in currencies and rates) did better than equal risk** (0.46 vs 0.22),
   mainly because the commodity and equity books were weak in the test. Not a pre-registered claim.

## Recent period (2025-01 to 2026-10, 22 months; reported only)
Blend Sharpe 0.43, long horizon 0.73. Too short to test anything.
