# P7 evidence for v2 (trend following)

Checked 2026-10-29. All links were opened; findings below are what the pages say, not our reading of the full papers unless stated. Fund filings quoted are descriptions of the manager, not performance claims.

## Sources

| # | Source | Type | One-line finding | Link |
|---|---|---|---|---|
| S1 | Hurst, Ooi, Pedersen, "A Century of Evidence on Trend Following Investing" (AQR, JPM 2017) | Firm research | Time-series momentum profitable across 110 years of multi-asset data; tests the "fluke" question and the effect of rising assets and higher correlations. Our v1 horizons (1/3/12 months) come from this paper. | [AQR](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing), [reprint](https://fairmodel.econ.yale.edu/ec439/hurst.pdf) |
| S2 | Hedge Fund Journal on Man AHL (Hamill, Rattray, van Hemert 2016, "Trend Following: Equity and Bond Crisis Alpha") | Firm research (secondary report) | Trend did well in the 1960-85 bond bear market, but the 30-year AHL record sits inside a bond bull market. Warns bonds and equities can fall together. | [article](https://thehedgefundjournal.com/man-ahl-marks-30-years/) |
| S3 | Campbell & Company, "Deconstructing Futures Returns: The Role of Roll Yield" (HFJ, Mar 2014) | Firm research (our source manager) | Cites an earlier Campbell paper: "nearly half of the cumulative performance of a trend-following strategy was attributable to roll yield." | [article](https://thehedgefundjournal.com/deconstructing-futures-returns/) |
| S4 | Campbell Strategic Allocation Fund LP, 10-K FY2015 | SEC | Margin-to-equity "typically 10% to 30%" and "rarely exceeds 30%"; real-time position monitoring. Gives a leverage sanity band. | [SEC](https://www.sec.gov/Archives/edgar/data/910467/000091046716000020/csaf10k_dec2015.htm) |
| S5 | pysystemtrade (R. Carver), backtesting docs | Public code (GPL-3.0, ~3.6k stars) | Position buffering (default 10%, trade to buffer edge), forecast scalars and caps, costs charged in Sharpe-ratio units via turnover, stitched (Panama) prices for signals but not as the denominator for % returns. | [docs](https://github.com/pst-group/pysystemtrade/blob/master/docs/backtesting.md) |
| S6 | Carver, *Systematic Trading* ch.12, as summarised by CXO Advisory | Practitioner book (secondary) | Costs should be at most one third of a conservative pre-cost Sharpe; set a maximum turnover ("speed limit"); faster rules need extra evidence. | [CXO](https://www.cxoadvisory.com/big-ideas/a-few-notes-on-systematic-trading) |
| S7 | Quantpedia, "Continuous Futures Contracts Methodology for Backtesting" | Practitioner blog | Recommends backwards-ratio adjustment with a simple first-of-month roll, because funds trade constant % of assets; unadjusted splices created fake jumps that flipped a profitable test to unprofitable. | [page](https://quantpedia.com/continuous-futures-contracts-methodology-for-backtesting/) |
| S8 | Darolles and Chevalier (2022), "Futures market liquidity and the trading cost of trend following strategies" | Academic, institutional trade data | Real manager data; recent weak trend returns tied to lower futures volatility plus a higher share of fixed costs. | [HAL/RePEc](https://ideas.repec.org/p/hal/journl/hal-04581741.html) |
| S9 | Etienne et al. (2025), "Revisiting the Structure of Trend Premia" | Academic (arXiv) | The 6-12 month horizon is largely redundant with its neighbours; a short plus long "barbell" keeps most performance. Weights learned per asset with turnover control. | [arXiv](https://arxiv.org/abs/2510.23150) |
| S10 | Kurth, Eisler, Rej, Bouchaud (2026), "Is Trend Still Your Friend?" | Academic (arXiv, CFM) | Short-term trend PnL collapsed after 2008 on small-tick contracts across all horizons; large-tick contracts intact. Capacity explanations rejected. About 100 futures, 1995-2025. | [arXiv](https://arxiv.org/abs/2607.01550) |
| S11 | Harvey et al. (Man AHL, Duke), "The Impact of Volatility Targeting" (JPM 2018), via Quantpedia | Firm research | Vol scaling lifts Sharpe for equities and credit, barely changes it for bonds, currencies, commodities; cuts tail risk everywhere. Their example uses a 20-day half-life estimate and does not compare estimators. | [Quantpedia](https://quantpedia.com/the-impact-of-volatility-targeting-on-equities-bonds-commodities-and-currencies), [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf) |

**Gaps (stated, not hidden).** No Reddit or Quant StackExchange thread could be opened and verified in this pass (the one forum hit, NexusFi, did not load), so no forum claim is used. No Winton or Aspect note was found. No head-to-head GARCH vs EWMA test for trend portfolios was found; plan 05 Task 5 stays an original test. Man's own marketing pages were skipped as promotional.

## What industry does that v1 does not

1. **Charges cost in Sharpe units and caps turnover** (S5, S6). v1 estimated per-contract costs from 2024 tick sizes and reported them after the fact.
2. **Uses a no-trade buffer** (S5): default 10% of position; trade only to the buffer edge. v1 rebalances to target.
3. **Treats roll yield as first-order** (S3, S7). v1 limitation L1: Yahoo front-month commodity series jump at each roll, and the cleaner removes only the 30 largest gaps, so roll yield is missing from commodity books.
4. **Separates the price used for signals from the price used for returns** (S5, S7): back-adjusted for signals, ratio-adjusted or unadjusted-with-explicit-roll for % returns.
5. **Reports a leverage band** (S4): 10-30% margin to equity. v1 targets 10% volatility with no margin check.
6. **Checks horizon redundancy and per-sector decay** (S9, S10). v1 assumes three horizons and one parameter set for all markets.

## v2 decisions

| # | Decision | Tag | Feeds | Reason |
|---|---|---|---|---|
| D1 | Audit every continuous series for roll gaps; compare v1's 30-gap clean vs backwards-ratio with an explicit first-of-month roll; record adjustment method, roll rule and as-of date in the results | **fix** | -> 05 Task 1 roll audit | S3 (roll yield about half of trend PnL) and S7; v1 L1 |
| D2 | Add a position buffer (10% default, trade to edge) and report turnover and cost in Sharpe units next to PnL | **copy** | -> 05 Task 3 execution realism | S5, S6 |
| D3 | Pre-register a cost ceiling: net cost at most one third of pre-cost Sharpe, plus a turnover speed limit; flag any horizon or sector that breaks it | **copy** | -> 05 Task 2 prereg, Task 3 | S6 |
| D4 | Add a margin-to-equity report (target path vs the 10-30% band) and a capacity table; do not claim capacity as the cause of any decay | **copy** | -> 05 Task 3 capacity | S4, S10 |
| D5 | Overfitting audit: test whether the 63-day horizon is redundant (barbell of 21 + 252 vs three horizons) and use a deflated or multiple-testing Sharpe | **fix** | -> 05 Task 4 overfitting audit (H2) | S9; v1 H2 already failed vs the 12-month horizon |
| D6 | Break results out by sector and by tick size (small vs large tick) for the seen period; the 2018+ weakness may be small-tick contracts, not "trend" | **fix** | -> 05 Task 3 and Task 8 seen-period re-run | S10 |
| D7 | Keep EWMA as the baseline vol estimator with a stated half-life (20 days in S11) and test GARCH against it per asset class; expect small gains for bonds, currencies, commodities | **copy** baseline, **fix** test | -> 05 Task 5 GARCH vs EWMA (H1) | S11; no direct trend-portfolio test exists |
| D8 | Judge crisis convexity (H3) with a bond-bull-market caveat: report the 2001-2017 CPCV result with and without the bond sector | **fix** | -> 05 Task 6 Hawkes / crash clustering, Task 8 | S2 |
| D9 | Label every figure after 2017 `seen period (v1 reported it)`; state the signal bar (close t, held from t+1) | **copy** (project rule) | -> 05 Task 8 | roadmap |
| D10 | Use real trade-level cost evidence as a sanity check for our tick-size cost model; if our cost is below it, raise ours | **fix** | -> 05 Task 3 | S8; v1 limitation L4 (costs probably too low for 2001-05) |

## Rejected ideas

| Idea | Why rejected |
|---|---|
| Treat "crisis alpha" from the Man AHL record as evidence of convexity | S2 itself notes the track record sits inside a bond bull market; we test H3 on our own data with a sector split instead. |
| Blame capacity or crowding for recent weakness | S10 tests and rejects it (timing, size, heterogeneity); we have no data to test it either. |
| Add short-term (<30 day) momentum or mean-reversion to P7 | S10 says short-term trend has decayed on small-tick contracts; Campbell's short-term models are P8 scope (v1 `institutional_methodology.md`). |
| Copy Carver's fixed forecast scalar and cap (average absolute forecast 10, cap 20) | Those numbers are his design, not tested on our data; adopt only the buffering and cost-in-Sharpe ideas. |
| Use fund Sharpe or performance claims from firm marketing pages | Promotional and past-performance disclaimers apply; used only as descriptions of method. |
