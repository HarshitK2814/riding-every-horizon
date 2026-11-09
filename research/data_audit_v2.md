# Data audit v2 (EXP-V2-001)

Period used for every number below: 2001-01..2017-12 (development + validation, 4385 days). Nothing from the seen period.
Code: `src/trend/v2/audit.py`, `scripts/v2/audit.py`. Tables: `results/v2/tables/roll_gaps.csv`, `adv_check.csv`, `roll_audit_sharpe.json`.

## Method
A day is flagged when a future's return departs from a non-rolling proxy by more than 5 robust sigmas
(trailing 250-day MAD, beta and scale lagged one day). Proxies: ^GSPC (ES), ^NDX (NQ), ^DJI (YM), ^N225 (NKD), GLD (GC), SLV (SI).
All other instruments use the leave-one-out sector median return as the proxy. Flagged returns are replaced by the same-day sector median
(`impute_flagged`). Rates and FX are the FRED carry-correct synthetic series (EXP-003), so their flags are not Yahoo roll gaps.

## Flags by sector
| sector | n_instruments | n_flags | mean_abs_gap |
|---|---|---|---|
| ags | 11 | 238 | 0.08155 |
| energy | 5 | 149 | 0.09038 |
| equity | 6 | 219 | 0.03731 |
| fx | 8 | 130 | 0.0311 |
| metals | 5 | 152 | 0.0673 |
| rates | 4 | 81 | 0.009474 |

## Flags by instrument
| ticker | sector | proxy | n_flags | mean_abs_gap | share_in_expected_roll_window |
|---|---|---|---|---|---|
| ES=F | equity | ^GSPC | 54 | 0.02316 | 0.4074 |
| NQ=F | equity | ^NDX | 58 | 0.02509 | 0.3793 |
| YM=F | equity | ^DJI | 57 | 0.02165 | 0.4386 |
| NKD=F | equity | ^N225 | 15 | 0.06372 | 0.1333 |
| ^STOXX50E | equity | sector median | 17 | 0.0504 | 0.1176 |
| ^FTSE | equity | sector median | 18 | 0.03985 | 0.1111 |
| GC=F | metals | GLD | 60 | 0.02184 |  |
| SI=F | metals | SLV | 40 | 0.04729 |  |
| HG=F | metals | sector median | 9 | 0.08911 |  |
| PL=F | metals | sector median | 27 | 0.06574 |  |
| PA=F | metals | sector median | 16 | 0.1125 |  |
| CL=F | energy | sector median | 46 | 0.06684 |  |
| BZ=F | energy | sector median | 30 | 0.04623 |  |
| NG=F | energy | sector median | 15 | 0.166 |  |
| HO=F | energy | sector median | 17 | 0.07632 |  |
| RB=F | energy | sector median | 41 | 0.09652 |  |
| ZC=F | ags | sector median | 25 | 0.07343 |  |
| ZW=F | ags | sector median | 6 | 0.08093 |  |
| ZS=F | ags | sector median | 20 | 0.07672 |  |
| ZM=F | ags | sector median | 23 | 0.1031 |  |
| ZL=F | ags | sector median | 2 | 0.06217 |  |
| KC=F | ags | sector median | 13 | 0.09323 |  |
| SB=F | ags | sector median | 12 | 0.09493 |  |
| CT=F | ags | sector median | 18 | 0.0846 |  |
| CC=F | ags | sector median | 5 | 0.09054 |  |
| LE=F | ags | sector median | 44 | 0.0453 |  |
| HE=F | ags | sector median | 70 | 0.09208 |  |
| UST2 | rates | sector median | 16 | 0.003209 | 0.3125 |
| UST5 | rates | sector median | 23 | 0.005969 | 0.3913 |
| UST10 | rates | sector median | 19 | 0.01335 | 0.3158 |
| UST20 | rates | sector median | 23 | 0.01537 | 0.3478 |
| EUR | fx | sector median | 18 | 0.01553 | 0.1667 |
| JPY | fx | sector median | 20 | 0.02983 | 0.2 |
| GBP | fx | sector median | 12 | 0.03302 | 0.25 |
| AUD | fx | sector median | 15 | 0.04653 | 0.1333 |
| CAD | fx | sector median | 3 | 0.0248 | 0 |
| CHF | fx | sector median | 20 | 0.03341 | 0.2 |
| NZD | fx | sector median | 16 | 0.03293 | 0.375 |
| MXN | fx | sector median | 26 | 0.03273 | 0.2308 |

`share_in_expected_roll_window` = share of flags on days 8-21 of Mar/Jun/Sep/Dec (quarterly-cycle sectors only). If flags were mostly roll gaps this would be high.
Chance level is about 16% (4 of 12 months x 14 of 30 days of the month). ES/NQ/YM sit near 40%, so a good part of their flags are roll gaps; NKD, STOXX, FTSE and the rates/FX synthetics are near chance, so their flags are mostly fat-tail days.
Peer-median flags in commodities, metals, energy, ags are NOT clean roll detections: a real sector-wide shock or a single-name move also trips them. Treat them as an upper bound.

## Before / after (v1 blend, net, 2001-2017, lag 1, v1 costs)
| series | Sharpe | 95% CI (stationary bootstrap, block 21) |
|---|---|---|
| v1 data | 0.359 | [-0.125, 0.861] |
| flagged days imputed (969 values) | 0.547 | [0.058, 1.044] |
| difference | 0.188 | CI half-width of v1 Sharpe = 0.493 |

## Decision
Rule (plan): use the imputed series in v2 if it moves dev+val Sharpe by more than its own CI half-width. The change is 0.188 against a half-width of 0.493.
**use_imputed_in_v2 = False**: v2 keeps the v1 data and reports the imputed run as a sensitivity. The point estimate rises, which is the direction a roll-yield fix would take,
but it is well inside the noise. The Sharpe numbers are in-sample audit numbers on dev+val (no hypothesis is tested here).

## ADV check (Yahoo front-month volume, median of the 20-day median over the last 250 days, vs `qengine/src/qengine/data/contracts.yaml`)
| synthetic | future | yahoo_adv_20d_median | yaml_adv_contracts | ratio_yahoo_over_yaml | off_by_gt_2x |
|---|---|---|---|---|---|
| UST2 | ZT=F | 6.855e+05 | 600000 | 1.142 | False |
| UST5 | ZF=F | 1.097e+06 | 900000 | 1.219 | False |
| UST10 | ZN=F | 1.675e+06 | 1600000 | 1.047 | False |
| UST20 | ZB=F | 4.152e+05 | 400000 | 1.038 | False |
| EUR | 6E=F | 2137 | 200000 | 0.01068 | True |
| JPY | 6J=F | 402.5 | 150000 | 0.002683 | True |
| GBP | 6B=F | 247.5 | 100000 | 0.002475 | True |
| AUD | 6A=F | 178 | 100000 | 0.00178 | True |
| CAD | 6C=F | 71.25 | 80000 | 0.0008906 | True |
| CHF | 6S=F | 2.45e+04 | 40000 | 0.6125 | False |
| NZD | 6N=F | 3.334e+04 | 30000 | 1.111 | False |
| MXN | 6M=F | 23 | 70000 | 0.0003286 | True |

Rates (ZT/ZF/ZN/ZB), CHF and NZD agree with the yaml within 2x. For 6E, 6J, 6B, 6A, 6C and 6M the Yahoo volume is 100-3000x below the yaml (6C: about 71 contracts a day; the
real CME CAD contract trades far more). That is a Yahoo data defect (the free feed reports a tiny fraction of FX futures volume), not evidence the yaml is wrong.
**Decision: contracts.yaml NOT changed** (the plan's "correct if off by >2x" rule would have cut FX capacity by 3 orders of magnitude on bad data). The yaml ADVs remain
analyst estimates; capacity for FX in Task 3 should be read with that caveat. Not verified against CME volume reports (no free source opened in this pass).

## Not done (future work)
`evidence_v2.md` D1 recommends backwards-ratio adjustment with an explicit first-of-month roll. That needs individual contract prices, which Yahoo does not provide
(only spliced front-month series), so it is not cheap and is not run. The imputation above is the available sensitivity. Future work: build ratio-adjusted series from
free exchange settlement files or a contract-level source and re-run.

## Limitations
- Flags are statistical, not confirmed roll dates. Imputation also removes some real shocks in commodity peers.
- 5 robust sigma on fat-tailed returns flags about 1% of days even without rolls.
- Bootstrap CI is for the Sharpe of one path; the before/after difference has no separate CI (the two series share most days).
