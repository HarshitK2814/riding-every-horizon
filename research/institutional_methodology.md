# What the manager discloses (Layer A)

Source: Equinox Funds Trust, **Form N-CSR** for the Equinox Campbell Strategy Fund, fiscal year ended
30 Sep 2019 (CIK 1498272, accession 0001580642-19-005370). Saved as
`research/sources/equinox_campbell_ncsr_2019.htm`. The fund delegates its futures book to Campbell & Company,
one of the oldest systematic trend-following managers. Quotes are verbatim.

| # | Quote | What we take from it |
|---|---|---|
| Q1 | "The trading strategy employed by Campbell is a diversified intermediate to long-term multi-strategy (but predominantly trend-following) approach." | The core is trend following across horizons. |
| Q2 | "the Campbell Program's (and consequently the Fund's) market exposure continues to be diversified across six sectors." | Six instrument groups. |
| Q3 | "Currencies, Interest Rates, and Equity Indices, in that order, represent the largest gross exposures, and total almost 83% ... while the physical commodity sectors, Metals, Agricultural Commodities, and Energy (in that order) total the remaining 17%." with the table "Currencies 37.7% Interest Rates 24.6% Equity Indices 20.9% Ag Commodities 7.6% Metals 6.1% Energy 3.1%" | The six sectors and their mix (used for the `disclosed_mix` book). |
| Q4 | "This was also the case for Campbell's medium- and long-term trend models, which took losses in equity indices before reducing long exposure and eventually becoming net short" | Separate medium- and long-horizon trend models. |
| Q5 | "Short-term mean-reversion (<10-day holding period) and short-term momentum (<30-day holding period) strategies were very impactful in January" | A short-horizon (< 30-day) momentum component. |

## The three layers

* **A. Disclosed:** six sectors (Q2, Q3); short-, medium- and long-term trend models (Q4, Q5); predominantly
  trend following (Q1).
* **B. Our reconstruction (not disclosed; from the public literature):** horizons of 21, 63 and 252 trading days
  (the 1-, 3- and 12-month look-backs of Hurst, Ooi and Pedersen 2017), volatility-scaled positions, equal risk per
  sector, a 10% portfolio volatility target, per-contract costs. Campbell's actual signals, horizons, weights and risk
  model are proprietary; we do **not** claim to reproduce them, only a rule built from what is disclosed.
* **C. Extensions:** the disclosed-mix sector budget; a crisis-convexity test; validation against the fund's own NAV.

Short-term mean reversion (Q5) is out of scope here; it belongs to the multi-style project (P8).
