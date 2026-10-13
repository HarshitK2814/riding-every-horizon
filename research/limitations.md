# Limitations

* **L1 Commodity roll gaps.** Yahoo continuous front-month commodity series jump at each roll by the calendar
  spread. The cleaning rule removes the largest gaps (30 values), but the commodity books still miss roll yield
  (carry). Bonds and currencies were rebuilt carry-correct from FRED (EXP-003); equities keep a small bias of
  roughly (r - d)/4 per quarterly roll.
* **L2 Synthetic bonds** are constant-maturity par Treasuries: no roll-down, no futures delivery option, and only
  US maturities (no free daily non-US yields).
* **L3 Cash-index proxies** for two European equity indices (price indices, not futures).
* **L4 Costs** are estimated from tick sizes at 2024 prices, not measured. They are probably too low for 2001-05
  and too high for liquid contracts today. No market-impact or capacity model (positions are in NAV units).
* **L5 Not the manager's model.** The filing discloses sectors and horizon types only. Our horizons, signal form
  and risk model come from the public literature (Layer B). A 0.52 correlation means we capture the common trend
  factor, not the manager's own alpha.
* **L6 One test window** of 84 months, so Sharpe CIs are wide. Design changes EXP-001 to EXP-004 were made on
  dev+val, including one (EXP-003) that a poor development result prompted.
* **L7 Fund NAVs** come from Yahoo adjusted closes; distributions show up as small jumps in some years.
