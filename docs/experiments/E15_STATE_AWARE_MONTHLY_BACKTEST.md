# E15: State-aware market move, monthly backtest

Rules fixed 2026-10-10, before the code. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** Check 2 (E14) found that the first 5 days of a crisis do not predict its next 60 days
  (analogs $45.7m error vs naive $47.0m). Credit spreads are known to mean-revert: when they are
  already wide, the next months more often bring tightening. This is a new, separate test; Check 2
  stays as reported and is not re-run. The idea came after seeing Check 2's misses, so Check 2's
  episodes are not used to judge it.
- **Data:** FRED daily Baa, Aaa, VIX, USD/INR and S&P 500 (`market_history.csv`, 1990–2023).
- **Origins:** the last trading day of every month, Jan 1995 – Sep 2023 (~345), the last one
  with 60 trading days of data after it.
- **Target:** the 60-trading-day moves from the origin, as in Check 2: change in Baa, Aaa and
  VIX, S&P 500 log return, USD/INR return.
- **State at the origin:** Baa level, VIX level, Baa change over the previous 20 trading days.
- **Model:** for each target, least squares on the 3 state values plus a constant, fitted only on
  earlier month-end origins (from Feb 1990) whose 60-day window ended on or before the origin
  date. Refitted at every origin. Nothing from the future, no tuning.
- **Naive:** the average of the same earlier moves (same data, no state).
- **Valued on the book:** each predicted, naive and realised move is applied to our 31 Dec 2019
  book exactly as in Check 2 (loss in USD and CET1 after).
- **Pass rule:** the model passes if its mean absolute loss error is at least 15% below the
  naive's (model MAE ≤ 0.85 × naive MAE). Also reported, not part of the rule: CET1 error, sign
  match, the share of months the model is closer, and a Diebold–Mariano test with Newey–West
  errors (lag 2, as the 60-day windows of nearby months overlap).
- **Scored once.** Pass or fail, the result goes into the README as it comes out.
- **If it passes:** Module B shows the state-aware 60-day move as a "most likely" view next to the
  stress episodes. Capital (CET1, reverse stress) still uses the stress episodes, because a stress
  test is meant to be severe, not likely. If it fails, Module B stays as it is.
- **Tests:** the training set never includes a window ending after the origin; the naive equals
  the mean of earlier moves; on a synthetic mean-reverting series the model beats the naive.

## Result

- **Question:** State-aware market move, monthly (345 months)
- **Outcome:** model/naive MAE 1.00 (pass ≤ 0.85), DM p 0.933
- **Verdict:** fail
- **Stored results:** `module_b_monthly_backtest.csv`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
