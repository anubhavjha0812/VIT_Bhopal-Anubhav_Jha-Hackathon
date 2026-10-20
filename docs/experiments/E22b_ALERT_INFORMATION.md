# E22b: Does a credit alert itself carry information?

Rules fixed 2026-10-11, before any alert is counted or any return computed. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** Test 1 (E14) compared "downgrade + full COVID episode" with "the same downgrade + the
  real 20-day market moves", so both sides shared the assumed downgrade and the gap was the
  episode. E22b asks only about the alert: after a gated credit alert, does the company's stock
  do worse than usual, and does a larger predicted alert loss go with a worse reaction?
- **Prediction (from the code, checked in M1 test 3):** `revalue(book, ticker, notches, episode,
  multiple=0)`. With `multiple = 0`: Aaa, Baa, z, USD/INR and the Treasury stay at their
  31 Dec 2019 levels; the obligor moves `notches` down (`parse_notches`, 1 if none found, D for
  Default) and its spread widens to the new rating's spread; every other name widens by that
  widening × its return correlation (contagion, ≥ 0); PDs: the obligor gets its new rating's
  PD, all others keep theirs (the Vasicek PD at z = 0 is below the normal PD, so the max keeps
  the normal one). The loss is the revalued change in the obligor's revolver ECL (stage changes
  at 3+ notches or a fall below BBB−), its bonds, every name's bonds and CDS through contagion,
  and swap CVA. The book and ratings are those of 31 Dec 2019 for every alert, so the USD size
  is illustrative; the predicted direction is always "loss".
- **Alerts:**
  - **Primary sample, history:** the E17 history set (`data/cache/history_signals.csv.gz`,
    FNSPID news headlines, 1 Jan 2010 – 8 Apr 2020), rows kept by the E17 relevance rules with
    role `subject`. Event = the E13 credit rules (`credit_event`; the engine's model never
    overrides them); impact = `impact_formula` with the stored E17 sentiment (full-precision
    scores, as E17) and source `news`; then `gate(…, trigger = 7)` unchanged. These years were
    never used to write the credit rules (E13 used the 2020 replay window), so they are out of
    sample for the rules. No parameter is fitted in E22b.
  - **Secondary, already seen:** the 4 alerts of the replay window (9 Apr – 10 Jun 2020), the
    E14 Test 1 alerts, scored the same way and reported apart.
- **Dates and window:** the publication timestamp's UTC calendar date is the alert date. Start =
  the close of the last trading day strictly before that date (no part of the reaction is
  missed, nothing after is used to place the start). End = the close 20 trading days after the
  start. Prices: `data/prices/prices_history.csv` (Yahoo closes, adjusted).
- **Observed outcome (a proxy, not a credit loss):** the obligor's market-adjusted 20-day
  return: log(close_end / close_start) − the same for `^GSPC`. Company credit spreads and CDS
  quotes are not available free, so the stock is used: equity falls when credit quality falls,
  but it also moves on everything else. This measures the equity reaction, never the credit
  loss, and no USD loss error is computed (the two are in different units).
- **Independence and exclusions:** per company, an alert within 20 trading days of an earlier
  counted alert of the same company is dropped (counted, not scored), so windows of one company
  never overlap. Alerts on different companies may overlap in time (stated). Alerts whose start
  or end close is missing are dropped and counted. A company not in the book cannot be revalued
  and is dropped and counted.
- **Baseline (naive):** the same company's market-adjusted 20-day return from every trading day
  in 2010-01-01 – 2020-04-08 (start-day windows defined the same way), with no alert
  information. Its share of negative windows per company is the base rate.
- **Metrics:**
  1. **Direction (primary):** k = alerts with a negative market-adjusted return, n = alerts
     scored. p0 = the mean of the alerts' company base rates. One-sided exact binomial test of
     k out of n against p0.
  2. Mean market-adjusted return of the alerts with a 95% bootstrap interval (10,000 resamples,
     seed 0), next to the baseline mean (reported).
  3. Spearman correlation between the predicted alert loss (USD) and the market-adjusted return
     across alerts, with its p-value (reported; expected negative if size matters).
- **Pass rule (fixed now):** if n ≥ 20, E22b passes when the one-sided binomial p < 0.05 (alerts
  are followed by a negative equity reaction more often than the company's base rate). If
  n < 20, there is no pass or fail: the result is reported as descriptive and underpowered. The
  Spearman correlation and the secondary sample never decide anything.
- **Outputs:** `data/outputs/e22b_alerts.csv` (one row per alert: date, ticker, text, notches,
  impact, gate reason, start/end dates, returns, predicted loss, kept or why dropped),
  `data/outputs/e22b_results.json`; runner `src/riskengine/module_b/alert_check.py`. E14, E15
  and E22 files are not touched.
- **Limitations:** stock returns are a noisy proxy for credit stress; market-adjusted, not
  beta-adjusted; the alert rules were written for 2020 social and news text and run here on
  news headlines only; the sentiment behind the impact is the full-precision E17 score, not
  the engine's int8; one book and one rating set (31 Dec 2019) for every year; FNSPID's date
  stamps are dates, so same-day timing within a day is ignored.
- **Results (2026-10-11, run once; `data/outputs/e22b_alerts.csv`, `e22b_results.json`; E14,
  E15 and E22 files unchanged, hashes checked).**
  - **Primary (history, 2010 – 8 Apr 2020):** 79 alerts fired; 39 dropped as within 20 trading
    days of an earlier alert on the same company; **n = 40** scored. Negative market-adjusted
    20-day return in 22 of 40 (55%) vs the companies' base rate 47.8%; one-sided binomial
    p = 0.23. **Verdict: FAIL** (p ≥ 0.05).
  - Mean market-adjusted return −1.7% [−4.1%, +0.6%] vs the baseline mean +0.3%. Spearman
    between the predicted alert loss and the return: +0.05 (p = 0.77); no link between Module
    B's predicted size and the reaction.
  - Secondary (replay window, seen): 4 fired, 1 overlapping, n = 3, 2 negative; reported only.
  - *Looked at afterwards (description only; the frozen E13 rules are not changed):* 23 of the
    40 scored alerts are on JPM and BAC, and most of those are not about the bank's own credit:
    commercial-mortgage securitisations named after the bank ("Fitch Downgrades J.P. Morgan
    2001-CIBC3"), an affirmation ("Fitch Affirms Bank of America's IDR"), a counterparty
    ("Bank of America Says MBIA Defaulted"). One NFLX alert is an upgrade (B+ to BB−) read as a
    downgrade. The gate's precision on history news is low; the gold set (E3) is where it
    would be measured. The failure is at least partly a trigger-precision problem, not only a
    pricing one.

## Result

- **Question:** Does a credit alert alone carry information? (alert-only prediction vs the stock's market-adjusted 20-day return)
- **Outcome:** history 2010 – Apr 2020: 79 alerts, 40 independent; 55% negative vs base 47.8%, binomial p = 0.23; mean −1.7% [−4.1%, +0.6%]; predicted size vs reaction Spearman +0.05
- **Verdict:** FAIL; many history alerts are false (bank-named securitisations, affirmations)
- **Stored results:** `e22b_alerts.csv`, `e22b_results.json`, `tests/test_module_b_alert_check.py`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
