# E22: Interest-rate shock in Module B

Rules fixed 2026-10-11, before any code change or run. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** E14 holds the 10-year Treasury at its book-date level (1.92%), so only credit moves.
  In real crises Treasury yields move too; leaving them out is Module B's largest modelling gap.
  E22 adds a controlled rate shock. It is a stress scenario, not a forecast of rates.
- **Shock (inputs read from `market_history.csv`, FRED `DGS10`; not results):** a **parallel
  shift**: every tenor moves by the 10-year change.

  | Scenario | Dates | UST10Y start → end | Δ (pp) |
  |---|---|---|---|
  | GFC 2008 | 15 Aug – 15 Dec 2008 | 3.84 → 2.53 | −1.31 |
  | Euro 2011 | 1 Jul – 4 Oct 2011 | 3.22 → 1.81 | −1.41 |
  | COVID 2020 | 14 Feb – 23 Mar 2020 | 1.59 → 0.76 | −0.83 |
  | Rates up (controlled) | — | — | +2.00 (Basel IRRBB parallel-up shock for USD); no credit, equity or FX move |

  In the three crises yields fell (a flight to safety), so fixed-rate assets gain on rates.
- **Formulas** (yields and spreads stored in % points, divided by 100; `k` = the episode
  multiple, as now):
  - Rate after: `r1 = max(r0 + k·Δr, 0)`, with r0 = 1.92% at 31 Dec 2019. Negative US yields
    are floored at 0 (stated).
  - **Bonds (implemented repricing):** fixed coupon `c = r0 + s0` (par at the book date, as
    now); value after `= N · P(c, r1 + s1, T)`, `P` = the existing annual-coupon `bond_price`.
    Rates up → value down.
  - **Pay-fixed interest rate swaps (new: mark-to-market added):** fixed rate `K = r0` (at par on
    the book date, so value before = 0); value after `= N · (r1 − K) · A(r1, T)`, `A` = the
    existing annuity. Rates up → the bank (paying fixed) gains. CVA stays as now: SA-CCR
    exposure from the notional only, so CVA does **not** respond to the swap's new value
    (stated limitation).
  - **CDS:** the risky annuity is discounted at `r1 + s1` instead of `r0 + s1` (small effect).
  - **Revolvers:** treated as floating-rate loans: no rate repricing; IFRS 9 ECL unchanged by
    rates.
  - **Capital:** rate gains and losses on bonds (fair value through OCI) and swaps (trading
    book) flow into CET1 after 25% tax, as credit losses do. Market-risk RWA uses the new bond
    value; its rate weights by maturity band are unchanged.
- **Switch:** `config.rates = "fixed" | "moved"`. `fixed` must reproduce E14 exactly.
- **Checks: pass criteria** (fixed now; a failure means the code is fixed, not the criterion):
  1. `rates = "fixed"` (or Δr = 0) reproduces E14 to the cent for every position, all three
     episodes at `k = 1`, and the stored E14 alert and crisis results.
  2. Direction: for a rate-only shock (no credit, z, FX), every bond loses value when rates rise
     and gains when they fall; every pay-fixed swap does the opposite.
  3. Size: for ±100 bp, each bond's full repricing is within 5% of its duration + convexity
     estimate (`duration_convexity`).
  4. Swaps are worth 0 at the book date (Δr = 0).
  5. Monotonic: rate-only shocks of +0, +100, +200, +300 bp give a total rate loss that does not
     fall.
  6. Floor: no scenario rate goes below 0.
  If all six pass, the engine default becomes `rates = "moved"`. Nothing else decides it.
- **Reported, never used to decide** (E14 and E15 files are kept unchanged; E22 writes new files):
  - Per scenario, `rates` fixed vs moved: loss split into credit and rate parts, CET1 after,
    reverse-stress multiple; plus the +200 bp scenario alone.
  - Test 2 (12 crises) and Test 3 (monthly), re-run with the Treasury move in both the
    prediction and the realised value: mean absolute loss error vs the naive, next to E14/E15.
    Pass rules of those tests are not re-applied to choose anything.
- **Limitations (stated with every result):** one 10-year rate for every tenor (a parallel
  shift; real curves twist); the bank's USD funding is not modelled, so a rate gain or loss on
  assets has no offsetting liability move (no full IRRBB / ALM); revolvers assumed floating;
  swap CVA ignores mark-to-market; no basis or curve risk; market-risk capital for swaps and CDS
  still not modelled; book still static at 31 Dec 2019.
- **Results (2026-10-11, run once, `data/outputs/e22_rates.json`; E14/E15 files unchanged, hashes
  checked).** All six checks pass (`tests/test_module_b_rates.py`, 13 tests, including the
  exact reproduction of the stored E14 alert and crisis results), so the engine default is now
  `rates = "moved"` (`config/default.toml`).

  | Scenario (× 1, no alert) | Δ UST10Y | Loss, rates fixed → moved | Rate part (bonds / swaps) | CET1 after, fixed → moved |
  |---|---|---|---|---|
  | GFC 2008 | −131 bp | $271.5m → $107.2m | −$164.3m (−$192.8m / +$28.6m) | 4.37% → 8.91% |
  | Euro 2011 | −141 bp | $47.8m → −$159.6m (a gain) | −$207.4m (−$238.2m / +$30.9m) | 10.58% → 15.57% |
  | COVID 2020 | −83 bp | $294.0m → $193.8m | −$100.2m (−$118.2m / +$17.9m) | 4.22% → 7.05% |
  | Rates up, controlled | +200 bp | $258.5m (bonds +$298.2m, swaps −$39.7m) | — | 13.00% → 6.18% |

  Reverse stress (multiple of the episode that takes CET1 to 8%), fixed → moved: GFC 0.58 → 1.23,
  euro 2.18 → 5.57, COVID 0.57 → 0.84.
  - *Reading:* in all three crises Treasury yields fell, so the fixed-rate bonds gained and
    offset a third to all of the credit loss; COVID still breaches the 8% floor. A pure
    +200 bp rise, with no credit move, also takes CET1 below the floor (6.18%): the bond book
    is the bank's main interest-rate risk.
  - *Caution:* the bank's USD funding is not modelled, so these rate gains are not offset by
    liabilities, as they would partly be in a real bank. The crisis results are therefore
    less severe than a full balance-sheet view; the rates-up result is the clearer use of E22.
  - *Reported only:* Test 2 (12 crises) loss error with rates moved: similar crises $31.0m vs
    naive $28.2m (E14: $45.7m vs $47.0m). Test 3 (345 months): model $42.9m vs naive $41.4m,
    ratio 1.04, direction 43% vs 52%, Diebold–Mariano p = 0.015 (E15: 1.00, p = 0.93). Errors
    fall in Test 2 for both methods; neither forecast beats the naive. Nothing was chosen on
    these numbers.

## Result

- **Question:** Interest-rate shock in Module B (parallel UST10Y shift)
- **Outcome:** 6 checks pass (incl. exact E14 reproduction) → default `rates = moved`. COVID loss $294m → $194m, CET1 4.22% → 7.05% (yields fell 83 bp); GFC 4.37% → 8.91%; +200 bp alone: CET1 13% → 6.18%. Forecast re-runs (reported only): still no edge over naive
- **Verdict:** checks pass; funding side not modelled
- **Stored results:** `e22_rates.json`, `e22_crisis_backtest.csv`, `e22_monthly_backtest.csv`, `tests/test_module_b_rates.py`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
