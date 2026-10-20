# E14: Module B book, shock and capital rules

Rules fixed 2026-10-10, before the code. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** Module B turns a gated credit alert (E13) into a loss and a capital ratio. The book,
  parameters and rules are fixed here so no number is tuned after seeing results.
- **Bank:** an Indian bank (capital in INR, RBI floor) with US corporate exposures in USD. The USD
  assets are funded in USD, so a rupee move changes RWA in INR, not P&L.
- **Obligors:** our 15 universe companies, S&P issuer ratings at 31 Dec 2019, each with a source in
  `ratings.csv`. Book date 31 Dec 2019, before every 2020 shock.
- **Size:** the bank holds 0.5% of each company's total debt at its last filing on or before
  31 Dec 2019 (SEC EDGAR XBRL, tag and accession saved). JPM and BAC are capped at the largest
  non-bank exposure. 0.5% is an illustrative mid-size bank. **Sensitivity:** the run is repeated at
  0.1%, 0.5%, 1% and 2%; if the CET1 change differs by more than 0.1 point across them, the share
  is instead derived from real data (a real bank's corporate book ÷ US corporate debt).
- **Split:** revolver = the company's credit line (XBRL tag, else a 10-K sentence found by regex
  and read by hand), else its short-term debt; bond = long-term debt, with cash flows from the
  10-K repayment schedule; term loan only where the 10-K reports one. Revolver EAD uses a 75%
  credit conversion factor (Basel F-IRB).
- **Hedges:** CDS protection bought on the 3 lowest-rated obligors (25% of exposure) and sold on
  the 3 highest-rated (10%); pay-fixed IRS on the 5 largest (20%), valued for CVA. Bought
  protection must gain when spreads widen (tested).
- **PD:** S&P long-run one-year default rates by letter grade, log-interpolated by notch, floor
  0.03% (Basel). D = 100%.
- **Spread:** FRED Moody's Aaa and Baa over the 10-year Treasury, log-linear by notch, extrapolated
  below BBB (this understates junk spreads; stated).
- **Shock on an alert:** notches parsed from the text ("Baa1 to Baa2"); none found = 1 notch;
  Default = D. The obligor moves to the new rating's PD and spread. Other obligors widen by the
  obligor's widening × their daily return correlation with it (2008–2019, negatives set to 0).
  Plus a market episode (GFC 2008, euro 2011, COVID 2020 by default): FRED Aaa/Baa spread moves,
  the 15-stock move and the USD/INR move (FRED `DEXINUS`) over the episode dates; `flat` or
  `scaled` by impact/10 (`config.shock`).
- **Stressed PD:** Vasicek `N((N⁻¹(PD) + √ρ·z)/√(1−ρ))`, ρ from the Basel corporate formula, z the
  episode's equal-weight 15-stock move in standard deviations (2008–2019 daily vol).
- **IFRS 9:** Stage 2 if cut ≥ 3 notches since the book date or below BBB-; Stage 3 at D. ECL:
  Stage 1 12-month PD, Stage 2 lifetime PD `1−(1−PD)^M`; LGD 45%.
- **Capital:** Basel IRB corporate RWA (F-IRB, LGD 45%, M from the position capped 1–5 years) on the
  stressed PD, converted to INR at the scenario USD/INR. CET1 starts at 13% of RWA; losses after
  25% tax; floor 8% (RBI: CET1 5.5% + capital conservation buffer 2.5%).
- **Reverse stress:** the multiple of the episode at which CET1 reaches 8% (bisection).
- **Monte Carlo:** 20,000 draws, Gaussian copula with the 15-stock return correlation (calm
  2008–2019, stressed 14 Feb – 23 Mar 2020), seed 42; VaR and ES at 99%, before and after.
- **Memo:** a fixed template, no LLM. Every number comes from the run; the trigger headline, the
  matched words and the gate reason (E13) are quoted.
- **Tests:** at least 30 for Module B: IRB against a hand calculation (PD 1%, LGD 45%, M 2.5),
  convexity against full repricing for 200 bp, hedge sign, staging, notch parsing, PD table
  rising as ratings fall, Vasicek at z = 0, Monte Carlo mean at zero correlation, reverse stress
  landing on 8%, FX raising RWA when the rupee falls.
- **Check 1, real alerts:** for the 3 real alerts (Moody's–Boeing, S&P–Disney, Fitch–Disney), the
  predicted change in the bond and CDS book is compared with the book repriced at the realised
  FRED moves over the next 20 trading days: sign match and error vs a flat-episode naive. FRED
  gives rating buckets, not company spreads, so this checks direction and scale, not
  company-level accuracy. Scored once.
- **Check 2, crisis backtest:** episodes are chosen by a rule, not by hand: every non-overlapping
  window in 1990–2023 where FRED `BAA10Y` widens by 75 bp or more within 60 trading days. For each
  episode, its first 5 trading days (changes in Baa, Aaa, VIX, S&P 500, USD/INR) are matched,
  leaving that episode out, to the 3 nearest other episodes (standardised Euclidean distance);
  their average 60-day moves are the prediction. The book (P&L and CET1) is valued at the
  predicted and at the realised 60-day moves. Reported: sign match and mean absolute error in
  P&L and CET1, against a naive prediction (the average of all other episodes). Scored once.
- **Later, if time:** equity puts (Black-Scholes, VIX as volatility) as a non-linear hedge.
- **Clarified while coding (2026-10-10, before any loss or capital number was computed):** z is
  measured in one-year standard deviations (2008–2019 daily vol × √252), because Vasicek's market
  factor belongs to the one-year PD. The first draft used the episode's own length (COVID z = 5.8)
  and gave an AAA name a 25% stressed PD; now COVID z = 1.83, GFC 1.72, euro 2011 0.94. The
  stressed PD is never below the rating's normal one-year PD.
- **Amended after the first run (2026-10-10, after seeing losses and CET1; before the checks):**
  the first run gave COVID CET1 13% → −4%, for two reasons, both fixed by Basel's own rules, not
  by tuning. (1) RWA rose 2.4× because the IRB formula was fed the Vasicek stressed PD; Basel IRB
  uses the long-run PD of the rating, so a downgrade raises RWA and a bad market alone does not.
  The stressed PD now drives IFRS 9 ECL only. (2) The book is 85% bonds ($2.6bn against $139m of
  capital), because the bank holds 0.5% of each company's long-term debt; a bank with a bond book
  also holds market-risk capital for it. Bonds now add Basel standardised market-risk RWA
  (specific risk: investment grade 0.25% / 1.0% / 1.6% by maturity, BB 8%, below BB- 12%;
  general interest-rate risk by maturity band, 0.7%–6%). The user chose this over holding the
  bonds at amortised cost or keeping the first result. Market risk of CDS and swaps is not
  modelled. Rejected runs and their numbers are kept in STEPS.md.

## Result

- **Question:** Module B checks: real alerts, 12 crises
- **Outcome:** sign right 2 of 3 alerts, size 18–75× too big; analogs no better than naive ($45.7m vs $47.0m)
- **Verdict:** negative / neutral
- **Stored results:** `module_b_alerts_check.csv`, `module_b_crisis_backtest.csv`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
