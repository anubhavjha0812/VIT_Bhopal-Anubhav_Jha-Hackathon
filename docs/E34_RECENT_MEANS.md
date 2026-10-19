# E34: rolling and exponentially weighted historical means

Rules fixed on 18 October 2026 before implementation or scoring. This follows
the user's suggestion after E33. No new data, features, labels or engine changes.

## Question and fixed candidates

The existing E32 mean weights every earlier completed monthly market-move vector
equally. Would giving recent history more weight improve its loss forecasts?
This is a new hypothesis about changing market conditions, not evidence that
recency weighting will win. Exponential smoothing assigns smaller weights to
older observations; see [Forecasting: Principles and Practice](https://otexts.com/fpp3/exponential-smoothing.html).

Try exactly two new candidates, with parameters fixed here, no search:

- **Rolling mean:** equal average of the last 60 eligible monthly target vectors
  (about five years). If fewer than 60 are eligible, use all eligible rows;
  this fixed warm-up keeps the first E32 origins instead of dropping cases.
- **EWMA (primary challenger):** all eligible monthly target vectors, with weights
  `2 ** (-age_months / 24)`, normalised to sum to one. Age is the difference in
  calendar months between the forecast origin and each target's origin. Half-life
  is fixed at 24 months: a target 24 months older has half the relative weight.
  This normalised finite-history weighted mean has no fitted initial level.

Apply the same weights separately to each of the six E32 market targets (Baa,
Aaa, VIX, S&P log return, USD/INR simple return, UST10Y change), then reprice that
single vector. Do not average market prices, already realised evaluation errors,
portfolio losses or CET1 ratios. EWMA is the preselected primary method; rolling
is secondary. No alternative windows, decays, trend terms, blends or model grid.

## Data, timing and pricing

Reuse unchanged E32/E33 helpers. Same local market snapshot; same 345 month-end
origins January 1995–September 2023; primary horizon 60 trading days, secondary
20. Use identical origins for every candidate and both horizons. Training starts
in February 1990. A target is eligible only after its window has ended and its
fixed 10-session assumed-availability buffer is complete. Training origins must
be strictly earlier than the forecast origin. Sort eligible origins before
weighting; never let newer unfinished windows enter a mean.

Rebuild from a market prefix with all later rows removed; forecasts, weights and
timing audits must match exactly at every scored origin. Keep the stored fixed
2019 SEC book, fixed book-date market base, rates moved, no issuer notching,
and the same expanding earlier-data equity volatility for every forecast and
realised scenario at that origin. Existing Module B source and defaults stay
unchanged. Read the frozen E32 mean/no-change controls and their saved outcomes;
verify their hashes, do not run E32 again.

The ten-session availability buffer is an assumption: the snapshot has no
verified release calendar or historical vintages. The 2019 book and market base
are counterfactual valuation constants. Reproduce the old actual loss/CET1
outcomes to one US cent and 1e-10 CET1 percentage points.

## Scoring and decision fixed now

Show all four candidates: E32 mean, E32 no-change, rolling mean, EWMA. Report
loss MAE USD millions, ratio to E32 mean, CET1 MAE percentage points, six market
target MAEs and per-year results. Use paired candidate-minus-control absolute
loss errors against mean and no-change. Reuse E32's 95% percentile moving-block
bootstrap: six monthly origins, 1,000 draws, seed 0. Reuse DM with lag 3/1 for
60/20 days. The sampling indices must be the same for all candidates.

EWMA has a useful development gain only if its primary 60-day ratio to the
E32 mean is <=0.85, its paired CI against the mean is entirely below zero, and
its point MAE is no worse than no-change. Otherwise say **"no demonstrated gain
from EWMA"**. Report rolling separately without replacing the preselected
primary candidate after results. Neither outcome changes production or item 10
to a confirmed win. These data and previous results have already been viewed;
this is another documented development experiment, not an untouched test.

## Execution and preservation

Write rules first, then one new evaluator/tests, run once offline, preserve old
files. An existing E34 result JSON must refuse a rerun; a crash before a JSON is
saved may be fixed and retried without changing this contract, with a note.
New JSON, predictions and per-year files only. Save code/rule/input/output hashes,
weight and timing audits, package versions and the decision. Append the outcome
here; log E34 in `docs/EXPERIMENTS.md` and the next `STEPS.md` entry. No network, commit, push, tuning or wiring.
Run fixture tests, full tests and Ruff; verify old baseline and preservation hashes.

## Results

One offline run completed on 18 October 2026, with no failed attempts, alternative
settings or retuning. Pre-run contract SHA-256:
`f9282b5e3f5cd60136e7eace77ccaf437028a58cc9349cf36f9f6d2296e7cf64`.
Both horizons have the same 345 origins, 31 January 1995–29 September 2023.
Lower loss/CET1 MAE is better.

| Horizon | Candidate | Loss MAE, USD m | Ratio to expanding mean | CET1 MAE, pp |
|---|---|---:|---:|---:|
| 60 days, primary | Frozen expanding mean | 41.408430 | 1.000000 | 1.110268 |
| 60 days | Frozen no change | 41.078904 | 0.992042 | 1.105118 |
| 60 days | Rolling, last 60 targets | 41.775104 | 1.008855 | 1.121835 |
| 60 days | EWMA, 24-month half-life | 41.769934 | 1.008730 | 1.124437 |
| 20 days, secondary | Frozen expanding mean | 23.077201 | 1.000000 | 0.618047 |
| 20 days | Frozen no change | 23.067225 | 0.999568 | 0.616483 |
| 20 days | Rolling, last 60 targets | 23.170430 | 1.004040 | 0.620707 |
| 20 days | EWMA, 24-month half-life | 23.150362 | 1.003170 | 0.621217 |

Paired absolute-loss error differences, USD millions; a negative difference
favours the new method. Intervals use the fixed six-month blocks, 1,000 draws,
seed 0. DM p-values are two-sided, lag 3 for 60 days and lag 1 for 20 days.

| Horizon | Method | Difference vs mean | 95% interval vs mean | 95% interval vs no change | DM p vs mean |
|---|---|---:|---|---|---:|
| 60 | Rolling | +0.366674 | [-0.476111, +1.214392] | [-0.206372, +1.721316] | 0.3879 |
| 60 | EWMA | +0.361504 | [-0.556648, +1.250041] | [-0.235050, +1.715856] | 0.4489 |
| 20 | Rolling | +0.093229 | [-0.120317, +0.364222] | [-0.096196, +0.413238] | 0.4331 |
| 20 | EWMA | +0.073161 | [-0.132362, +0.318290] | [-0.117129, +0.359948] | 0.5636 |

**Decision: no demonstrated gain from EWMA.** All three prewritten conditions
fail: the primary ratio is 1.008730, its interval crosses zero, and its point
error exceeds no-change. Rolling also has higher point error than both controls
at both horizons. This does not establish that recency weighting can never help;
these two fixed candidates did not improve this development replay. Keep the
retained benchmark and production unchanged. Do not search new windows or
half-lives on these results and present the best one as a fresh test.

All 690 future-truncation checks match vectors, weight hashes and timing audits
exactly; all timing violations are zero. The first 60-day rolling forecast uses
the fixed 56-target warm-up. Maximum realised-outcome differences from E32 are
$0.000000030 and less than 1e-14 CET1 pp. Neither old controls nor prior outputs
were regenerated. Saved six-target MAEs show some FX improvement but higher
primary Baa/Aaa/VIX/equity/UST errors than the expanding mean; target-specific
improvements do not override the preselected portfolio-loss objective.

New files: `data/outputs/e34_recent_means.json`,
`e34_recent_means_predictions.csv` (2,760 rows), and
`e34_recent_means_per_year.csv` (232 rows). The JSON includes all six target MAEs,
paired results, audits, versions, the contract text/hash and source/input/output
hashes. No fresh-test, live-performance or category-winning claim follows.

Validation: 16 new synthetic fixture tests; full suite 825 passed, 1 expected
failure; Ruff passes. All 175 pre-existing protected files and the original
17-file Module B baseline are preserved. No existing Module B source, engine
defaults, rules, models, frozen files or old results changed. No commit or push.

Note: some non-methodological notes were removed from this public copy; the rules of the monthly experiment are unchanged.
