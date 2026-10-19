# E32: chronological market-move forecast replay

Rules fixed on 18 October 2026, before coding or scoring. No network, model
search, weight search or engine change. E14/E15/E22 results have already been seen;
this is development work, not a confirmatory test.

## Data and information dates

- Use the local `data/reference/market_history.csv` only, sorted by its unique
  observation dates. Forward fill from earlier rows only; never backfill.
- Origins are the last stored trading day of every month from January 1995
  through the last month with a completed 60-trading-day target. Use these same
  origins for both horizons and all six candidates. Do not select crisis origins.
  Training origins start in February 1990, as in E15.
- The file contains observation dates, **no release timestamps or historical
  vintages**. To avoid treating observation dates as release dates, impose a fixed
  **10 stored trading-session availability buffer on every series**, including
  training outcomes. This is a conservative timing assumption, not a verified
  release calendar. It cannot remove later revisions in this snapshot. Therefore
  report this as a chronological replay under assumed availability, not proof of
  a fully vintage-correct, real-time forecast.
- At an origin, compute the E15 state (Baa level, VIX level and Baa's trailing
  20-session change) using the latest observation admitted by that buffer.
  Historical training states use the same buffer at their own origins. A target
  enters training only when its end plus the buffer is on or before the origin.
  Training origins must also be strictly earlier than the forecast origin.
- Fit distances and OLS only on those completed, assumed-available training
  targets. For pricing's equity-to-z scale use expanding S&P daily log-return
  volatility through the buffered observation cutoff, annualised by sqrt(252).
  Never use the old full 2008–2019 volatility window for an earlier forecast.
  Rebuilding after deleting all rows later than an origin must leave every
  forecast, neighbour and scale identical. Store timing-violation counts and
  check that invariance at every scored origin before valuing any forecast.

## Targets, pricing and candidates

Primary horizon: 60 stored trading days; secondary: 20. Targets are changes from
the origin's close to the close H sessions later (days 1 through H): Baa and Aaa
spread changes, VIX change, S&P log return, USD/INR simple return and UST10Y change.
The outcome itself is scored without shifting its window by the release buffer.

Import E15's month-end, completed-training and least-squares helpers, and E14's
move-vector helper. Reuse the unchanged `revalue` function with `rates="moved"`
(E22). Read the stored 31 December 2019 SEC book; never rebuild it. Apply all
forecast and realised vectors to the same fixed book-date spread, Treasury and
FX levels. Those fixed 2019 valuation levels and the hypothetical book are
counterfactual pricing constants, not predictors available at a 1995 origin.
Use the same origin-specific, earlier-data equity scale for all its valuations.
No issuer downgrade is added. VIX is evaluated as a target but is not a direct
pricing input in the existing engine.

Six candidates, with identical origins and training targets within each horizon:

1. **Historical mean:** expanding mean of earlier completed, available vectors.
2. **No change:** zero for every component.
3. **Fixed shock (reference only):** equity -10% (`log(0.9)`), UST10Y +200 bp
   (`d_ust=2` percentage points); zero credit, VIX and FX moves. This is not fitted.
4. **Clean analog:** mean target of the three closest earlier eligible month-end
   states. Standardise the three E15 state variables using training mean/sample
   standard deviation only; a zero scale becomes 1. Break distance ties by the
   earlier origin. Require at least three training rows; a shortage is an error,
   not a reason to drop selected origins after scoring.
5. **E15 state model:** OLS on the same three state values and a constant, refitted
   on the earlier eligible training set using E15's `forecast` helper.
6. **75/25 combination:** `0.75 * historical_mean + 0.25 * state_model`, separately
   for each move component, then reprice that vector. Do not average losses or
   CET1 ratios. No other weights or combinations.

## Scoring and fixed decision

- Report each candidate's loss MAE in USD millions, ratio to the historical
  mean's loss MAE, CET1 mean absolute error in percentage points, and MAE of
  each target. Spread/rate/VIX errors retain their input units; S&P log-return
  and USD/INR simple-return errors are fractions.
- The paired loss difference is candidate absolute loss error minus historical
  mean absolute loss error, in USD millions. Use a two-sided 95% percentile
  interval from a moving-block bootstrap: six consecutive monthly origins per
  block, non-circular starts, 1,000 resamples, seed 0. Use the same resampled
  origin indices for every candidate. Report per-year metrics as well.
- Reuse E15's Diebold–Mariano helper on paired absolute loss errors, with
  Newey–West lag `ceil(H / 21)`: 3 months for 60 days, 1 for 20 days. A constant
  zero difference has statistic 0 and p-value 1; flag other degenerate variances.
- The combination is better only if its **60-day loss-MAE ratio <= 0.85** and
  its paired bootstrap interval is entirely below zero. Otherwise the decision
  is **"no gain over the historical mean"** and the mean remains the research
  benchmark. Neither outcome changes the engine, dashboard or stress defaults.
- Show the stored old E14 crisis-analog MAE beside the new clean monthly analog;
  also show the stored E22 rates-moved crisis analog for context. State the
  numerical gaps. These use different origin selection, information, sample
  size and (for E14) rates, so the gaps **do not isolate the causal effect of
  leakage**. Do not present them as a measured leakage correction.

## Run, records and limits

Write a new evaluator and fake-frame tests, then run once. `--run` must refuse
if `data/outputs/e32_forecast_replay.json` exists. If the run crashes before that
JSON exists, fix it and rerun; never rerun after it exists. Preserve old outputs.
Store all move forecasts, realised moves, information cutoffs, training counts,
neighbours and valuations in a new E32 predictions CSV; store per-year metrics
in a new E32 CSV. The JSON records the contract, input/code hashes, metrics,
bootstrap settings, timing checks and decision. Append results here only after
the run. Verify the stored Module B baseline hashes at the end.

Monthly targets overlap heavily, especially at 60 days. Roughly 30 years and a
handful of crises give low power. This is a hypothetical static 2019 book, not
observed historical bank P&L or capital. Historical series lack release vintages;
the fixed buffer does not prove real-time data availability. E22 still uses one
parallel Treasury shift, omits USD funding offsets and CDS/swap market-risk
capital, and does not let swap CVA follow mark-to-market. Prior results and this
market history have already been viewed. No claim of an untouched test is made.

## Results

Run once, offline, completed 18 October 2026 at 16:51 UTC. Stored in
`data/outputs/e32_forecast_replay.json`. There were no failed run attempts and
the evaluator was not rerun after the JSON was written. The pre-run contract
SHA-256 is `c9ac9e5ddf8871e8b3f36af6a6b4a6710efb8b99a4a9feec628bd25fe08c871e`.

Both horizons use the same 345 monthly origins, 31 January 1995 through
29 September 2023. The last targets end on 27 December 2023 (60 days) and
30 October 2023 (20 days). Earlier eligible training counts range from 56 to 400
for 60 days, and 58 to 402 for 20 days. All 690 full-data versus truncated-data
forecast checks passed exactly; all recorded origin, target-end, assumed
availability and state-date violation counts are zero. These checks validate
chronology under the fixed availability assumption, not historical vintages.

The interval is the paired candidate-minus-mean **absolute loss error** in USD
millions; a negative value favours the candidate. It is not an interval for the
candidate's MAE alone. Bootstrap: 1,000 paired six-month moving blocks, seed 0.
CET1 error is in percentage points. The DM tests use lag 3 for 60 days and lag 1
for 20 days; displayed p-values of 0.0000 are rounded, not exact zero.

### 60 trading days: primary

| Candidate | Loss MAE ($m) | Ratio to mean | CET1 MAE (pp) | Paired difference 95% CI ($m) | DM p |
|---|---:|---:|---:|---|---:|
| Historical mean | 41.408 | 1.000 | 1.110 | [0.000, 0.000] | 1.0000 |
| No change | 41.079 | 0.992 | 1.105 | [-1.336, +0.515] | 0.4570 |
| Fixed shock (reference) | 260.289 | 6.286 | 6.736 | [+207.396, +229.186] | 0.0000 |
| Clean analog | 47.426 | 1.145 | 1.264 | [+2.707, +8.472] | 0.0001 |
| E15 state model | 42.986 | 1.038 | 1.144 | [-0.133, +2.970] | 0.0536 |
| 75/25 combination | 41.650 | 1.006 | 1.115 | [-0.180, +0.543] | 0.2202 |

### 20 trading days: secondary

| Candidate | Loss MAE ($m) | Ratio to mean | CET1 MAE (pp) | Paired difference 95% CI ($m) | DM p |
|---|---:|---:|---:|---|---:|
| Historical mean | 23.077 | 1.000 | 0.618 | [0.000, 0.000] | 1.0000 |
| No change | 23.067 | 1.000 | 0.616 | [-0.271, +0.187] | 0.9275 |
| Fixed shock (reference) | 258.000 | 11.180 | 6.758 | [+230.789, +239.393] | 0.0000 |
| Clean analog | 24.559 | 1.064 | 0.689 | [+0.246, +2.811] | 0.0518 |
| E15 state model | 23.566 | 1.021 | 0.633 | [-0.128, +1.031] | 0.0691 |
| 75/25 combination | 23.174 | 1.004 | 0.620 | [-0.058, +0.238] | 0.1582 |

**Decision: no gain over the historical mean.** The combination's primary ratio
is 1.006, above the required 0.85, and its paired interval includes zero. Keep
the historical mean as the research benchmark. No-change is slightly lower on
point MAE, but its interval also includes zero; no new selection rule was applied.
Nothing was wired into the engine or dashboard, and no stress default changed.

### Old versus clean analog: diagnostic only

| Analog check | Sample | Rates | Loss MAE ($m) | Clean monthly minus old ($m) |
|---|---|---|---:|---:|
| Stored E14 | 12 future-selected crises | fixed | 45.679 | +1.747 |
| Stored E22 rerun | Same 12 crises | moved | 31.013 | +16.413 |
| E32 clean monthly | 345 ordinary month-ends | moved | 47.426 | — |

The old numbers were read from their stored CSVs, not rescored. The clean analog
is worse here. The gaps do not measure leakage alone: sample selection, state
definition, information timing, scale and (for E14) rates also differ. Removing
future neighbours improves the validity of the comparison; it does not establish
better forecasts.

All six target MAEs are in the JSON. Per-origin forecasts, realised vectors,
training cutoffs/counts, neighbours and valuations are in
`e32_forecast_replay_predictions.csv` (4,140 rows); per-year loss, CET1 and target
errors are in `e32_forecast_replay_per_year.csv` (348 rows). Both CSVs are hashed
in the JSON, along with the inputs and source files.

Validation: 22 new fake-data tests; full suite **793 passed, 1 expected failure**;
`uv run ruff check src tests` passed. Protected source/decision hashes and the
17 old reference/output hashes were checked. No network, commit or push.
