# E33: predict portfolio loss directly

Rules fixed on 18 October 2026, before implementation or new scoring.
Requested after E32. This is development work on already viewed history; it
cannot establish an untouched final win.

## Question and reason

E32 predicts six market moves by their means/OLS, then reprices the book. Its
primary score is absolute **portfolio loss** error. These are different targets:
the median portfolio loss is the appropriate point forecast for absolute loss
error, and nonlinear repricing does not preserve means or medians componentwise.
That motivates this experiment; it does not predict that it will succeed.
See [Gneiting, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902)
and the [quantile-regression documentation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html).

Try exactly two new candidates, with no search over models, inputs or weights:

1. **Direct historical median:** median of earlier completed, available portfolio
   loss targets. Separately take the median of earlier CET1 targets.
2. **Direct quantile model (primary challenger):** two separate linear median
   regressions (`quantile=0.5`), for loss and CET1. Inputs are only E15's existing
   Baa level, VIX level and trailing 20-session Baa change. No new features.

Both forecasts are specific to this book. They do not produce a coherent market
scenario or jointly consistent loss/CET1 pair, and cannot replace the scenario
engine. Improvement here would be evidence about portfolio forecasting only.

## Fixed data, outcome and timing rules

- Use local data only; no network, new labels or data selection. Reuse E32's
  data preparation, state definitions, month ends and completed-target filters.
- Use the exact 345 E32 month-end origins, January 1995–September 2023, at both
  horizons. Primary: 60 trading days; secondary: 20. No crisis selection for
  the monthly experiment. Training origins start in February 1990.
- Reuse the stored 31 December 2019 SEC book, fixed book-date market base and
  unchanged E22 pricing with `rates="moved"`, no issuer notching. Training target
  at historical origin j is the portfolio loss and CET1 after repricing its
  realised H-day market vector, using **j's own earlier-data equity volatility**,
  exactly as for an E32 realised outcome. Save those training targets.
- All training states are buffered by E32's fixed 10-session assumed availability
  rule. A training target enters only when its H-day window plus the same buffer
  has ended by the forecast origin. Training origins must be strictly earlier.
- The local snapshot has no release timestamps or historical vintages. The
  buffer is an assumption, not verified real-time availability. The 2019 book
  and valuation base are counterfactual constants, not historical predictors.
- Fit feature mean/standard deviation on eligible training rows only (population
  standard deviation; a constant feature gets scale 1). For each target subtract
  its training median and divide by its training IQR; zero IQR gets scale 1.
- Use `QuantileRegressor(quantile=0.5, alpha=0.05, solver="highs")`, intercept
  included, separately for loss and CET1. Alpha 0.05 is fixed before scoring as
  a modest L1 penalty on the scaled problem; it is not selected by validation.
  No grid, feature additions, clipping, reweighting, rolling-window search or
  second alpha if the result fails. Negative predicted losses (gains) are allowed.
- Remove every market row after an origin, rebuild its completed training history
  and refit: both predictions and all fitted scales must be exactly unchanged.
  Memoized repricing may reuse a value only under a key containing its complete
  market move vector and earlier-data volatility; never cache a target by date
  alone. This allows the real truncation audit without repeated identical pricing.

## Comparisons, uncertainty and decision

Read the already saved E32 historical-mean and no-change predictions as frozen
controls, verified against their recorded hashes. Never run E32 `--run` or
overwrite its files. Verify E33 realised outcomes reproduce E32's at the same
origins, to a maximum absolute difference of one US cent and 1e-10 CET1 pp.

Report all four candidates: frozen E32 mean, frozen no-change, direct median,
direct quantile. Metrics: loss MAE in USD millions, ratio to the E32 mean and
CET1 MAE in percentage points; per-year results. Direct heads do not forecast
the six market variables, so no invented per-market-target score is reported.

For every candidate report paired absolute loss-error differences against each
of the three controls (E32 mean, no-change, direct median). Reuse E32's six-month
moving-block percentile bootstrap, 1,000 draws, seed 0, and DM lag 3/1 for 60/20
days. Lower candidate-minus-control difference is better.

The primary quantile challenger has a useful development gain only if, at 60
days, (a) its MAE / E32 mean MAE <= 0.85 with paired CI entirely below zero,
(b) its paired CI against direct median is entirely below zero, and (c) its
point MAE is no worse than no-change. Otherwise say **"no demonstrated gain
from the direct quantile model"**. Median-versus-mean gains are reported
separately as a baseline finding, never claimed as a trained-model win. Neither
outcome changes production, the dashboard or stress defaults.

## Execution, preservation and records

Write this contract first, then source and fake-data tests. Run the new monthly
experiment once; `--run` refuses an existing E33 JSON. A crash before saving a
JSON may be fixed and retried with the same contract, with the failure logged.
No retuning after results. Save JSON, predictions, training outcomes and per-year
metrics to new E33 files only. Save rule/code/input/artifact hashes and package
versions. Append results here, add E33 to `docs/EXPERIMENTS.md` and the next
`STEPS.md` number.

Existing Module B source, frozen E24 files, `docs/DECISIONS.md`, models, E32,
old outputs remain unchanged. Verify
the old baseline manifest and new preservation snapshot. Run meaningful fixture
tests, the full test suite and Ruff. No network, commit or push.

## Results

Monthly replay completed once on 18 October 2026 at 17:49 UTC. No failed scoring
attempts and no parameter changes. The pre-run rules SHA-256 is
`6eaa6e402ef66912bb49bcf9c391c31152c66da9d1041725549678dd1153967f`.
Full record: `data/outputs/e33_direct_loss.json`.

Both horizons use all 345 frozen E32 origins. All 690 full versus truncated-market
rebuild/refit checks passed exactly, including fitted scalers and coefficients.
All training-origin, target-end, assumed-availability and state-date violation
counts are zero. Repriced realised outcomes match E32: maximum loss difference
$0.000000030 and CET1 difference below 1e-14 percentage points.

### 60 trading days: primary

| Candidate | Loss MAE ($m) | Ratio to E32 mean | CET1 MAE (pp) | Paired error gap vs mean, 95% CI ($m) |
|---|---:|---:|---:|---|
| Frozen E32 mean | 41.408 | 1.000 | 1.110 | [0.000, 0.000] |
| Frozen no change | 41.079 | 0.992 | 1.105 | [-1.336, +0.515] |
| Direct historical median | 41.773 | 1.009 | 1.116 | [-0.042, +0.967] |
| Direct quantile model | 42.171 | 1.018 | 1.130 | [+0.076, +1.356] |

The quantile model's paired mean error is $0.763m **higher** than the frozen
mean; DM lag 3 p=0.0275. Its gap versus direct median is +$0.398m, 95% CI
[-$0.160m, +$0.789m]. The median baseline itself does not improve on the mean.

### 20 trading days: secondary

| Candidate | Loss MAE ($m) | Ratio to E32 mean | CET1 MAE (pp) | Paired error gap vs mean, 95% CI ($m) |
|---|---:|---:|---:|---|
| Frozen E32 mean | 23.077 | 1.000 | 0.618 | [0.000, 0.000] |
| Frozen no change | 23.067 | 1.000 | 0.616 | [-0.271, +0.187] |
| Direct historical median | 23.218 | 1.006 | 0.619 | [-0.204, +0.536] |
| Direct quantile model | 23.303 | 1.010 | 0.620 | [-0.094, +0.538] |

Quantile-versus-mean gap +$0.226m; DM lag 1 p=0.1563. Quantile-versus-median gap
+$0.085m, 95% CI [-$0.071m, +$0.201m]. All paired monthly intervals use the fixed
six-month moving blocks, 1,000 draws and seed 0. The exact control is named in
each `paired` entry of the JSON.

**Decision: no demonstrated gain from the direct quantile model.** All four
prewritten conditions fail. The model has 1.84% higher primary loss MAE than
the mean. It was not adopted, and no alpha, feature or objective search followed.
Matching the fitting objective to the scoring objective was a reasonable
hypothesis; it did not make this particular forecast more accurate.

The historical-mean research benchmark and the production scenario engine stay
as before. These outcomes concern a fixed hypothetical book on viewed history,
not actual bank-loss validation. Direct heads supply no coherent market vector
or joint loss/CET1 scenario; this result adds no market-direction claim.

Stored artifacts: `e33_direct_loss_predictions.csv`,
`e33_direct_loss_training.csv`, `e33_direct_loss_per_year.csv`. The JSON holds
all fitted-model/timing audits, per-control intervals, DM tests, versions and
rule/code/input/artifact hashes.

Validation: 16 new fake-data tests; full suite 809 passed, 1 expected failure;
repository Ruff passed. E32 and all previous outputs remain read-only. No
network, production wiring, commit or push.

Note: some non-methodological notes were removed from this public copy; the rules of the monthly experiment are unchanged.
