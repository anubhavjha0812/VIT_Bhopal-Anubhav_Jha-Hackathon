# E35: one fixed Supertrend regime forecast

Rules fixed on 19 October 2026 before code, data download or scoring, following
a request for one last Supertrend attempt. This is a development
experiment on already viewed history. No change to production or old results.

## Indicator, data and single candidate

Use real daily S&P 500 index high, low and close, 1990-01-01 through 2023-12-31,
from a separate Yahoo Finance snapshot via the existing yfinance dependency.
Old price/market files stay unchanged. Cache new bars and download provenance
under ignored `data/cache/e35/`; no download occurs in the scoring command.
Require every existing market observation date to have an actual bar and its
close to agree within $0.01. Abort on missing bars, invalid prices or larger
disagreement; do not fill or invent high/low from closes. This snapshot is
retrieved now, without historical vintages, which remains a timing limitation.

Use one daily Supertrend, **ATR length 10, multiplier 3**, with no parameter
search. True range is max(high-low, abs(high-previous close),
abs(low-previous close)); on the first bar use high-low. Seed Wilder ATR with
the first ten true ranges' arithmetic mean, then update
`ATR = (9 * previous_ATR + TR) / 10`. There is no signal before that seed.
At the first valid ATR initialise the trend as down and the line at the upper
band. Use the continuing final upper/lower bands and close crossings described
by [TradingView's Supertrend definition](https://www.tradingview.com/support/solutions/43000634738-supertrend/).
Equality does not flip direction. Compute chronologically from the first bar;
never restart the indicator at each forecast date or fit its initial state.

The only new candidate is **supertrend_regime_mean**. At each forecast origin:

1. Take the same E32 observation cutoff, ten existing market sessions earlier.
   Use the Supertrend direction at that cutoff, never the origin's later bar.
2. Tag each earlier monthly training origin with its own buffered direction.
   Retain only E32 targets whose whole future window and assumed ten-session
   availability buffer have ended by the forecast origin.
3. Average the six market-move target vectors of earlier eligible origins with
   the same up/down direction. If there are fewer than three, fall back to all
   eligible targets' expanding mean and record that fallback.
4. Reprice that one market vector on the unchanged book. This is our fixed
   conversion from an indicator direction to a risk forecast; Supertrend itself
   does not supply credit, FX, rate or portfolio-loss forecasts. No manual sign
   conversion, new magnitudes, trading strategy, hedge rule or blend is tested.

## Timing, scoring and decision

Use all 345 unchanged E32 month-end origins, January 1995–September 2023;
60 trading days primary, 20 secondary. No case dropping. Keep the fixed 2019
SEC book/base, rates moved, no notches, and the same expanding earlier-data
equity volatility. The ten-session buffer is assumed availability, not verified
release dates. Read and hash-check frozen E32 expanding-mean/no-change controls;
do not rerun E32/E33/E34. Reproduce actual loss within $0.01 and CET1 within
1e-10 percentage points.

For every scored origin, remove all later market AND OHLC rows, rebuild the
indicator and training design, and require identical vectors, selected-origin
hashes and timing audits. Report zero timing violations, same-regime counts,
fallback counts and up/down counts. Future bars must not change earlier signals.

Report loss MAE USD millions, ratio to the expanding mean, CET1 MAE pp,
six target MAEs and per-year results for all three candidates. Paired candidate
minus mean/no-change absolute-loss errors use E32's six-consecutive-month block
bootstrap, 1,000 draws, seed 0; DM lag 3/1 at 60/20 days. A useful development
gain needs primary MAE ratio <=0.85, a paired interval against mean entirely
below zero, and point MAE no worse than no-change. Otherwise say **no demonstrated
gain from the fixed Supertrend regime forecast**. No tuning or automatic adoption.

## Execution and preservation

Write only a new evaluator/tests and new E35 results; log the outcome in this
file, `docs/experiments/EXPERIMENTS.md` and the next `STEPS.md` entry. Record input/code/rule/artifact hashes and package versions.
Scoring refuses existing result artifacts. Fixes before a saved result may be
retried under the unchanged rules, with any failure recorded. The separate
download can be retried for transport failure without scoring or changing rules.

Run synthetic tests, full tests and Ruff; verify the preservation snapshot and
original Module B baseline. Keep existing Module B source, models, defaults,
frozen files, labels, holdouts and `docs/DECISIONS.md` unchanged. No commit/push.
If genuine compatible OHLC cannot be retrieved, finish the fixture-tested code
and report the data blocker; do not replace it with a close-only proxy.

Note: some non-methodological notes were removed from this public copy; the rules of the monthly experiment are unchanged.
