# E43: does a daily turnover cap cut Module A's cost without hurting it?

Rules written on 19 Oct 2026, before any code and before any scored run. They are not changed after the run. One scored run; a run that wrote a result file is never repeated.

## Why

E42 (`docs/experiments/MODULE_A_DEADBAND.md`) found that a 2-point deadband cut turnover by only about 16%, because about 70% of Module A's turnover comes from per-name moves above 2 points. A rule that damps the big moves is the next control. A daily turnover cap limits the total weight moved in one day and moves toward the target by as much as the cap allows.

Disclosure: the diagnostic that suggested this control was computed on the 2010 to Jun 2020 history, so it saw the test years' move sizes (not their returns). E17 already found that slower or partial moves (every 5 days, or 20% of the way) did worse than the daily full move on the 2010-2016 development years. This experiment is run anyway because a cap only damps large days, which is different from partial moves every day. It may well fail again, and the result is kept either way.

## Policy (fixed)

Same targets, live-name mask, cost (5 bps on turnover), windows and baseline as E17/E42. Each day, with `w` the current weight and `goal` the target (normalised over live names):
1. If a held name is no longer live, rebalance fully to the target (as E17).
2. Otherwise let `gap = sum(|goal - w|)`. Move `w + (goal - w) * min(1, T / gap)`, where `T` is the daily turnover cap.
3. The result is a convex mix of two long-only, sum-1, capped weight vectors, so it stays long-only, sums to 1 and respects the 15% cap. No re-capping is applied.
4. No cap (the E17 chosen run) must be reproduced exactly; this is tested.

## Candidates, windows, choice

- Grid, fixed now: no cap (baseline), and `T` in {0.05, 0.10, 0.15} total weight moved per day. No other value is tried. (Baseline average daily turnover is about 0.26.)
- Windows as E42: development 2010-01-01 to 2016-12-31 for the choice; test 2017-01-01 to 2020-06-10 (with and without the 43-day replay); tweets 2021-09-30 to 2022-09-30.
- Choice on development only: the grid value with the highest Sharpe difference against equal weight net of cost. If no cap beats the baseline on development, the experiment ends ("no cap beats the daily move on development") and the test and tweet windows are not scored.
- The chosen `T` is scored once on test, test without replay and tweets.

## Decision rule (all must hold on the test window, chosen `T` against no cap)

1. A cap is chosen on development.
2. Average daily turnover falls by at least 30%.
3. Net Sharpe is higher than the no-cap run.
4. Max drawdown is not worse by more than 1 point.
5. On the tweet window, net Sharpe is not lower than the no-cap run.

The paired Sharpe-difference interval (moving-block bootstrap) is reported, not part of the rule. Whatever the result: no default, config, stored output, API number or README figure changes. A pass means only "a turnover cap is a measured cost saver".

## Honest limits

- The windows were already scored in E17 and E42 for the baseline policy; the cap is chosen on development and scored once, but the data are not new.
- The move-size diagnostic saw the test years' targets.
- A cost saving does not create alpha: equal weight stays the benchmark; Module A shows no proven edge.
- No sentiment or event accuracy claim.

## Result

One scored run on 19 October 2026. The rules above are unchanged. Development 2010-2016, Sharpe difference against equal weight, net of 5 bps:

| Daily turnover cap | none (E17 daily move) | 0.15 | 0.10 | 0.05 |
|---|---:|---:|---:|---:|
| Sharpe difference vs equal weight | **0.0948** | 0.0318 | 0.0350 | 0.0636 |

No cap beats the daily move on development, so per the fixed rule the experiment ends there: the test and tweet windows were not scored and nothing is adopted. Damping the big days costs more signal than it saves in trading cost, the same direction E17 found for slower or partial moves. Together E17, E42 and E43 say Module A's turnover is the signal itself; cheaper trading controls do not help.

Nothing changed: Module A, its config, stored outputs and the API are untouched. Files: `src/riskengine/module_a_turncap.py`, `tests/test_module_a_turncap.py`, `data/outputs/module_a_turncap.json` and its run lock.
