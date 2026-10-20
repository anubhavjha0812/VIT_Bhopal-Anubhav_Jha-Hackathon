# E42: does a trade deadband cut Module A's cost without hurting it?

Rules written on 19 Oct 2026, before any code and before any scored run. They are not changed after the run. One scored run; a run that wrote a result file is never repeated.

## Why

Module A tilts weights by news sentiment every day. On the stored 2010 to Jun 2020 history its average daily turnover is about 0.25, which at 5 bps costs about 1.25 bps a day, about 3 points a year. E17 already tried slower trading (every 5 days, or moving 20% of the way) and the full daily move was best on the 2010-2016 development years, so those two controls are not repeated. A deadband is a different control: leave a name alone unless its target weight is far enough from its current weight. This experiment asks only whether it saves cost. It does not claim an edge: E17 found none (test Sharpe difference -0.071, interval crosses zero) and the placebo p is about 0.4.

## Policy (fixed)

Each day, with the same targets, live-name mask and cost (5 bps on turnover) as E17:
1. If a held name is no longer live (no next-day return), rebalance fully to the target, as E17 does.
2. Otherwise, for each name, trade to its target only if the absolute gap between target and current weight is larger than `d`; hold it otherwise.
3. Renormalise the live names to sum 1 and re-apply the existing 15% cap (`cap_weights`). Long-only.
4. `d = 0` must reproduce the E17 chosen run (daily, full move) exactly; this is tested.

## Candidates, windows, choice

- Grid, fixed now: `d` in {0 (baseline), 0.005, 0.01, 0.02} absolute weight. No other value is tried.
- Same data and windows as E17: primary variant (rules applied) from `load_history()`, development 2010-01-01 to 2016-12-31, test 2017-01-01 to 2020-06-10 (the 43-day replay sits inside it, so a without-replay row is also reported), and the fresh tweet window 2021-09-30 to 2022-09-30.
- Choice on development only: the `d` with the highest Sharpe difference against equal weight, net of cost, as in E17. If `d = 0` wins on development, the result is "no deadband beats the daily move on development", the test and tweet windows are not scored, and the experiment ends.
- The chosen `d` is scored once on the test window, test without replay, and the tweet window.

## Decision rule (all must hold on the test window, chosen `d` against `d = 0`)

1. The chosen `d` is above 0.
2. Average daily turnover falls by at least 30%.
3. Net Sharpe is higher than the `d = 0` run.
4. Max drawdown is not worse than the `d = 0` run by more than 1 point.
5. On the tweet window, net Sharpe is not lower than the `d = 0` run.

The paired Sharpe-difference interval (moving-block bootstrap as in E17) is reported but is not part of the rule.

Whatever the result: no default, config, stored output, API number or README figure changes in this work. A pass means "a deadband is a measured cost saver" and is listed as an optional follow-up; it does not turn Module A into an edge.

## Honest limits

- The history and tweet windows were already scored in E17 for the baseline policy; the deadband itself is chosen on development and scored once, but the windows are not new.
- A cost saving does not create alpha: equal weight stays the benchmark and Module A still shows no proven edge.
- AI-made sentiment signals; no sentiment or event accuracy claim.
- The 43-day demo window is inside the test window and is not a separate test.

## Result

One scored run on 19 October 2026. The rules above are unchanged. The `d = 0` run reproduces the stored E17 test numbers exactly (Sharpe 1.0855, turnover 0.262; checked before the run; no deadband value was scored in that check).

**Development 2010-2016 (choice):** Sharpe difference against equal weight, net of cost: d = 0: 0.0948, 0.005: 0.0982, 0.01: 0.0949, **0.02: 0.0985 (chosen)**. The four values are within 0.004 of each other; daily turnover on development falls from 0.255 to 0.215 at d = 0.02.

| Window (net of 5 bps) | d = 0 Sharpe | d = 0.02 Sharpe | Equal weight Sharpe | Turnover d = 0 -> 0.02 | Max drawdown d = 0 -> 0.02 |
|---|---:|---:|---:|---:|---:|
| Test 2017 - Jun 2020 | 1.0855 | 1.1131 | 1.157 | 0.262 -> 0.219 | -34.3% -> -34.6% |
| Test without the 43-day replay | 0.8923 | 0.9164 | 0.9437 | 0.261 -> 0.219 | -34.3% -> -34.6% |
| Tweets 2021-22 | -0.7749 | -0.7759 | -0.7697 | 0.252 -> 0.207 | -36.9% -> -37.2% |

Sharpe difference d = 0.02 minus d = 0 on test: +0.028 [-0.012, 0.076]; without replay +0.024 [-0.011, 0.076]; tweets -0.001 [-0.063, 0.043]. All intervals cross zero.

**Decision rule:** chosen d above 0: yes. Test Sharpe higher: yes (+0.028). Drawdown not worse by more than 1 point: yes. **Turnover down at least 30%: no (down 16.4%). Tweet-window Sharpe not lower: no (-0.7759 vs -0.7749).** Decision: **not supported**. Nothing was changed: Module A, its config, stored outputs and the API are untouched.

What this shows: a deadband up to 2 points of weight cuts turnover by about 16% (development and test alike) and the Sharpe gain is inside the noise. A fixed turnover cap or a wider band might cut more, but those values were not in the grid and are not tried after the result. Equal weight stays the benchmark; Module A still shows no proven edge.

**Why the cut was small (diagnostic run after the result; not part of the rule, no new scoring).** On the 2010 to Jun 2020 history, with 15 names and a median held weight of 6.7%, the daily per-name target moves split as: under 1 point 28% of turnover, 1 to 2 points 19%, 2 to 5 points 39%, above 5 points 30%. About 70% of turnover comes from moves above 2 points, which a 2-point deadband never stops. The deadband can only trim the small moves, which is why turnover fell about 16% instead of 30%.

Files: `src/riskengine/module_a_deadband.py`, `tests/test_module_a_deadband.py`, `data/outputs/module_a_deadband.json` and its run lock.
