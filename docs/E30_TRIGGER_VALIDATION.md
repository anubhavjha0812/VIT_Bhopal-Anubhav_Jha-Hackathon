# E30: do non-credit alerts predict real market reactions?

Written before any run. Question: if Module B were also triggered by non-credit news, would the trigger catch real price shocks better than chance, than volatility alone, and than the usual "formula impact >= 7" trigger?

## Fixed now
- Rows: the impact v3 test sets of E28 (headline x stock rows). Primary: **F fresh 2019** and **F fresh 2020** (stocks never used to fit, v3 fitted on 2009-2018). Secondary: S seen 2019 and 2020 (our 15 stocks, years already viewed in E8).
- A **real shock** = the stock's 2-day abnormal move |CAR[0,+1]| is at least 5% (market-model, as in E8). Base rate is reported. Second label, reported only: the move is at least 3 normal two-day moves (z >= 3).
- Triggers (all fixed here):
  1. v3 >= P95 and v3 >= P99 of the expected move: thresholds are the 95th and 99th percentile of v3's expected move on the 2009-2018 fit rows (not the test).
  2. v3 >= P99 and at least one other headline on the same stock in the previous 24 h (corroboration).
  3. Volatility only: the same number of rows fired as trigger 1 (top 5% and top 1% of `vol20` in the test set).
  4. Random: the same fire rate as trigger 1; its expected precision is the base rate.
  5. Common formula trigger: formula impact >= 7 and >= 8.
- Metrics: precision (fired rows followed by a real shock), recall (real shocks that were fired), lift = precision / base rate. Intervals: bootstrap over stock-days (rows of the same stock-day are not independent), 1,000 resamples, seed 0.
- Decision: v3 may be used as an additional trigger only if, on both F years, precision at P99 beats volatility-only and the interval of the difference excludes 0. Otherwise Module B keeps the credit gate and the market-wide trigger, and the result is reported as it is. One run, no tuning of thresholds or labels after seeing it.
- Limits: a big move after a headline is not proof that the headline was a crisis; this validates "the market reacted", not "it was a crisis". 2019-2020 were already seen in E28 (development check, not blind). Headline rows, not stock-days, are the unit; recall is at row level.

## Result

18 Oct 2026. Ran once using local data; no thresholds or labels were tuned. Frozen model: `d_clipped_no_vol`; fit rows (2009–2018): 9,756. Fit-only expected-move thresholds: P95 = 0.05402444 (5.402%), P99 = 0.09127842 (9.128%).

Primary label: |CAR[0,+1]| >= 5%. Precision and recall are fractions; lift is a multiple.

| Trigger | F 2019 precision | Recall | Lift | F 2020 precision | Recall | Lift |
|---|---:|---:|---:|---:|---:|---:|
| v3 P95 | 0.232 | 0.054 | 3.23 | 0.338 | 0.454 | 1.97 |
| Volatility, P95 matched count | 0.095 | 0.022 | 1.32 | 0.336 | 0.452 | 1.96 |
| Random P95, seed 0 draw | 0.046 | 0.011 | 0.65 | 0.166 | 0.224 | 0.97 |
| v3 P99 | 0.076 | 0.005 | 1.05 | 0.411 | 0.162 | 2.39 |
| Volatility, P99 matched count | 0.067 | 0.004 | 0.94 | 0.428 | 0.169 | 2.49 |
| Random P99, seed 0 draw | 0.118 | 0.007 | 1.64 | 0.180 | 0.071 | 1.05 |
| v3 P99 + corroboration | 0.076 | 0.005 | 1.06 | 0.407 | 0.142 | 2.37 |
| Formula >= 7 | 0.103 | 0.028 | 1.44 | 0.229 | 0.024 | 1.33 |
| Formula >= 8 | 0.123 | 0.020 | 1.72 | 0.273 | 0.019 | 1.59 |

F 2019: 26,964 headline-stock rows, 9,331 stock-days; shock base rate 0.0717 (7.17%).

F 2020: 15,098 headline-stock rows, 5,177 stock-days; shock base rate 0.1717 (17.17%).

Random expected precision equals each year's base rate; the table shows the single seed-0 draw. Volatility and random triggers each fire exactly as many rows as v3 at the corresponding level; fit-based P95/P99 need not fire 5%/1% of a test set.

Paired precision difference (v3 P99 minus volatility at the same fire count), with 1,000 stock-day bootstrap resamples, seed 0:

| Primary set | Difference | 95% interval | Passes fixed rule |
|---|---:|---|---|
| F 2019 | 0.0084 | [-0.1338, 0.2058] | no |
| F 2020 | -0.0176 | [-0.0884, 0.0479] | no |

Do not adopt v3 as an extra non-credit trigger; keep the credit and market-wide gates.

The [stored JSON](../data/outputs/e30_trigger_validation.json) includes fired counts, both labels, precision/recall intervals for every trigger, and the secondary S sets. Undefined ratios are stored as null; intervals omit undefined resamples and record the number of valid draws.

Caveats: a big price move validates "the market reacted", not "it was a crisis". 2019–2020 were already seen in E28, so this is a development check, not a blind test. The unit is a headline-stock row; recall is at row level, with uncertainty clustered by stock-day. F 2020 uses the existing fresh-set window ending 10 June, not the full year. The z >= 3 label and S sets are reported only. No trigger was wired into the engine.
