# Impact v3: does a robustness fix keep the score working in a crisis year? (E28)

Written before any run. Papers used for ideas: event-study methodology (windows and scaling of abnormal returns) and Group DRO (optimise the worst group), see `private_notes/literature_report.md`.

## Why
Impact v2 (ridge on a volatility-scaled target) beats the volatility-only baseline in fresh 2019 (Spearman 0.294 vs 0.145) but fails in fresh 2020 (-0.041 vs 0.307). A likely cause: `vol20` and `vix_prev` leave the training range in 2020, the linear model extrapolates, and predictions are clipped to 0 (many ties). This is a hypothesis, not a finding.

## Rules, fixed now
- Choices use only rolling folds: train 2009..Y-1, validate Y, for Y in 2011 (stress fold: euro crisis, high volatility), 2015, 2016, 2017, 2018. Fresh 2019, fresh 2020 and the seen-stock years are NOT used for any choice. They are scored once, after the choice is frozen in `data/outputs/impact_v3_frozen.json`.
- Candidates (six): (a) v2 ridge unchanged; (b) ridge with every numeric feature clipped to the training 1st..99th percentile; (c) ridge without `vol20` and `vix_prev` as regressors (volatility still scales the target and the expected move); (d) b and c together; (e) group-reweighted ridge (Group DRO idea): training years are the groups, 5 rounds, each round the group weights are multiplied by exp(0.5 x group loss) and renormalised, strong L2 (alpha 10); (f) volatility-only baseline. `vol20` is already a prior-window estimate, so no separate lagged-volatility candidate.
- Choice rule: highest WORST-fold Spearman gain over volatility-only among the five folds. A candidate must also lose no more than 0.02 mean Spearman against (a). If none beats (a) on the worst-fold gain, (a) stays and we say so.
- After freezing: report the frozen choice, (a) and volatility-only on fresh 2019 and fresh 2020 (F and S stocks), same metrics and intervals as v2. If 2020 still fails, say so.
- 2019 and 2020 were already viewed in E8, so this is a development change, not a new blind test.

## Result (selection frozen first: `data/outputs/impact_v3_frozen.json`; test run once: `impact_v3_results.json`)
Rolling folds (Spearman vs |CAR|; folds 2011, 2015-2018): v2 ridge -0.249, 0.393, 0.475, 0.382, 0.398 (worst gain over volatility-only -0.66); chosen **d (clip + no volatility regressors)** 0.401, 0.365, 0.462, 0.377, 0.404 (worst gain -0.012; mean 0.402 vs 0.280). Volatility-only: 0.413, 0.295, 0.375, 0.266, 0.340.

| Test (n) | volatility only | v2 ridge | v3 (d) |
|---|---:|---:|---:|
| F fresh 2019 (26,964) | 0.143 | 0.295 | 0.280 |
| F fresh 2020 (15,098) | 0.331 | -0.092 | **0.353** |
| S seen 2019 (6,214) | 0.270 | 0.321 | 0.327 |
| S seen 2020 (6,454) | 0.340 | 0.110 | **0.381** |

The 2020 failure is gone: v3 beats volatility-only in all four sets. The v2 numbers here are the raw expected-move Spearman, so they differ slightly from the 1-10-scale table in `IMPACT_V2.md` (-0.041 in F 2020). The 2020 gain over volatility-only is small (+0.02 to +0.04). 2019/2020 were already viewed in E8, so this is a development change, not a new blind test. v3 is not yet wired into the live pipeline, where impact is still the formula.

## In the engine (E28c)
`config/default.toml` sets `impact = "learned"`. The final v3 model (clip + no volatility regressors, refit on 2009-2018, 9,756 rows) is saved in `models/impact-v3/impact_v3.joblib` (10 KB, tracked) by `uv run python -m riskengine.impact.learned --fit`. In the pipeline, news rows of stocks with price history take the learned 1-10 score (deciles of the 2009-2018 fit) as `impact`; the old formula stays in `impact_formula`, and `expected_move` / `move_score` are kept. Tweets and stocks without prices keep the formula. `/analyze` takes text only (no price history), so it keeps the formula too.

Stored signals (`data/outputs/signals.jsonl.gz`) were updated with `--apply-signals` (3,182 news rows); the formula-only file is kept as `signals_formula_impact.jsonl.gz`.

Effects checked on the replay window: the credit gate fires the same 4 alerts (same rows, impact 8, 8, 9, 8) before and after; only 1 credit row is a scored news row. The naive trigger (impact above 7) now flags 2,621 rows instead of 7, because deciles put 30% of news above 7; the gate is unaffected. Module A does not use impact.

Replay-window check (Apr-Jun 2020, 3,182 news rows, Spearman vs the real |CAR[0,+1]|, `impact_v3_live_check.json`): formula 0.131, volatility only 0.203, v3 expected move **0.243**. 2020 was already seen in E28, so this is a development check, not a blind test.

## Tweets (E28d), rules written before any run
The v3 model above is trained on news only; tweets keep the formula. Question: does a model trained on tweets rank the real moves better than volatility only and the formula?
- Data: the Akkerman tweets (9 Apr - 16 Jul 2020), linked to our stocks, at most 20 per stock per day (seed 0, same rule as the engine sample). Train 9 Apr - 15 May 2020. Gap 16 - 31 May (the 2-day target overlaps). Test 1 Jun - 16 Jul 2020, never used for any choice.
- Model: the same recipe as v3 (clipped ridge on log(1+z), volatility only scales the target and the expected move). Features: topic probabilities, E2 event, sentiment (own model), absolute sentiment, word count (`vix_prev` is dropped as in v3). No publisher, no 24 h count, no novelty. No tuning; one fit, one test.
- Rule: tweets switch to the learned score only if, on the test, Spearman vs the real |CAR[0,+1]| is higher than both volatility only and the formula, and the 95% paired-bootstrap interval of the gain over volatility only is above 0. Otherwise tweets keep the formula and the result is reported as it is.
- Limits: one 3-month crisis period, so the model learns that regime only; the test is out of sample in time, not in market regime.

### Tweets result (run once: `data/outputs/impact_v3_tweets.json`)
Train 14,552 rows, test 22,761 rows (1 Jun - 16 Jul 2020). Spearman vs the real |CAR[0,+1]|: v3 tweets **0.140**, volatility only 0.134, formula 0.011. Gain over volatility only +0.006, row-bootstrap 95% interval [0.004, 0.009] (rows of one stock-day are not independent, so the interval is too narrow). The rule passed, so tweets now take the learned score. The real gain is over the formula, which does not rank tweet moves at all; over plain volatility it is tiny. One crisis period only.

### Credit gate keeps the severity formula
With the learned score as `impact` for tweets, the gate fired a different set (lost DIS twice and BAC, gained XOM and JPM): the learned score says how far the stock may move, not how serious the credit event is, and a real downgrade on a calm stock got impact 2-6. So the gate (`gate.py`), the alert size (`replay.py`, scaled shock in `backtest.py`) use `impact_formula`. With that the gate fires the same 4 alerts as before (BA, DIS, DIS, BAC). The naive trigger (impact above 7) now flags 6,661 rows (29%) instead of 7.
