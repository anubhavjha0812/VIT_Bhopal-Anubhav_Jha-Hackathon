# E25 re-test: candidate E vs candidate C on a fresh set (rule written before labelling)

Written 2026-10-17, before any row of the new set was labelled or scored.
`docs/DECISIONS.md` is not edited.

## Why

- On D1 (153 rows) E scored 0.476 and C 0.516. But C's rules were built while looking at D1, so D1 favours C.
- D1 is also small: 4 Product news and 2 M&A rows.
- So we test again on a fresh, larger set. Same rule for E and C.

## The set (R3)

- 400 headlines from the dev window (2019-01-01 to 2020-06-30), built by `src/riskengine/eval/e25_retest.py` (seed 26).
- 200 random rows (natural mix) + 25 rows for each of 8 keyword groups (M&A, Macro, Geo, Management, Earnings, Analyst, Legal, Product), so rare classes appear.
- No text matches the D1/D2 dev rows, the final holdouts, E23, or any training set (same leak filter as the E25 pool).
- Keywords only choose which headlines are drawn. They are not labels.
- Labels: Claude labels blind (text only, `gold/e24/GUIDE.md`), no model output seen. DevxAnu can have ChatGPT check them. They are AI-derived and are never called human or expert reviewed. Report each reference separately if there is more than one.

## Rule (fixed now)

1. Score A, C, D and E with the per-row method. Primary number: fixed 12-class macro-F1.
2. **E replaces C** only if E beats C by at least 0.02 macro-F1 on R3 **and** no class with 10 or more R3 rows loses more than 0.05 F1 against C.
3. Otherwise C stays. A tie or a small win is not enough (E is the more complex model).
4. No threshold, temperature or rule changes after seeing R3 scores. No retraining on R3.
5. R3 is a development set. D2 selection and the freeze still follow `docs/experiments/E25_TEACHER_STUDENT.md` section 7, and the final holdouts are scored once after the freeze.
6. Report D1 and R3 together, whatever the result.

## Not allowed

- No D2 or final holdout row is opened. R3 labels are not used for training.
- No second round of choosing a new set if R3 disagrees with the rule.
