# E20: Choosing the engine sentiment model on news as well as tweets

Rules fixed 2026-10-11, before any dev or new test row is scored. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** E7 chose the engine model on a tweet-only dev set, and the E5 model stayed. E18 then
  showed it transfers worse to news than the E7 distilled model. The E18 sets have been seen,
  so they cannot choose the model; this rule chooses on dev sets and confirms on a fresh test.
- **Candidates** (no new training; both bge-small, so engine size and speed stay):
  (A) the current engine model, E5, int8; (B) the E7 distilled student, exported to int8 with the
  E9 recipe (opset 17, dynamic quantisation, no calibration). Decisions use the int8 versions,
  because the engine runs int8.
- **Dev sets** (never used for training or testing by either candidate):
  - Tweets: the E7 dev split of `sent_train.csv` (10%, stratified, seed 42, minus the E7 drops).
  - News: 2,000 rows drawn at random (seed 0) from the NWGI *train* split, labels folded as E18.
    Rows that match either candidate's training texts (E18 leak check) are dropped first. Its
    labels were written by a GPT model; it only steers the choice and is never reported as a
    result.
- **Rule:** score = mean of the two dev macro-F1s. B replaces A only if its score is higher by
  more than 0.01 **and** its tweet-dev macro-F1 is no more than 0.01 below A's. Otherwise A stays.
- **Confirmation (fresh test, scored once, after the choice):** SEntFiN 1.1 (Kaggle
  `ankurzing/aspect-based-sentiment-analysis-for-financial-news`, `SEntFiN-v1.1.csv`), every
  headline with exactly one entity, labelled with that entity's sentiment (positive, neutral,
  negative). Human-labelled news headlines, never scored by us. Leak check as E18 first; seen
  rows are dropped. Metrics as E18, plus the paired bootstrap of B − A.
- **Switch:** the engine changes to B only if the rule picks B **and** the SEntFiN paired
  difference is not "lower" (its interval is not wholly below 0). If the rule picks B but SEntFiN
  says lower, A stays and both results are reported.
- **Also reported, marked as already seen:** E1 (TFNS test) and the four E18 sets for B int8.
- **Compute:** int8 export and scoring are small CPU jobs (about 6,000 texts per model).
- **Results (2026-10-11, `data/outputs/e20_results.json`).** Dev macro-F1, int8:

  | Dev set | A (engine) | B (distilled) |
  |---|---|---|
  | Tweets, TFNS dev (937) | 0.864 | 0.829 |
  | News, NWGI train sample (2,000) | 0.404 | 0.523 |
  | Mean | 0.634 | 0.676 |

  B's mean is higher by 0.042, but it loses 0.035 on tweets (limit 0.01), so **A stays**.
  Confirmation, SEntFiN single-entity headlines (7,903, 0 seen in training): A 0.745
  [0.736, 0.755], B 0.765 [0.756, 0.774] macro-F1; paired B − A +0.020 [0.013, 0.027], higher.
  The engine stays A, as the rule fixed before the test. Neither model is better on both tweets
  and news; a model that is would be a new experiment.

## Result

- **Question:** Choose the engine sentiment model on tweet + news dev
- **Outcome:** dev: engine 0.864 / 0.404, distilled 0.829 / 0.523 → engine stays; SEntFiN 7,903: 0.745 vs 0.765, paired +0.020 [0.013, 0.027]
- **Verdict:** engine stays (trade-off)
- **Stored results:** `e20_results.json`, `e20_predictions.csv.gz`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
