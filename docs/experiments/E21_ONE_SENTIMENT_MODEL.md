# E21: One sentiment model for tweets and news

Rules fixed 2026-10-11, before any training, labelling or scoring. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** E20 showed a trade-off: the engine model (A) is better on tweets, the E7 distilled
  model on news. E21 trains one bge-small model on both and confirms it on fresh tests.
- **Fresh tests** (scored once, after the choice):
  - **News, primary:** FinEntity (`yixuantt/FinEntity`, revision `e86e5451a1`, ODC-BY; Reuters
    paragraphs with entity-level sentiment by human annotators). Kept: the 851 paragraphs whose
    entities all carry the same label; that label is the paragraph's (Positive / Neutral /
    Negative → positive / neutral / negative; the `tag` and `label` fields agree on every
    entity). Leak check done before this rule was written: 0 exact and 0 near matches against
    every set below (highest cosine 0.66).
  - **Tweets, primary:** 300 tweets from the equinxx set (30 Sep 2021 – 29 Sep 2022), hand
    labelled blind under the rules below. Eligible: after the engine's cleaning
    (`load_equinxx`), linked to exactly one of our companies, at least 5 words. Sample: 30 per
    company for the 10 companies with at least 50 eligible tweets (TSLA, AAPL, AMZN, META, AMD,
    NFLX, GOOGL, DIS, BA, KO). Within a company, tweets are taken in a random order (seed 21),
    skipping a tweet that matches (exact or cosine ≥ 0.9) any text in the sets below or a tweet
    already taken. The engine scored these tweets' sentiment in E17 only as daily averages;
    no per-tweet score is shown to the annotator.
  - **Tweets, secondary, already observed:** the E1 TFNS test (2,388). Never used to train or
    choose a model, but its scores for many models have been seen, so it is not a fresh test.
- **Sets checked for overlap:** TFNS train and test, PhraseBank (all sentences), FiQA-2018 (all
  splits), NWGI train and test, SEntFiN, FinEntity, and the E21 training and dev rows.
- **Annotation rules (tweets):** one label per row for the named company:
  - *positive:* the tweet says or implies good news, a rise, or a bullish view for that company
    or its stock (for example "TSLA up 5% on deliveries", "loading more AAPL").
  - *negative:* bad news, a fall, or a bearish view ("shorting NFLX", "Disney misses").
  - *neutral:* no clear direction: plain facts without good or bad meaning, questions, links,
    mixed views with no dominant side.
  - *skip:* not about this company, not English, spam or an advert, or unreadable. Skipped rows
    are left out and counted.
  - Sarcasm gets its intended meaning. The annotator labels from the text alone: no prices, no
    model outputs, no other tools. Notes are optional. One annotator (the author); this is
    reported. The file holds no model predictions or suggested labels.
- **Training data** (human labels only; leak-filtered against every test and dev row above):
  TFNS train minus the E7 dev split; PhraseBank `Sentences_75Agree`; FiQA-2018 all splits
  (FiQA is no longer a test); SEntFiN single-entity headlines minus the news dev split.
  **NWGI is not used for training** (its labels come from a proprietary model) and is not used
  in E21 at all.
- **Dev sets:** tweets: the E7 TFNS dev split (937). News: 10% of SEntFiN single-entity
  headlines, stratified by label, seed 21. SEntFiN was the E20 confirmation test; from E21 on
  it is training and dev data and is no longer reported as held out.
- **Candidates:** `BAAI/bge-small-en-v1.5`, E5 recipe (max length 64, batch 32, AdamW lr 5e-5,
  4 epochs, 10% warm-up), seeds 42, 43, 44, trained on Kaggle. Two weightings: (C1) every row
  weight 1; (C2) TFNS rows weight 2 in the loss. Dev score = mean of tweet-dev and news-dev
  macro-F1; each run keeps its best epoch by dev score; the best run over both weightings and
  all seeds is the candidate C. C is exported to int8 (E9 recipe).
- **Choice (on dev, int8 for both):** C replaces the engine model A only if C's dev score is
  higher than A's by more than 0.01 **and** C's tweet-dev macro-F1 is no more than 0.01 below A's.
- **Confirmation (once, after the choice, A and C int8):** FinEntity, the 300 tweets, and TFNS
  test (marked observed). Metrics: macro-F1 (primary), accuracy, per-class F1, each with a 95%
  bootstrap interval (1,000 resamples, seed 0), and the paired bootstrap of C − A macro-F1 on
  the same rows (higher / lower / tie as E18). The switch stands only if neither primary fresh
  set says "lower". Nothing is tuned after these scores; they are reported as they come.
- **Stored:** sample, labels, per-row predictions and a manifest (dataset revisions, file and
  model hashes) in `data/outputs/` and the experiment register.
- **Amendment (2026-10-11, after the dev choice, before any fresh test is scored):** the 300
  tweet labels were not made blind by one person as planned. They were drafted by three
  general-purpose AI assistants (ChatGPT and Gemini independently; Claude settled the rows where
  they disagreed), then reviewed and finalised by the author, under the rules above. So:
  - The tweet set is reported as **AI-assisted, human-reviewed**, and as **secondary**. It is
    not a human-labelled blind test, and no claim rests on it alone.
  - **FinEntity is the only primary fresh test.** The switch condition becomes: FinEntity's
    paired difference is not "lower" (the engine choice was already A on dev, so this changes
    no decision).
  - The assistants were used only to label this test set; none is part of the engine.
  - File: `data/annotation/e21_tweets_final_labels.csv` (hash in the results).
- **Results (2026-10-11, `data/outputs/e21_results.json`, per-row predictions in
  `e21_predictions.csv.gz`).** Training: best run C2 (TFNS weight 2), seed 42, epoch 4. Dev,
  int8: A 0.864 tweets / 0.734 news, C 0.814 / 0.858; C loses 0.049 on tweets (limit 0.01), so
  **A stays**. Confirmation, scored once, macro-F1 [95% interval], paired C − A:

  | Set | A (engine) | C (E21 model) | C − A | |
  |---|---|---|---|---|
  | FinEntity, 851 (primary, human) | 0.765 [0.734, 0.794] | 0.700 [0.669, 0.731] | −0.065 [−0.094, −0.036] | lower |
  | Tweets 2021–22, 251 of 300 (49 skipped; AI-assisted, human-reviewed) | 0.452 [0.385, 0.517] | 0.591 [0.530, 0.654] | +0.139 [0.079, 0.201] | higher |
  | TFNS test, 2,388 (observed before) | 0.836 [0.818, 0.853] | 0.841 [0.823, 0.857] | +0.004 [−0.012, 0.021] | tie |

  - The dev sets pointed the wrong way on both fresh sets: C was better on news dev (SEntFiN,
    Indian headlines) but worse on FinEntity (Reuters paragraphs), and worse on tweet dev (TFNS)
    but better on the 2021–22 tweets. Dev sets drawn from the training sources did not predict
    performance on new sources.
  - The engine model scores 0.45 on the 2021–22 tweets, far below its TFNS 0.84: retail tweets
    from a later year are much harder than TFNS's news-style tweets.
  - Engine unchanged; nothing tuned after these scores.

## Result

- **Question:** One sentiment model for tweets + news (trained on Kaggle, confirmed on fresh tests)
- **Outcome:** dev int8: engine 0.864 / 0.734, new model 0.814 / 0.858 → engine stays. Fresh: FinEntity 851 (human) engine 0.765 vs 0.700 (paired −0.065, lower); 2021–22 tweets 251 (AI-assisted, human-reviewed) 0.452 vs 0.591 (+0.139, higher); TFNS tie
- **Verdict:** engine stays (wins on primary FinEntity); E21 model kept as a lead for retail tweets; cross-source generalisation = future research
- **Stored results:** `e21_results.json`, `e21_predictions.csv.gz`, `e21_training_*.json`, `e21_sample.json`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
