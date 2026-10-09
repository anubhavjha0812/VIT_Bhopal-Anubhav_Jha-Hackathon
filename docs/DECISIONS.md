# Evaluation decisions

Each rule below is written **before** the evaluation it governs is run, and is not changed
afterwards to improve a score. Dates are when the rule was fixed.

## E1. Sentiment test set (fixed 2026-10-06)

- **Set:** Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`,
  MIT), file `sent_valid.csv`, all 2,388 rows. Never used for training or tuning.
- **Labels:** Bearish → negative, Bullish → positive, Neutral → neutral.
- **Prediction:** the model's most likely of the three classes (argmax).
- **Metrics:** accuracy and macro-F1 over the three classes, plus per-class F1.
- **Not used to test FinBERT:** Financial PhraseBank, because FinBERT was trained on it.
  It may be reported for our own model only, with that caveat.

## E2. Event-class test set (fixed 2026-10-06)

- **Set:** Twitter Financial News Topic (`zeroshot/twitter-financial-news-topic`, MIT),
  file `topic_valid.csv`, all 4,117 rows. Never used for training or tuning.
- **Mapping to our classes** (fixed before scoring any model):

  | TFN-topic label | Our class |
  |---|---|
  | Fed \| Central Banks, Macro | Macroeconomic |
  | M&A \| Investments | M&A |
  | Politics | Geopolitical |
  | all other 16 labels | Other |

- Our classes Default, Credit downgrade, Covenant breach and Product launch have no clean
  TFN-topic equivalent, so they are **not** scored here. They are scored on our hand-labelled
  gold set (E3).
- **Metric:** macro-F1 over the four mapped classes, plus per-class precision, recall and F1.

## E3. Gold set (sampling fixed 2026-10-06, before any credit-class model is trained)

- **Pool:** FNSPID headlines for our 15 stocks, 2019-01-01 to 2020-06-10, after cleaning
  (`ingest/news.py`).
- **Sample:** 300 headlines, seed 7. Half (150) from headlines containing any broad
  credit-related word (rating, Moody's, Fitch, downgrade, default, bankrupt, Chapter 11,
  covenant, lender, debt, bond, credit, loan), half (150) at random from the rest. The
  credit-word half exists only so rare credit classes appear at all; it is broader than the
  keyword rules on purpose, so it does not favour them.
- **Labelling:** by hand, one of our 8 classes per headline, from `data/gold/gold_to_label.csv`
  which shows no model output and no keyword guess. Unsure → "Other".
- **Frozen** once labelled (`data/gold/gold.csv`). These 300 headlines are excluded from every
  training set (including any Gemma-labelled data) and are scored once per final model.
- **Metric:** macro-F1 over the 8 classes, plus per-class precision, recall and support.

## E5. Fine-tuning setup (fixed 2026-10-06, before any training run)

- **Training data:** only the *train* files, `sent_train.csv` (9,543) and `topic_train.csv`
  (16,990). The *valid* files used as test sets in E1/E2 are never downloaded by the training
  notebook.
- **Model selection:** 10% of each train file, stratified, seed 42, is held out as a dev set.
  Epoch choice and the final checkpoint use dev macro-F1 only.
- **Backbones:** `ProsusAI/finbert` (accuracy) and `BAAI/bge-small-en-v1.5` (speed), same
  recipe for both: max length 64, batch 32, AdamW lr 3e-5 (bge-small 5e-5), 4 epochs,
  linear warm-up 10%, seed 42.
- **Events:** the topic model predicts the 20 TFN topics; topics are mapped to our classes
  with the E2 table *after* prediction, so training never sees our class names.
- Each fine-tuned model is scored once on the E1/E2 test sets after training ends. Results
  are reported whether better or worse than the baseline.
- **Which model becomes the engine default** (fixed after training, before any test scoring):
  the one with the higher best dev macro-F1; if the two are within 0.01, the smaller one
  (faster on CPU). All trained models are still reported on the test sets.

## E6. Learned impact score (fixed 2026-10-06, before any impact model is fitted)

- **Rows:** FNSPID headlines for our 15 stocks after cleaning, one row per (headline, stock).
  Gold-set headlines (E3) are dropped.
- **Event day:** first trading day whose New York close (16:00) is after the headline time.
- **Target:** |CAR[0,+1]|, the absolute sum of market-model abnormal returns on the event day and
  the next. Market = S&P 500 (^GSPC). Alpha and beta per stock from daily returns over trading
  days -280 to -31 before the event day (min 120 days, else the row is dropped).
- **Features:** only what is known at publication time: event class and 20 topic
  probabilities (our topic model), sentiment and |sentiment| (our sentiment model), source,
  publisher (top 10, else "other"), headline length, number of headlines for the stock in the
  previous 24 h, the stock's 20-day realised volatility and the VIX close, both from the day
  before the event day.
- **Time split:** train 2009–2016, validation 2017–2018, test A 2019, test B 2020-01-01 to
  2020-06-10. Test years are used once, after the model is chosen.
- **Candidates:** the fixed formula (baseline), ridge regression, gradient-boosted trees
  (scikit-learn `HistGradientBoostingRegressor`, used in place of LightGBM so the image needs
  no extra system library). Chosen by validation Spearman. Targets are log(1 + |CAR| in %).
- **Refit:** the chosen model is refitted on train + validation (2009–2018) before testing.
- **Scale 1–10:** deciles of the refitted model's predictions on 2009–2018.
- **Metrics (per test year):** Spearman between impact and |CAR| with a 95% bootstrap interval
  (1,000 resamples, seed 0); mean |CAR| for impact ≥ 8 vs ≤ 3.

## E7. Sentiment teacher and distilled student (fixed 2026-10-09, before any training run)

- **Leak found in E5 data:** 84 texts in `sent_train.csv` match a text in the E1 test set once
  URLs, case and punctuation are removed. The E5 models were trained with them; their E1 scores
  stay reported as they are, with this caveat. E7 removes them.
- **Training data:** `sent_train.csv` + Financial PhraseBank `Sentences_75Agree` (3,453, CC BY-NC-SA
  3.0) + FiQA-2018 sentiment train and validation splits (`pauri32/fiqa-2018`, 1,063; label
  0 → positive, 1 → neutral, 2 → negative). Every text whose normalised form (lower case, URLs
  removed, non-alphanumerics → space) matches an E1 test text is dropped. The test file is read
  for its `text` column only, never its labels. Texts that appear more than once with different
  labels (FiQA repeats a sentence per aspect) are dropped; remaining duplicates keep one copy. FiQA's test split (150) is not used for anything.
- **Consequence:** Financial PhraseBank can no longer be reported as a test for our models (E1).
- **Dev set:** the same 10% of `sent_train.csv` as E5 (stratified, seed 42), minus the dropped
  rows. PhraseBank and FiQA rows are never in dev, so dev measures the TFNS domain only.
- **Teacher:** `microsoft/deberta-v3-base`, max length 64, batch 32, AdamW lr 2e-5, 4 epochs,
  10% linear warm-up, seeds 42, 43, 44. Each seed keeps its best epoch by dev macro-F1; the seed
  with the best dev macro-F1 is the teacher.
- **Student (distillation):** `BAAI/bge-small-en-v1.5`, same data, dev set and recipe as E5 except
  6 epochs. Loss = 0.5 × cross-entropy on the hard label + 0.5 × T² × KL(teacher ‖ student) at
  temperature T = 2. Teacher probabilities come from the chosen teacher on the training texts.
  Seeds 42, 43, 44; best seed by dev macro-F1.
- **Test:** the chosen teacher and the chosen student are each scored once on E1, with accuracy,
  macro-F1, a 95% bootstrap interval (1,000 resamples, seed 0) and E4 efficiency.
- **Engine default:** the student replaces the E5 bge-small model if its dev macro-F1 is higher.
  This is decided before test scoring.

## E8. Impact score v2 (fixed 2026-10-09, before any v2 model is fitted)

- **Why:** in v1 (E6) the stock's recent volatility alone ranked moves better than the chosen
  model (Spearman 0.27–0.34 vs 0.17). Volatile stocks move more whatever the news says, so
  volatility-only is the fair naive baseline. v1 stays reported as it is.
- **Bug found in v1:** pandas 3 stores these timestamps in microseconds, but the "headlines in
  the previous 24 h" feature assumed nanoseconds, so in v1 it counted roughly the previous
  1,000 days. v1 results stay reported as they are, with this caveat. v2 uses the fixed feature
  (with a unit test).
- **Rows, event day, CAR:** as E6. Headline features use the same E5 sentiment and topic models
  as v1, so v1 and v2 differ only in target, features and model.
- **Target:** z = |CAR[0,+1]| / (√2 × vol20), the move in units of the stock's normal two-day
  move. Fit on log(1 + z). **Impact** = predicted z × √2 × vol20, i.e. the expected size of
  the move; deciles of it on 2009–2018 give the 1–10 scale.
- **Features:** E6 features plus **novelty** = 1 − the highest cosine similarity between the
  headline and any headline for the same stock in the previous 72 hours (word 1–2-grams,
  hashing vectoriser, no fitting); 1 when there is none. |sentiment| is already in E6.
- **Baselines:** volatility-only (impact = vol20, the official baseline) and the E6 formula.
- **Candidates:** ridge, gradient boosting (E6 settings), and their average on the log(1 + z)
  scale (ensemble).
- **Choice:** rolling-origin validation. Train on 2009 up to year Y − 1, validate on Y, for
  Y = 2015, 2016, 2017, 2018. Score = mean Spearman between impact and |CAR| over the four years.
  The highest mean wins. The chosen model is refitted on 2009–2018.
- **Test F (fresh, primary):** S&P 100 members (2020 list) other than our 15, headlines
  2019-01-01 to 2020-06-10 from FNSPID by their `Stock_symbol`, prices from yfinance. Every
  symbol is included; rows without enough price history for the market model drop out as in E6.
  No row of test F is used for fitting or choosing. Scored once; 2019 and 2020 reported
  separately.
- **Test S (seen):** our 15 stocks in 2019 and 2020. These years were already looked at in v1,
  so test S is reported for continuity only and marked as seen.
- **Metrics:** E6 metrics (Spearman vs |CAR| with 95% bootstrap interval, mean |CAR| for impact
  ≥ 8 vs ≤ 3), plus Spearman vs z, which shows how much a model knows beyond volatility.
- **Success:** the chosen model beats volatility-only on Spearman vs |CAR| in both years of
  test F. Reported either way.
- **Engine default:** v2 replaces v1 in the engine if its mean rolling-validation Spearman beats
  volatility-only. Decided before testing.

## E9. int8 ONNX version (fixed 2026-10-09, before any int8 model is scored)

- **Which model:** the sentiment model chosen as engine default under E7 (or the E5 one if E7
  does not replace it).
- **How:** export to ONNX (opset 17), then onnxruntime dynamic quantisation, int8 weights
  (`nlp/onnx_int8.py`). No calibration data, no further training.
- **Test:** scored once on E1 with the E7 metrics and E4 efficiency. Size = the `model.onnx` file
  (for torch models: weights in memory, which match their fp32 file size).
- **Engine default:** int8 replaces the torch model only if its E1 accuracy is within 1 point of
  the torch model's. This is the one choice made on test scores; both are reported.

## E10. One shared encoder with two heads (fixed 2026-10-09, before any training run)

- **Why:** the engine runs a sentiment model and a topic model on every text. One encoder with a
  sentiment head and a topic head gives both in one pass. Those two outputs are also the
  headline inputs of the impact model (sentiment, |sentiment|, event, topic probabilities).
- **Model:** `BAAI/bge-small-en-v1.5` encoder, first-token vector, dropout 0.1, one linear head
  for the 3 sentiment classes and one for the 20 TFN topics.
- **Leak found in E5 topic data:** 999 texts in `topic_train.csv` (903 train, 96 dev) match an
  E2 test text after the E7 normalisation. The E5 topic models were trained with them; their E2
  scores stay reported as they are, with this caveat.
- **Data:** sentiment = the E7 training data and dev set, with the E7 teacher's probabilities;
  topic = `topic_train.csv` with the E5 dev split (10%, stratified, seed 42). Topic texts that
  match an E2 test text after the E7 normalisation, or repeat with different labels, are
  dropped, as in E7.
- **Training:** each step takes one sentiment batch and one topic batch (the smaller set is
  cycled); loss = E7 sentiment loss (0.5 hard + 0.5 × T² × KL, T = 2) + topic cross-entropy.
  Batch 32, max length 64, AdamW lr 5e-5, 6 epochs, 10% warm-up, seeds 42, 43, 44.
- **Choice:** per seed, the epoch with the best mean of sentiment and topic dev macro-F1; then
  the seed with the best such mean.
- **Test:** scored once on E1 (sentiment head) and E2 (topic head, mapped with the E2 table),
  with E7 metrics and E4 efficiency. Compared with the single-task bge-small models (E7 student
  for sentiment, E5 bge-small topic model).
- **Success:** accuracy within 1 point of the single-task model on both tests. Reported either way.

## E4. Efficiency (fixed 2026-10-06)

Next to every accuracy figure: CPU throughput (texts per second, batch 64, after a warm-up
batch), model size on disk, and peak resident memory of the scoring process.
