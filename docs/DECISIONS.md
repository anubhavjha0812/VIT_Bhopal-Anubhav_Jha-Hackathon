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

## E4. Efficiency (fixed 2026-10-06)

Next to every accuracy figure: CPU throughput (texts per second, batch 64, after a warm-up
batch), model size on disk, and peak resident memory of the scoring process.
