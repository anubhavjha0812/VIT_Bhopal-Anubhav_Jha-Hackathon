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

## E11. Engine switch to our models (fixed 2026-10-09, before the topic int8 model is scored)

- **Sentiment:** the E9 int8 model (bge-small fine-tuned, int8 ONNX).
- **Events:** the E5 bge-small topic model, exported to int8 as in E9, replaces the E5 choice
  (FinBERT topic). This choice is made after the E2 test scores were seen (0.960 vs 0.966), so
  it is stated as a trade: 0.6 points of accuracy for a model about 1/13 the size and ~3.5×
  the speed, which the engine needs on CPU. Both stay reported.
- **Topic int8 test:** scored once on E2 with E9 metrics. It is used only if its E2 accuracy is
  within 1 point of the bge-small torch topic model; otherwise the torch model is used.
- **Impact:** stays the formula (E8 v2 is not used; it failed the 2020 test).
- **Demo:** the committed sample is re-scored once with the new models; signal counts, event
  mix, high-impact count and Module A metrics are reported next to the baseline run.

## E12. A bank that only gives a stock rating is not the subject (fixed 2026-10-09, before the code)

- **Why:** a hand check of the 41 high-impact demo texts found JPM and BAC linked to news about
  other companies ("iRobot $IRBT Downgraded by JPMorgan Chase & Co. to Underweight" → JPM).
  These would trigger Module B on the wrong bank.
- **Rule:** JPM or BAC (our two universe companies that publish stock ratings) is tagged
  `analyst` instead of `subject` for a text when all three hold: (1) the text is about a stock rating (downgrade, upgrade,
  rating, price target / PT, overweight, underweight, outperform, underperform, neutral,
  equal-weight, market/sector perform, "to buy/sell/hold", coverage); (2) the bank is named as
  the analyst: at the start followed by a rating action, after "by" / "at", or as "<bank>
  analysts"; (3) the bank's own cashtag is not in the text. Applied at scoring time; ingest is
  unchanged, so the impact training data is unchanged. `analyst` rows are kept and shown (audit
  trail) but Module A, the alert count and Module B use `subject` rows only.
- **Seen before fixing:** the rule was drafted while reading demo texts, and a prototype was run
  on the demo signals (128 links removed; 25 sampled, all correct; 2 Disney links wrongly
  removed, which is why the rule is limited to the two banks). So the demo numbers are a design
  check, not a test.
- **Check:** every `analyst`-tagged link on the demo is read by hand and reported as "X of N correct";
  30 remaining JPM/BAC rating texts are read for misses. Rules first; a learned "which company
  is this news about" model may replace them later and is compared against them.

## E13. Credit rules and the Module B trigger gate (fixed 2026-10-09, before the code)

- **Why:** the credit keyword rules fire on stock-analyst rating changes and on "default search
  engine", and miss "cuts … to junk". Of 41 demo texts with impact > 7, 1 is a real credit
  event; two more real ones (Moody's–Boeing, S&P–Disney) scored 7 and were missed.
- **Analyst rating:** a rating action (downgrade, upgrade, initiate, reiterate, maintain, cut,
  raise, lower) with a stock-rating word (buy, sell, hold, neutral, overweight, underweight,
  outperform, underperform, equal-weight, market/sector perform, price target / PT) and no
  agency or credit word is never a credit event. The class stays the topic model's (8 classes
  unchanged); the text is flagged `analyst_rating`.
- **Credit downgrade:** a cut word (downgrade, cut, lower, slash) with a rating agency (Moody's,
  Fitch, DBRS, Standard & Poor's, S&P but not the S&P index) or a credit word (junk, credit
  rating, IDR, investment grade, high yield, bond, debt rating, CreditWatch); or an agency with
  "outlook … negative".
- **Default:** bankrupt, Chapter 11, insolvent, missed coupon/interest/debt payment, "defaulted",
  "defaults on". Plain "default" only with a debt word in the same text (debt, loan, bond,
  coupon, payment, creditor, lender, obligation).
- **Covenant breach:** the existing rule. These rules replace the keyword credit override in the
  engine only; the keyword baseline (E2) and the impact training data are unchanged.
- **Impact:** +1 when a rating agency is named (an official rating action, like the existing +1
  for news). Chosen after seeing that the two missed real events scored 7.
- **E12 addition:** "<bank> analysts / strategists / economists" makes the bank an analyst even
  without rating words ("Bank of America analysts see oil at $40").
- **Gate (Credit Chhanni):** a (text, company) row triggers Module B only if (1) the company is
  the subject (E12); (2) it is the obligor: named in the text and, if the text has cashtags,
  the first cashtag is its own; (3) the event is a credit class; (4) impact > 7 (case study);
  (5) a rating agency is named, or a different text on the same company and credit class
  arrived in the previous 24 hours.
- **Check:** the demo numbers (naive vs gated alerts, every gated alert read by hand, the 3 real
  events above as "should fire") are a design check, since those texts were read while writing
  this rule. The test is the gold set (E3), labelled blind: per-class precision and recall for
  the credit classes, keywords vs these rules, scored once. A learned credit model may be
  compared against these rules later.
- **Amended after the first demo check (2026-10-09, before the gold test):** the check showed
  "Google pays Apple to be the default search engine … such payments …" as Default and missed
  "Moody’s" written with a curly apostrophe. So: plain "default" needs a debt word within three
  words of it; hedged bankruptcy ("won't go bankrupt", "may file for bankruptcy", "brink of
  bankruptcy") is not Default, as the gold labelling guide already says; agency names accept
  either apostrophe. The demo is re-scored once more after this.

## E14. Module B book, shock and capital rules (fixed 2026-10-10, before the code)

- **Why:** Module B turns a gated credit alert (E13) into a loss and a capital ratio. The book,
  parameters and rules are fixed here so no number is tuned after seeing results.
- **Bank:** an Indian bank (capital in INR, RBI floor) with US corporate exposures in USD. The USD
  assets are funded in USD, so a rupee move changes RWA in INR, not P&L.
- **Obligors:** our 15 universe companies, S&P issuer ratings at 31 Dec 2019, each with a source in
  `ratings.csv`. Book date 31 Dec 2019, before every 2020 shock.
- **Size:** the bank holds 0.5% of each company's total debt at its last filing on or before
  31 Dec 2019 (SEC EDGAR XBRL, tag and accession saved). JPM and BAC are capped at the largest
  non-bank exposure. 0.5% is an illustrative mid-size bank. **Sensitivity:** the run is repeated at
  0.1%, 0.5%, 1% and 2%; if the CET1 change differs by more than 0.1 point across them, the share
  is instead derived from real data (a real bank's corporate book ÷ US corporate debt).
- **Split:** revolver = the company's credit line (XBRL tag, else a 10-K sentence found by regex
  and read by hand), else its short-term debt; bond = long-term debt, with cash flows from the
  10-K repayment schedule; term loan only where the 10-K reports one. Revolver EAD uses a 75%
  credit conversion factor (Basel F-IRB).
- **Hedges:** CDS protection bought on the 3 lowest-rated obligors (25% of exposure) and sold on
  the 3 highest-rated (10%); pay-fixed IRS on the 5 largest (20%), valued for CVA. Bought
  protection must gain when spreads widen (tested).
- **PD:** S&P long-run one-year default rates by letter grade, log-interpolated by notch, floor
  0.03% (Basel). D = 100%.
- **Spread:** FRED Moody's Aaa and Baa over the 10-year Treasury, log-linear by notch, extrapolated
  below BBB (this understates junk spreads; stated).
- **Shock on an alert:** notches parsed from the text ("Baa1 to Baa2"); none found = 1 notch;
  Default = D. The obligor moves to the new rating's PD and spread. Other obligors widen by the
  obligor's widening × their daily return correlation with it (2008–2019, negatives set to 0).
  Plus a market episode (GFC 2008, euro 2011, COVID 2020 by default): FRED Aaa/Baa spread moves,
  the 15-stock move and the USD/INR move (FRED `DEXINUS`) over the episode dates; `flat` or
  `scaled` by impact/10 (`config.shock`).
- **Stressed PD:** Vasicek `N((N⁻¹(PD) + √ρ·z)/√(1−ρ))`, ρ from the Basel corporate formula, z the
  episode's equal-weight 15-stock move in standard deviations (2008–2019 daily vol).
- **IFRS 9:** Stage 2 if cut ≥ 3 notches since the book date or below BBB-; Stage 3 at D. ECL:
  Stage 1 12-month PD, Stage 2 lifetime PD `1−(1−PD)^M`; LGD 45%.
- **Capital:** Basel IRB corporate RWA (F-IRB, LGD 45%, M from the position capped 1–5 years) on the
  stressed PD, converted to INR at the scenario USD/INR. CET1 starts at 13% of RWA; losses after
  25% tax; floor 8% (RBI: CET1 5.5% + capital conservation buffer 2.5%).
- **Reverse stress:** the multiple of the episode at which CET1 reaches 8% (bisection).
- **Monte Carlo:** 20,000 draws, Gaussian copula with the 15-stock return correlation (calm
  2008–2019, stressed 14 Feb – 23 Mar 2020), seed 42; VaR and ES at 99%, before and after.
- **Memo:** a fixed template, no LLM. Every number comes from the run; the trigger headline, the
  matched words and the gate reason (E13) are quoted.
- **Tests:** at least 30 for Module B: IRB against a hand calculation (PD 1%, LGD 45%, M 2.5),
  convexity against full repricing for 200 bp, hedge sign, staging, notch parsing, PD table
  rising as ratings fall, Vasicek at z = 0, Monte Carlo mean at zero correlation, reverse stress
  landing on 8%, FX raising RWA when the rupee falls.
- **Check 1, real alerts:** for the 3 real alerts (Moody's–Boeing, S&P–Disney, Fitch–Disney), the
  predicted change in the bond and CDS book is compared with the book repriced at the realised
  FRED moves over the next 20 trading days: sign match and error vs a flat-episode naive. FRED
  gives rating buckets, not company spreads, so this checks direction and scale, not
  company-level accuracy. Scored once.
- **Check 2, crisis backtest:** episodes are chosen by a rule, not by hand: every non-overlapping
  window in 1990–2023 where FRED `BAA10Y` widens by 75 bp or more within 60 trading days. For each
  episode, its first 5 trading days (changes in Baa, Aaa, VIX, S&P 500, USD/INR) are matched,
  leaving that episode out, to the 3 nearest other episodes (standardised Euclidean distance);
  their average 60-day moves are the prediction. The book (P&L and CET1) is valued at the
  predicted and at the realised 60-day moves. Reported: sign match and mean absolute error in
  P&L and CET1, against a naive prediction (the average of all other episodes). Scored once.
- **Later, if time:** equity puts (Black-Scholes, VIX as volatility) as a non-linear hedge.
- **Clarified while coding (2026-10-10, before any loss or capital number was computed):** z is
  measured in one-year standard deviations (2008–2019 daily vol × √252), because Vasicek's market
  factor belongs to the one-year PD. The first draft used the episode's own length (COVID z = 5.8)
  and gave an AAA name a 25% stressed PD; now COVID z = 1.83, GFC 1.72, euro 2011 0.94. The
  stressed PD is never below the rating's normal one-year PD.
- **Amended after the first run (2026-10-10, after seeing losses and CET1; before the checks):**
  the first run gave COVID CET1 13% → −4%, for two reasons, both fixed by Basel's own rules, not
  by tuning. (1) RWA rose 2.4× because the IRB formula was fed the Vasicek stressed PD; Basel IRB
  uses the long-run PD of the rating, so a downgrade raises RWA and a bad market alone does not.
  The stressed PD now drives IFRS 9 ECL only. (2) The book is 85% bonds ($2.6bn against $139m of
  capital), because the bank holds 0.5% of each company's long-term debt; a bank with a bond book
  also holds market-risk capital for it. Bonds now add Basel standardised market-risk RWA
  (specific risk: investment grade 0.25% / 1.0% / 1.6% by maturity, BB 8%, below BB- 12%;
  general interest-rate risk by maturity band, 0.7%–6%). The user chose this over holding the
  bonds at amortised cost or keeping the first result. Market risk of CDS and swaps is not
  modelled. Rejected runs and their numbers are kept in STEPS.md.

## E4. Efficiency (fixed 2026-10-06)

Next to every accuracy figure: CPU throughput (texts per second, batch 64, after a warm-up
batch), model size on disk, and peak resident memory of the scoring process.
