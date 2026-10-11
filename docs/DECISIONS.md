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
- **Credit gate:** a (text, company) row triggers Module B only if (1) the company is
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

## E15. State-aware market move, monthly backtest (fixed 2026-10-10, before the code)

- **Why:** Check 2 (E14) found that the first 5 days of a crisis do not predict its next 60 days
  (analogs $45.7m error vs naive $47.0m). Credit spreads are known to mean-revert: when they are
  already wide, the next months more often bring tightening. This is a new, separate test; Check 2
  stays as reported and is not re-run. The idea came after seeing Check 2's misses, so Check 2's
  episodes are not used to judge it.
- **Data:** FRED daily Baa, Aaa, VIX, USD/INR and S&P 500 (`market_history.csv`, 1990–2023).
- **Origins:** the last trading day of every month, Jan 1995 – Sep 2023 (~345), the last one
  with 60 trading days of data after it.
- **Target:** the 60-trading-day moves from the origin, as in Check 2: change in Baa, Aaa and
  VIX, S&P 500 log return, USD/INR return.
- **State at the origin:** Baa level, VIX level, Baa change over the previous 20 trading days.
- **Model:** for each target, least squares on the 3 state values plus a constant, fitted only on
  earlier month-end origins (from Feb 1990) whose 60-day window ended on or before the origin
  date. Refitted at every origin. Nothing from the future, no tuning.
- **Naive:** the average of the same earlier moves (same data, no state).
- **Valued on the book:** each predicted, naive and realised move is applied to our 31 Dec 2019
  book exactly as in Check 2 (loss in USD and CET1 after).
- **Pass rule:** the model passes if its mean absolute loss error is at least 15% below the
  naive's (model MAE ≤ 0.85 × naive MAE). Also reported, not part of the rule: CET1 error, sign
  match, the share of months the model is closer, and a Diebold–Mariano test with Newey–West
  errors (lag 2, as the 60-day windows of nearby months overlap).
- **Scored once.** Pass or fail, the result goes into the README as it comes out.
- **If it passes:** Module B shows the state-aware 60-day move as a "most likely" view next to the
  stress episodes. Capital (CET1, reverse stress) still uses the stress episodes, because a stress
  test is meant to be severe, not likely. If it fails, Module B stays as it is.
- **Tests:** the training set never includes a window ending after the origin; the naive equals
  the mean of earlier moves; on a synthetic mean-reverting series the model beats the naive.

## E16. Nine learned event classes (fixed 2026-10-10, before any training or scoring)

- **Why:** with the E2 table the model learns only Geopolitical, Macroeconomic, M&A and Other.
  "Product launch" is never predicted (the topic data has no such class: 134 launch texts in the
  demo window by keywords, 0 by the model), and earnings, analyst, legal and personnel news all
  fall into Other. The TFN topics already separate these, so the mapping is widened.
- **Classes:** 9 learned + 3 credit rule classes = 12. "Product launch" is renamed
  "Product news" (the TFN topic is any company or product news, not only launches).
- **Mapping** (fixed now, never changed after seeing results; all 20 TFN topics):

  | TFN-topic label | Our class |
  |---|---|
  | Politics | Geopolitical |
  | Fed \| Central Banks, Macro, Treasuries \| Corporate Debt, Energy \| Oil, Gold \| Metals \| Materials, Currencies | Macroeconomic |
  | M&A \| Investments, IPO | M&A |
  | Company \| Product News | Product news |
  | Earnings, Dividend | Earnings |
  | Analyst Update | Analyst rating |
  | Legal \| Regulation | Legal/Regulatory |
  | Personnel Change | Management change |
  | General News \| Opinion, Markets, Stock Commentary, Stock Movement, Financials | Other |

- **Credit classes:** Default, Credit downgrade and Covenant breach stay the E13 rules (frozen)
  and override the model, as now. They are scored later on the gold set.
- **Gold set (E3):** not labelled yet, so its label list grows from 8 to these 12 classes. The
  sample itself is unchanged.
- **Impact severity** (E6 formula; old classes unchanged, fixed before anything is re-scored):
  Product news 4 (as Product launch), Earnings 5, Analyst rating 4 (an opinion on the stock, not
  on credit), Legal/Regulatory 6 (fines and lawsuits reach cash, like Geopolitical and Macro),
  Management change 5. Only credit classes can trigger Module B, so these move impact and
  Module A only.
- **Keyword baseline:** rules for the new classes, written before scoring. Order (first match
  wins): Default, Credit downgrade, Covenant breach, Analyst rating (the E13 stock-rating rule),
  M&A, Legal/Regulatory (lawsuit, sues, SEC, antitrust, probe, fine, court, settlement),
  Management change (CEO, CFO or chief executive with steps down, resigns, appoints or names),
  Earnings (earnings, EPS, beats or misses estimates, quarterly results, dividend, guidance),
  Geopolitical, Macroeconomic, Product news (the old launch rule), else Other.
- **Training:** the E5 recipe (bge-small, max length 64, batch 32, AdamW lr 5e-5, 4 epochs, 10%
  warm-up, seed 42, the E5 dev split) on `topic_train.csv` with the E10 leak filter (texts that
  match an E2 test text, or repeat with different labels, are dropped). Two candidates:
  (A) a 9-class head trained on the mapped labels; (B) a 20-topic head, mapped after prediction.
  The higher dev 9-class macro-F1 wins; within 0.01, (A).
- **int8:** the winner is exported as in E9 and used only if its test accuracy is within 1 point
  of the full-precision model (as E11). Both are reported.
- **Test:** the E2 test set (4,117) under this mapping, scored once per model: keyword baseline,
  the current engine model (E11 int8, 20 topics remapped; it was trained with the 999 leaked
  texts found in E10, so its scores flatter it), the new model and its int8 version. Metrics:
  accuracy, macro-F1 over the 9 classes, per-class precision, recall and F1, with 95% bootstrap
  intervals (1,000 resamples, seed 0) for accuracy, macro-F1 and each class's F1.
- **Pass rule** (on the model the engine would use): (a) Product news F1 ≥ 0.85, and
  (b) Geopolitical, Macroeconomic and M&A each lose at most 2 F1 points against the current
  engine model under this same mapping. The 9-class macro-F1 is reported as it comes; it is not
  comparable to the 4-class E2 figures.
- **Engine:** if E16 passes, the new model replaces the E11 topic model. If it fails, the engine
  keeps the E11 model with this 9-class mapping. Either way the result goes into the README.

## E17. Longer history for Module A (fixed 2026-10-10, before any label or return is seen)

- **Why:** Module A has only been tested on the 43-day replay window, the one period with both news
  and tweets. That is too short to tell skill from luck. Our FNSPID slice (headlines filed under
  our 15 symbols) is thin before 2017 (289 in 2010, 1,245 in 2016), but the same file has 17,000
  to 37,000 headlines a year that name one of our companies under another symbol.
- **Name-linked headlines:** `fetch_fnspid --by-name` keeps a headline filed under another symbol
  when it names one of our companies (`Linker`): 262,927 such rows. A sample of 100 (seed 7) showed
  about half only mention the company. The relevance rules in `ingest/relevance.py` (sha256
  `adc911bc335fd87d…`) were written from that sample and frozen before the samples below were
  drawn. In order: not English (or mis-encoded), another company carrying the name (per-company
  veto list), fund notice, holdings list, market list, former employer, for JPM/BAC also bank
  conference and bank as analyst (E12 rule plus research-desk patterns), and the company not named
  within the first 6 words. Headlines filed under our own symbol keep only the language rule.
- **Measuring the rules (labels never train anything):**
  - S1: 200 (headline, company) pairs drawn at random from all name-linked pairs dated 2010-01-01
    to 2020-06-10, before the rules, seed 17, excluding the 100 design headlines.
  - S2: 100 pairs drawn the same way from headlines filed under our own symbol, excluding the
    gold-set headlines (E3).
  - The label file shows only id, company and headline: no rule outcome, no sentiment, no price.
  - Label: is the headline about this company, news that matters for its stock (Y/N)? Three
    open-licence models label each pair with one fixed prompt, scored by the next-token
    probability of Y vs N: Qwen3-8B (Apache 2.0), OLMo-2-1124-7B-Instruct (Apache 2.0) and
    Phi-4-mini-instruct (MIT). The majority is the label. No closed model is used, so anyone can
    rebuild the labels with `notebooks/relevance_labels/`.
  - Amended 2026-10-10, after the first labelling run and before any precision was computed:
    OLMo-2 never answered Y (its Y probability was below 0.07 on all 300 pairs), so reading the
    next-token probability failed for it. All three models now answer by greedy generation of up
    to 4 tokens, read as the first Y or N; the next-token probability is the fallback when neither
    appears. A labeller that answers Y on fewer than 5% or more than 95% of pairs is reported as
    degenerate and left out; the label then needs both remaining models to agree, and pairs where
    they disagree are marked unsure, reported, and added to the human audit.
  - Human audit: 30 random labelled pairs (seed 0) checked by hand; agreement is reported.
  - Metrics: precision before and after the rules, recall of the rules (correct pairs kept / all
    correct pairs), each with a 95% Wilson interval, overall, by period (2010–2016, 2017–2020) and
    for JPM+BAC vs the rest; S2 precision.
  - Target: S1 precision after the rules ≥ 0.85, reported with its interval whatever it is. A
    period is labelled "weak data" only if its precision after the rules is below 0.80.
  - The rules do not change after the labels are seen. A change would need a new sample.
- **History set:** headlines under our symbol (language rule only) plus name-linked pairs that pass
  the rules, 2010-01-01 to 2020-06-10, rows where the company is the subject (E12). Sentiment from
  the engine's fine-tuned bge-small sentiment model. A headline is usable from the first trading
  day after its New York date (as Module A).
  - Amended 2026-10-10, before any headline was scored: the ~172,000 headlines are scored on a
    Kaggle GPU with the full-precision model (the engine runs its int8 copy, E9, which is CPU-only
    and would take about an hour). The same job scores a random 5,000 with the int8 copy and
    reports the correlation and sign agreement between the two (`notebooks/history_sentiment/`).
- **Signal test (IC):** on each trading day with at least 5 companies that have a usable headline in
  the last 3 trading days, the Spearman correlation across those companies between mean sentiment
  and the next day's return. Mean daily IC per period, with a 95% moving-block bootstrap interval
  (blocks of 20 days, 1,000 resamples, seed 0).
- **Portfolio test:** Module A as now (tilt 0.5 per z-score, cap 15%, 3-day lookback, 5 bp
  costs). Each day the universe is the companies with a close that day and the next; equal weight
  is 1/n over that universe. Four candidates: rebalance every 1 or 5 trading days × weights moved
  fully (λ = 1) or 20% of the way (λ = 0.2) to the target on rebalance days.
- **Periods:** choose on 2010–2016 (dev), by Sharpe minus equal-weight Sharpe. Score the chosen
  candidate once on 2017-01-01 to 2020-06-10 (test). The test period contains the replay window
  (9 Apr – 10 Jun 2020) whose result was already seen, so the test is also reported without it.
  Fresh test, scored once: equinxx tweets 30 Sep 2021 – 29 Sep 2022 (linked by their text),
  prices from yfinance.
- **Pass rules:** signal: test mean IC > 0 with the interval above 0. Portfolio: test Sharpe of the
  chosen candidate above equal weight; the Sharpe difference is reported with a moving-block
  interval, plus yearly return, volatility, drawdown and turnover.
- **Effect of the rules:** IC and the chosen candidate are also run on symbol-only headlines and on
  unfiltered name-linked headlines. Reported, never used to choose; the filtered set is primary.
- **Engine:** if both tests pass, the chosen candidate becomes Module A's default and the replay is
  re-reported. Otherwise Module A stays as it is and the README says the tilt has no proven edge.
- **Leaks guarded:** no labels train a model; labellers never see returns; design and measure
  samples are disjoint; gold-set headlines are excluded; dev and test years are separate.

## E18. More public sentiment test sets (fixed 2026-10-11, before any of them is scored)

- **Why:** E1 is one test set. Scores on the other widely used public sets let our engine be
  compared with other published results on identical files, not only on our own choice of data.
- **Sets** (each scored once per model, all rows unless dropped below):

  | Set | File | Rows | Labels |
  |---|---|---|---|
  | Financial PhraseBank | `takala/financial_phrasebank` v1.0, `Sentences_50Agree` and `Sentences_AllAgree` | 4,846 / 2,264 | as given |
  | FiQA-2018 sentiment | `pauri32/fiqa-2018`, test split | 150 | 0 → positive, 1 → neutral, 2 → negative (as E7) |
  | NWGI | `oliverwang15/news_with_gpt_instructions`, test split, column `news` | 4,047 | any positive → positive, neutral, any negative → negative |

- **Label caveat:** NWGI's labels were written by a GPT model, not by people. It is reported as
  a check against a widely used set, never as human-verified accuracy.
- **Models:** the engine model (E5 bge-small, int8, E9), its full-precision copy, the E5
  fine-tuned FinBERT, the FinBERT baseline, and the E7 teacher, E7 student and E10 shared encoder. Prediction and metrics as E1
  (argmax, accuracy, macro-F1, per-class F1, 95% bootstrap intervals with 1,000 resamples and
  seed 0), plus E4 efficiency. Input is cut at the models' 64 tokens, as in the engine.
- **Leak check (before any scoring):** each test text is compared with every text the model
  was trained on: `sent_train.csv` for the E5 models; plus PhraseBank `Sentences_75Agree` and
  FiQA train and validation for the E7 and E10 models; PhraseBank for both FinBERT models (their
  base model was fine-tuned on it).
  - Exact: the E7 normalised forms are equal.
  - Near: cosine similarity of word 1–2-gram TF-IDF vectors (fit on the two sets) ≥ 0.9.
  - Test rows with an exact or near match are counted, and the score is given on all rows and
    on the clean rows. The clean rows are the primary figure.
  - A model whose training set contains the test set itself is marked **in-sample**: the E7, E10
    and both FinBERT models on PhraseBank. Their PhraseBank scores are shown with that mark and are
    never counted as a result.
- **Claims:** "higher than" another published figure is written only when the files, rows and
  label mapping are the same and our 95% interval lies wholly above that figure. Otherwise the
  result is "within the interval" (a tie), or the comparison is "not like for like".
- **Engine:** these scores change nothing in the engine. They are reported as they come.
- **Amendment (2026-10-11, after the engine rows, before any comparison with another
  figure):** overlapping intervals are not by themselves a tie. When per-row predictions exist
  for both sides (our own models, or a public model we can run on the same rows), the
  comparison is a paired bootstrap of the macro-F1 difference on the same resampled rows
  (1,000 resamples, seed 0): "higher" only if its 95% interval lies above 0, "lower" if below,
  otherwise a tie. When only a published number exists, the rule above applies. Per-row
  predictions are now saved (`data/outputs/e18_predictions.csv.gz`), and what was scored is
  frozen in `data/outputs/e18_manifest.json` (dataset revisions, file and model hashes, library
  versions). Changes to input length or training data are separate experiments, each fixed in
  advance; E18 only measures.
- **Results, engine models (2026-10-11).** Leak check: the E5 models saw 0 rows of all four
  sets; the E7 and E10 models saw all of PhraseBank AllAgree, 3,461 of 4,846 PhraseBank 50Agree
  rows and 1 FiQA test row; both FinBERT models saw all of PhraseBank. Macro-F1, 95% interval:

  | Set | Engine (int8) | Full precision |
  |---|---|---|
  | PhraseBank AllAgree (2,264) | 0.866 [0.849, 0.881] | 0.880 [0.863, 0.895] |
  | PhraseBank 50Agree (4,846) | 0.685 [0.670, 0.700] | 0.702 [0.686, 0.717] |
  | FiQA-2018 test (150) | 0.573 [0.486, 0.649] | 0.602 [0.521, 0.676] |
  | NWGI test (4,047, GPT labels) | 0.406 [0.390, 0.419] | 0.414 [0.399, 0.428] |

  What this does and does not show:
  - *Measured:* no training overlap, so these are clean held-out scores.
  - *Reading:* a model trained only on tweets keeps 0.87 on the sentences annotators fully agree
    on, a promising sign of transfer from tweets to news sentences.
  - *Limitation:* long texts. NWGI is news paragraphs and the model reads 64 tokens; its
    neutral class scores 0.28 F1.
  - *Limitation:* FiQA has 150 rows, so its interval is about 16 points wide.
  - *Limitation:* NWGI labels come from a GPT model; they are not a human-labelled benchmark.
  - *Reading:* int8 costs 1–3 macro-F1 points against full precision on every set, for a
    smaller CPU-only model.
- **Results, all models (2026-10-11, Kaggle job `notebooks/e18_sentiment`, per-row predictions
  saved; model weights checked against the repo's sha256).** The int8 rows were re-run there on
  CPU and agree with the laptop run within 0.001 macro-F1; speed figures are the laptop's.
  Macro-F1 on rows each model never saw in training (all rows unless a count is given):

  | Model | PhraseBank AllAgree | PhraseBank 50Agree | FiQA test | NWGI test |
  |---|---|---|---|---|
  | Engine, int8 (E5, TFNS only) | 0.865 | 0.685 | 0.573 | 0.406 |
  | Same, full precision | 0.880 | 0.702 | 0.602 | 0.414 |
  | E5 fine-tuned FinBERT | in-sample | in-sample | 0.617 | 0.474 |
  | FinBERT off the shelf | in-sample | in-sample | 0.562 | 0.554 |
  | E7 distilled bge-small | in-sample | 0.637 (1,385) | 0.676 (149) | 0.541 |
  | E7 DeBERTa teacher | in-sample | 0.657 (1,385) | 0.705 (149) | 0.574 |
  | E10 shared encoder | in-sample | 0.652 (1,385) | 0.641 (149) | 0.558 |

  Paired bootstrap, macro-F1 difference on the same clean rows (`data/outputs/e18_paired.json`):
  - Full precision vs int8: +0.016 [0.010, 0.023] on 50Agree, +0.015 [0.008, 0.023] on AllAgree,
    +0.008 [0.001, 0.015] on NWGI (all **higher**); FiQA +0.029 [−0.021, 0.082], a tie.
  - Engine vs FinBERT off the shelf: FiQA tie (+0.011 [−0.080, 0.097]); NWGI **lower**
    (−0.148 [−0.162, −0.133]).
  - Engine vs the E7 distilled model, same size: FiQA **lower** (−0.106 [−0.190, −0.015]), NWGI
    **lower** (−0.135 [−0.146, −0.123]). Engine vs teacher: lower on both.
  - *Reading:* the engine model, trained on tweets only, transfers worse to news text than the
    models that also saw PhraseBank and FiQA training rows. These test sets are not used to pick
    a new engine model: any switch is a new experiment, chosen on a dev set fixed in advance.

## E20. Choosing the engine sentiment model on news as well as tweets (fixed 2026-10-11, before any dev or new test row is scored)

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

## E21. One sentiment model for tweets and news (fixed 2026-10-11, before any training, labelling or scoring)

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

## M1. Module B sensitivity checks (fixed 2026-10-11, before the checks are written or run)

Consistency checks on the real book (`data/reference/book.csv`) and market data; they change
no result. Expected behaviour, fixed now. If a check fails, the formula is investigated; the
expectation is not changed to fit the result.

1. **No shock, no loss:** no obligor, `multiple = 0` → loss 0, CET1 after = before, every
   rating, spread and PD unchanged.
2. **Bigger market shock, bigger loss:** for each episode (GFC 2008, euro 2011, COVID 2020) and
   `multiple` 0, 0.5, 1, 1.5, 2 (no obligor): total loss does not fall and CET1 after does not
   rise as the multiple grows; Baa spread after and z grow with it.
3. **`multiple = 0` removes the episode:** with a downgrade, Aaa and Baa stay at their book-date
   levels, z = 0 and USD/INR does not move; only the obligor's rating changes, and every other
   name's spread moves by its contagion term only (≥ 0).
4. **More notches, more credit risk:** one obligor, `multiple = 0`, 0 to 6 notches: its PD after,
   its revolver ECL, the bank's total loss and RWA after do not fall as notches grow.
5. **Stressed PD:** Vasicek PD rises with z, and is at least the normal one-year PD in the
   scenario (z > 0).
6. **Default:** `notches = 99` → rating D, revolver in stage 3 with ECL = LGD × EAD, bond worth
   notional × (1 − LGD).
7. **Reverse stress:** at the multiple it returns, CET1 is at the 8% floor (within 0.05 point),
   and at 90% of that multiple CET1 is above the floor.

## E22. Interest-rate shock in Module B (fixed 2026-10-11, before any code change or run)

- **Why:** E14 holds the 10-year Treasury at its book-date level (1.92%), so only credit moves.
  In real crises Treasury yields move too; leaving them out is Module B's largest modelling gap.
  E22 adds a controlled rate shock. It is a stress scenario, not a forecast of rates.
- **Shock (inputs read from `market_history.csv`, FRED `DGS10`; not results):** a **parallel
  shift**: every tenor moves by the 10-year change.

  | Scenario | Dates | UST10Y start → end | Δ (pp) |
  |---|---|---|---|
  | GFC 2008 | 15 Aug – 15 Dec 2008 | 3.84 → 2.53 | −1.31 |
  | Euro 2011 | 1 Jul – 4 Oct 2011 | 3.22 → 1.81 | −1.41 |
  | COVID 2020 | 14 Feb – 23 Mar 2020 | 1.59 → 0.76 | −0.83 |
  | Rates up (controlled) | — | — | +2.00 (Basel IRRBB parallel-up shock for USD); no credit, equity or FX move |

  In the three crises yields fell (a flight to safety), so fixed-rate assets gain on rates.
- **Formulas** (yields and spreads stored in % points, divided by 100; `k` = the episode
  multiple, as now):
  - Rate after: `r1 = max(r0 + k·Δr, 0)`, with r0 = 1.92% at 31 Dec 2019. Negative US yields
    are floored at 0 (stated).
  - **Bonds (implemented repricing):** fixed coupon `c = r0 + s0` (par at the book date, as
    now); value after `= N · P(c, r1 + s1, T)`, `P` = the existing annual-coupon `bond_price`.
    Rates up → value down.
  - **Pay-fixed interest rate swaps (new: mark-to-market added):** fixed rate `K = r0` (at par on
    the book date, so value before = 0); value after `= N · (r1 − K) · A(r1, T)`, `A` = the
    existing annuity. Rates up → the bank (paying fixed) gains. CVA stays as now: SA-CCR
    exposure from the notional only, so CVA does **not** respond to the swap's new value
    (stated limitation).
  - **CDS:** the risky annuity is discounted at `r1 + s1` instead of `r0 + s1` (small effect).
  - **Revolvers:** treated as floating-rate loans: no rate repricing; IFRS 9 ECL unchanged by
    rates.
  - **Capital:** rate gains and losses on bonds (fair value through OCI) and swaps (trading
    book) flow into CET1 after 25% tax, as credit losses do. Market-risk RWA uses the new bond
    value; its rate weights by maturity band are unchanged.
- **Switch:** `config.rates = "fixed" | "moved"`. `fixed` must reproduce E14 exactly.
- **Checks: pass criteria** (fixed now; a failure means the code is fixed, not the criterion):
  1. `rates = "fixed"` (or Δr = 0) reproduces E14 to the cent for every position, all three
     episodes at `k = 1`, and the stored E14 alert and crisis results.
  2. Direction: for a rate-only shock (no credit, z, FX), every bond loses value when rates rise
     and gains when they fall; every pay-fixed swap does the opposite.
  3. Size: for ±100 bp, each bond's full repricing is within 5% of its duration + convexity
     estimate (`duration_convexity`).
  4. Swaps are worth 0 at the book date (Δr = 0).
  5. Monotonic: rate-only shocks of +0, +100, +200, +300 bp give a total rate loss that does not
     fall.
  6. Floor: no scenario rate goes below 0.
  If all six pass, the engine default becomes `rates = "moved"`. Nothing else decides it.
- **Reported, never used to decide** (E14 and E15 files are kept unchanged; E22 writes new files):
  - Per scenario, `rates` fixed vs moved: loss split into credit and rate parts, CET1 after,
    reverse-stress multiple; plus the +200 bp scenario alone.
  - Test 2 (12 crises) and Test 3 (monthly), re-run with the Treasury move in both the
    prediction and the realised value: mean absolute loss error vs the naive, next to E14/E15.
    Pass rules of those tests are not re-applied to choose anything.
- **Limitations (stated with every result):** one 10-year rate for every tenor (a parallel
  shift; real curves twist); the bank's USD funding is not modelled, so a rate gain or loss on
  assets has no offsetting liability move (no full IRRBB / ALM); revolvers assumed floating;
  swap CVA ignores mark-to-market; no basis or curve risk; market-risk capital for swaps and CDS
  still not modelled; book still static at 31 Dec 2019.
- **Results (2026-10-11, run once, `data/outputs/e22_rates.json`; E14/E15 files unchanged, hashes
  checked).** All six checks pass (`tests/test_module_b_rates.py`, 13 tests, including the
  exact reproduction of the stored E14 alert and crisis results), so the engine default is now
  `rates = "moved"` (`config/default.toml`).

  | Scenario (× 1, no alert) | Δ UST10Y | Loss, rates fixed → moved | Rate part (bonds / swaps) | CET1 after, fixed → moved |
  |---|---|---|---|---|
  | GFC 2008 | −131 bp | $271.5m → $107.2m | −$164.3m (−$192.8m / +$28.6m) | 4.37% → 8.91% |
  | Euro 2011 | −141 bp | $47.8m → −$159.6m (a gain) | −$207.4m (−$238.2m / +$30.9m) | 10.58% → 15.57% |
  | COVID 2020 | −83 bp | $294.0m → $193.8m | −$100.2m (−$118.2m / +$17.9m) | 4.22% → 7.05% |
  | Rates up, controlled | +200 bp | $258.5m (bonds +$298.2m, swaps −$39.7m) | — | 13.00% → 6.18% |

  Reverse stress (multiple of the episode that takes CET1 to 8%), fixed → moved: GFC 0.58 → 1.23,
  euro 2.18 → 5.57, COVID 0.57 → 0.84.
  - *Reading:* in all three crises Treasury yields fell, so the fixed-rate bonds gained and
    offset a third to all of the credit loss; COVID still breaches the 8% floor. A pure
    +200 bp rise, with no credit move, also takes CET1 below the floor (6.18%): the bond book
    is the bank's main interest-rate risk.
  - *Caution:* the bank's USD funding is not modelled, so these rate gains are not offset by
    liabilities, as they would partly be in a real bank. The crisis results are therefore
    less severe than a full balance-sheet view; the rates-up result is the clearer use of E22.
  - *Reported only:* Test 2 (12 crises) loss error with rates moved: similar crises $31.0m vs
    naive $28.2m (E14: $45.7m vs $47.0m). Test 3 (345 months): model $42.9m vs naive $41.4m,
    ratio 1.04, direction 43% vs 52%, Diebold–Mariano p = 0.015 (E15: 1.00, p = 0.93). Errors
    fall in Test 2 for both methods; neither forecast beats the naive. Nothing was chosen on
    these numbers.

## E22b. Does a credit alert itself carry information? (fixed 2026-10-11, before any alert is counted or any return computed)

- **Why:** Test 1 (E14) compared "downgrade + full COVID episode" with "the same downgrade + the
  real 20-day market moves", so both sides shared the assumed downgrade and the gap was the
  episode. E22b asks only about the alert: after a gated credit alert, does the company's stock
  do worse than usual, and does a larger predicted alert loss go with a worse reaction?
- **Prediction (from the code, checked in M1 test 3):** `revalue(book, ticker, notches, episode,
  multiple=0)`. With `multiple = 0`: Aaa, Baa, z, USD/INR and the Treasury stay at their
  31 Dec 2019 levels; the obligor moves `notches` down (`parse_notches`, 1 if none found, D for
  Default) and its spread widens to the new rating's spread; every other name widens by that
  widening × its return correlation (contagion, ≥ 0); PDs: the obligor gets its new rating's
  PD, all others keep theirs (the Vasicek PD at z = 0 is below the normal PD, so the max keeps
  the normal one). The loss is the revalued change in the obligor's revolver ECL (stage changes
  at 3+ notches or a fall below BBB−), its bonds, every name's bonds and CDS through contagion,
  and swap CVA. The book and ratings are those of 31 Dec 2019 for every alert, so the USD size
  is illustrative; the predicted direction is always "loss".
- **Alerts:**
  - **Primary sample, history:** the E17 history set (`data/cache/history_signals.csv.gz`,
    FNSPID news headlines, 1 Jan 2010 – 8 Apr 2020), rows kept by the E17 relevance rules with
    role `subject`. Event = the E13 credit rules (`credit_event`; the engine's model never
    overrides them); impact = `impact_formula` with the stored E17 sentiment (full-precision
    scores, as E17) and source `news`; then `gate(…, trigger = 7)` unchanged. These years were
    never used to write the credit rules (E13 used the 2020 replay window), so they are out of
    sample for the rules. No parameter is fitted in E22b.
  - **Secondary, already seen:** the 4 alerts of the replay window (9 Apr – 10 Jun 2020), the
    E14 Test 1 alerts, scored the same way and reported apart.
- **Dates and window:** the publication timestamp's UTC calendar date is the alert date. Start =
  the close of the last trading day strictly before that date (no part of the reaction is
  missed, nothing after is used to place the start). End = the close 20 trading days after the
  start. Prices: `data/prices/prices_history.csv` (Yahoo closes, adjusted).
- **Observed outcome (a proxy, not a credit loss):** the obligor's market-adjusted 20-day
  return: log(close_end / close_start) − the same for `^GSPC`. Company credit spreads and CDS
  quotes are not available free, so the stock is used: equity falls when credit quality falls,
  but it also moves on everything else. This measures the equity reaction, never the credit
  loss, and no USD loss error is computed (the two are in different units).
- **Independence and exclusions:** per company, an alert within 20 trading days of an earlier
  counted alert of the same company is dropped (counted, not scored), so windows of one company
  never overlap. Alerts on different companies may overlap in time (stated). Alerts whose start
  or end close is missing are dropped and counted. A company not in the book cannot be revalued
  and is dropped and counted.
- **Baseline (naive):** the same company's market-adjusted 20-day return from every trading day
  in 2010-01-01 – 2020-04-08 (start-day windows defined the same way), with no alert
  information. Its share of negative windows per company is the base rate.
- **Metrics:**
  1. **Direction (primary):** k = alerts with a negative market-adjusted return, n = alerts
     scored. p0 = the mean of the alerts' company base rates. One-sided exact binomial test of
     k out of n against p0.
  2. Mean market-adjusted return of the alerts with a 95% bootstrap interval (10,000 resamples,
     seed 0), next to the baseline mean (reported).
  3. Spearman correlation between the predicted alert loss (USD) and the market-adjusted return
     across alerts, with its p-value (reported; expected negative if size matters).
- **Pass rule (fixed now):** if n ≥ 20, E22b passes when the one-sided binomial p < 0.05 (alerts
  are followed by a negative equity reaction more often than the company's base rate). If
  n < 20, there is no pass or fail: the result is reported as descriptive and underpowered. The
  Spearman correlation and the secondary sample never decide anything.
- **Outputs:** `data/outputs/e22b_alerts.csv` (one row per alert: date, ticker, text, notches,
  impact, gate reason, start/end dates, returns, predicted loss, kept or why dropped),
  `data/outputs/e22b_results.json`; runner `src/riskengine/module_b/alert_check.py`. E14, E15
  and E22 files are not touched.
- **Limitations:** stock returns are a noisy proxy for credit stress; market-adjusted, not
  beta-adjusted; the alert rules were written for 2020 social and news text and run here on
  news headlines only; the sentiment behind the impact is the full-precision E17 score, not
  the engine's int8; one book and one rating set (31 Dec 2019) for every year; FNSPID's date
  stamps are dates, so same-day timing within a day is ignored.
- **Results (2026-10-11, run once; `data/outputs/e22b_alerts.csv`, `e22b_results.json`; E14,
  E15 and E22 files unchanged, hashes checked).**
  - **Primary (history, 2010 – 8 Apr 2020):** 79 alerts fired; 39 dropped as within 20 trading
    days of an earlier alert on the same company; **n = 40** scored. Negative market-adjusted
    20-day return in 22 of 40 (55%) vs the companies' base rate 47.8%; one-sided binomial
    p = 0.23. **Verdict: FAIL** (p ≥ 0.05).
  - Mean market-adjusted return −1.7% [−4.1%, +0.6%] vs the baseline mean +0.3%. Spearman
    between the predicted alert loss and the return: +0.05 (p = 0.77); no link between Module
    B's predicted size and the reaction.
  - Secondary (replay window, seen): 4 fired, 1 overlapping, n = 3, 2 negative; reported only.
  - *Looked at afterwards (description only; the frozen E13 rules are not changed):* 23 of the
    40 scored alerts are on JPM and BAC, and most of those are not about the bank's own credit:
    commercial-mortgage securitisations named after the bank ("Fitch Downgrades J.P. Morgan
    2001-CIBC3"), an affirmation ("Fitch Affirms Bank of America's IDR"), a counterparty
    ("Bank of America Says MBIA Defaulted"). One NFLX alert is an upgrade (B+ to BB−) read as a
    downgrade. The gate's precision on history news is low; the gold set (E3) is where it
    would be measured. The failure is at least partly a trigger-precision problem, not only a
    pricing one.

## E23. Credit-event gold set for the frozen alert gate (fixed 2026-10-11, before any sample is drawn or label seen)

- **Why:** E22b failed, and reading its alerts showed many that are not credit events for the
  named company (bank-named securitisations, affirmations, a counterparty, an upgrade). The gate
  (E12, E13) has never been measured against human labels. E23 measures it; it extends E3 (whose
  300 class-labelling headlines stay as they are and are excluded from this sample).
- **Gate snapshot:** before export, sha256 of `src/riskengine/gate.py`, `nlp/events.py`,
  `nlp/impact.py`, `ingest/linking.py`, `ingest/relevance.py` and `impact_trigger` are written
  to `data/outputs/e23_gate_snapshot.json`. The gate is not changed before it is evaluated.
- **Unit:** a (headline, company) pair, as the gate sees it.
- **Pool:** the E22b universe: E17 history headlines 1 Jan 2010 – 8 Apr 2020 kept by the E17
  relevance rules with role `subject`, scored as in E22b (E13 credit rules, impact formula with
  the stored sentiment, source news, gate with trigger 7), plus the replay-window signals
  (9 Apr – 10 Jun 2020, news and tweets) through the same gate. Pairs whose headline is one of
  the 300 E3 headlines are removed.
- **Strata (seed 23):**
  - **S1, gate fired:** every fired pair (a census).
  - **S2, credit rule but gate silent:** 60 at random from pairs the E13 rules call a credit
    class where the gate did not fire.
  - **S3, hard negatives:** 60 at random from pairs the rules do *not* call credit whose
    headline contains a credit-flavoured word or pattern: affirm, upgrade, outlook, rating,
    Moody's, Fitch, S&P, DBRS, securiti, CMBS, mortgage trust, "Classes", a deal code such as
    `2006-FL2`, bankrupt, default, liquidity, refinanc, covenant, waiver, junk, bond, debt,
    lender, counterparty.
  - **S4, random:** 40 at random from all other pairs (to find credit events the rules miss).
  Each stratum's population size is stored, so recall can be weighted back to the pool.
- **Blind file:** `data/gold/credit/credit_to_label.csv` shows headline id, headline, date and
  target company only; no event class, impact, gate result or stratum. System outputs sit in
  `data/gold/credit/sample_meta.csv`. AI suggestions, if ever added, go in a separate
  `ai_suggestion` column and never count as labels; rows stay `unreviewed` until a person
  reviews them.
- **Labels:** defined in `data/gold/credit/GUIDE.md` (written before labelling): target match
  correct, credit-event relevant, subtype (default, downgrade, upgrade, covenant, liquidity,
  refinancing, affirmation, counterparty, securitization, other), materiality, whether the gate
  should fire, rationale, adjudication status. Only rows with status `reviewed` or
  `adjudicated` are scored; `unreviewed` and `excluded` are counted.
- **Metrics (once labels are reviewed, scored once):**
  1. **Gate:** precision on S1 with a Wilson 95% interval; recall weighted by stratum
     population sizes, with a stratified bootstrap interval (1,000 resamples, seed 0).
  2. **Credit-relevance rules (E13):** precision and weighted recall of "rules say credit"
     against "credit-event relevant".
  3. **Target matching:** precision of the linked (headline, company) pairs, overall and by
     stratum. Recall of matching is **not measurable** here: only linked pairs are sampled.
  4. Subtype confusion matrix: rule class (default / downgrade / covenant / none) vs human
     subtype; the gate's false-positive share by human subtype.
  5. Splits by source (history news; replay news; replay tweets) and period (2010–2016;
     2017 – 8 Apr 2020; replay window), shown only with n ≥ 10, all with n.
- **No pass rule:** E23 measures; it does not decide. Any change to the gate is a later,
  separately pre-registered experiment, tuned on a part of the data that E23 does not use for
  its final numbers. A small sample is reported as such.
- **Amendment (2026-10-11, after the export, before any label exists; the author's decision):**
  AI-made labels count as the ground truth. Rows labelled by AI carry `adjudication_status =
  ai_labelled` (or `reviewed` if the author then checked them) and are scored. Every E23 result
  states who labelled it (the models' names, and whether a person reviewed the rows), as E21
  did. The AI tools only label this test set; none is part of the engine.

## E24. Event classes on news the engine was not built on (fixed 2026-10-11, approved v4, before any sample is drawn or label made)

- **Weaknesses addressed (from my own error analysis):** W1 "Product news" swallows any company
  news; W3 earnings news without $/EPS is missed; W4 low-confidence model guesses still get an
  event class; W5 news and tweets are measured together. **Deferred:** W2 (war and sanctions read
  as Macroeconomic) needs a world-news source and goes to a separate experiment; E24 does not
  change Geopolitical or Macroeconomic. E13 credit rules are unchanged. The other project's
  gold-2 is a seen diagnostic only and decides nothing.
- **Sets:**
  - **Dev:** E3's 300 FNSPID headlines (2019 – 10 Jun 2020), split before labelling into D1 (150,
    rule design and optional temperature) and D2 (150, candidate and threshold choice). The split
    is **prediction-stratified** (by E3's design half and the engine's predicted class, seed 24),
    not true-class stratified: no labels exist at split time. Real label support of D1 and D2 is
    reported after annotation, before any performance number.
  - **Final news (200):** FNSPID 2017-01-01 – 2018-12-31, E17 history kept by the relevance rules,
    role subject, unique headlines. An **earlier-period evaluation**, not a future test: the dev
    period and the model's training data are later.
  - **Final tweets (100):** equinxx 2021–22 unique tweets. A **later-period, cross-source
    evaluation**. News and tweets are never pooled.
  - **Selection:** simple random sample, seed 24, from a stably sorted frame (published time,
    text); no class, prediction or label is used. Frame bias (company-linked news, mostly Other)
    is reported.
  - **Excluded** (exact normalised match or cosine ≥ 0.9): TFN topic train+valid, TFNS
    train+valid, E3, the E23 sample, E22b alerts, all E17 relevance samples, the E21 tweet
    sample, FinEntity, the other project's gold-1, gold-2 and entity check, and near-duplicates
    within the final sample.
- **Taxonomy and metric:** the existing 12 classes (Default, Credit downgrade, Covenant breach,
  Geopolitical, Macroeconomic, M&A, Product news, Earnings, Analyst rating, Legal/Regulatory,
  Management change, Other), fixed now. "Product news" is labelled under a new narrow definition
  (a product event: launch, unveiling, release, recall, product deal), written into the guide
  before annotation; E16/TFN-topic results stay as they are. **Primary metric: macro-F1 over all
  12 classes, `zero_division = 0`, never re-selected by support.** All candidates, **including
  A**, are scored against the same new-definition labels; A's predictions are not changed.
- **Annotation (AI-assisted, human-adjudicated; allowed by the organisers' "AI assistance … with
  full honesty", no rule bans AI for labelling; no AI tool is part of the engine):**
  - Labeller A = ChatGPT, labeller B = Gemini. Recorded: exact model/version, date, full prompt
    (with the guide), settings, interface. Each sees only the blind file and the guide; neither
    sees the other's output or any engine prediction. Their outputs are stored apart, hashed.
  - **Primary-event rule:** the label is the main news fact the headline reports. Precedence only
    for documented overlaps: agency credit action + stock rating → credit class; earnings +
    guidance or dividend → Earnings; regulator decision on a deal → Legal/Regulatory if the
    decision is the news, else M&A; recall + lawsuit → Legal/Regulatory, recall alone → Product
    news; an executive leaving amid a probe → Management change if the departure is the news.
  - A ≠ B, or either marks it ambiguous → a person adjudicates (`human_adjudicated`). A = B →
    `ai_agreed`; a 20% random audit of agreed rows (seed 24) by a person, whose label replaces
    the AI one (`ai_agreed_human_audited`); the audit correction rate is reported with a Wilson
    95% interval. Undecidable → `excluded` (counted). Cohen's kappa of A vs B is reported. The
    labels are described as AI-assisted, not independent human gold.
- **Candidates:** A = current engine; B = A + narrow Product-news mapping (other company news →
  Other); C = B + an earnings precedence rule; D = C + one global abstention threshold. Order:
  E13 credit rules → earnings rule → product rule → model; abstention applies only to model
  predictions (below threshold → Other, counted as a prediction, not dropped). Rules are written
  from these definitions on D1 only. No code, keyword list, weights or labels of the other
  project are used.
- **Calibration and threshold:** temperature scaling (Guo et al., 2017) is fitted on D1 only, and
  only if D1 has at least 5 classes with ≥ 10 labelled rows; else raw probabilities. Threshold
  grid fixed: {0.30, 0.35, …, 0.70}, chosen on D2 only (ties → lowest); a boundary choice is
  reported as a limitation, the grid is never widened. No class-specific tuning for a class with
  < 10 D1 rows.
- **Choice (D2 only):** highest 12-class macro-F1, provided no class with ≥ 10 D2 rows loses more
  than 0.05 F1 against A; within 0.01 → the simpler candidate. The chosen candidate, its rules,
  threshold and temperature are frozen (hashes) before the final set is labelled.
- **Final evaluation (once):** paired bootstrap on the final rows: the same resampled row
  indices for A and the chosen candidate, 1,000 resamples, seed 0, 95% percentile interval of
  Δ = macro-F1(chosen) − macro-F1(A). **Every resample uses the fixed 12-class list with
  `zero_division = 0`, even when a class is absent from that resample.**
  - News: pass if the lower bound > 0; fail if the upper bound < 0; else, or if the interval is
    wider than 0.20, **inconclusive**.
  - Tweets (non-inferiority, margin 0.05): pass if the lower bound > −0.05; fail if the upper
    bound < −0.05; else, or if wider than 0.20, **inconclusive**.
  - The engine switches (`events_rules = on`) only if news **and** tweets pass; otherwise A
    stays. No retest with a larger sample, no change of sample or threshold after the result.
- **Reported, news and tweets apart:** rule coverage by rule, model coverage, abstention rate,
  per-class precision, recall, F1 and support, confusion matrices. Classes with < 5 or no support
  are named; no reliable per-class claim is made for them. TFN-topic and gold-2 scores are
  reported only.
- **Safeguards:** deterministic exclusion (two separate runs must give the same sha256 of the
  blind file, or no freeze); provenance per row (source dataset, file and its sha256, original
  id or timestamp, frame rule); final labels are read only when `e24_frozen.json` exists and the
  blind-file hash matches, and only once; read-only backups of frozen artefacts in
  `private_notes/e24_frozen/` with a sha256 manifest, checked before evaluation. Nothing existing
  is changed; outputs are `e24_*`. Licences: FNSPID CC BY-NC 4.0, equinxx CC0 (tweet text under
  platform terms, gitignored), TFN MIT; Guo et al. (2017) cited.

## E4. Efficiency (fixed 2026-10-06)

Next to every accuracy figure: CPU throughput (texts per second, batch 64, after a warm-up
batch), model size on disk, and peak resident memory of the scoring process.
