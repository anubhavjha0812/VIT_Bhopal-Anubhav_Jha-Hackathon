# Data metadata

What is in `data/`, where it came from, who labelled it, why it exists and where the project uses it.
Snapshot: 2026-10-16. Row counts were read from the files. Decision numbers (E1, E23, …) refer to `docs/DECISIONS.md`.

## How to read this file

1. **Section 1** is the map of the folder.
2. **Section 2** covers every dataset labelled (fully or partly) by AI. Read it first.
3. **Sections 3–6** cover everything else, folder by folder.
4. **Section 7** lists known issues.

### Label trust levels

| Tag | Meaning |
|---|---|
| **HUMAN** | Labels written by people (public datasets, as published). |
| **AI + HUMAN REVIEW** | AI drafted the labels, the author reviewed and finalised them. |
| **AI ONLY** | AI labels, no human check. Not independent gold. Must be reported that way. |
| **AI (PUBLIC)** | A public dataset whose labels were written by an AI model. |
| **MODEL-SCORED** | Our own models' predictions or scores. Not labels. |
| **UNLABELLED** | Blank input waiting for labels. |

## 1. Folder map

| Folder | Size | What it holds | Main role |
|---|---:|---|---|
| `gold/` | 0.3 MB | Blank inputs and guides for the hand-checked test sets (credit gold, E24 events gold) | Measure the event classifier and the credit alert gate |
| `annotation/` | 0.1 MB | 300 tweets for sentiment testing, with AI-assisted labels | Fresh tweet sentiment test (E21) |
| `relevance/` | 0.1 MB | Samples where LLMs judged "is this headline about the company?" | Check the company-linking rules (E17) |
| `eval/` | 7.9 MB | Public benchmark sets | Sentiment and event tests (E1, E2, E18, E20, E21) |
| `cache/` | 24 MB | Training data for E21, plus scored headlines/tweets | Speed (no re-scoring), Module A history |
| `raw/` | 5.7 GB | Source news, tweets and SEC filings | Input to every ingest step |
| `samples/` | 1.2 MB | Small committed slice of news (replay window) | Lets a clean clone run the demo |
| `prices/` | 4.7 MB | Daily closing prices and volume | Module A returns, impact model, backtests |
| `reference/` | 0.5 MB | Company list, loan book, ratings, SEC debt, market history | Module B (credit and capital risk) |
| `live/` | 0.06 MB | Live GDELT news signals | Live page of the dashboard |
| `models/` | 1.1 MB | Trained impact models | Impact score |
| `outputs/` | 2.5 MB | Results, predictions, backtests, alerts | Experiment register, dashboard, reports |

`private_notes/` (outside `data/`, git-ignored) holds the AI labels themselves. See 2.1.

---

## 2. AI-labelled datasets

Four of our own datasets have AI labels (2.1–2.4). One public dataset has AI labels (2.6). The 300 final holdout rows (2.5) are still unlabelled.

| # | Dataset | Rows | Labelled by | Trust | Where the labels live |
|---|---|---:|---|---|---|
| 2.1 | Credit-event gold (E23) | 243 | ChatGPT + Claude, resolved by ChatGPT | AI ONLY | `private_notes/…/reference_v6/credit_labelled.csv` |
| 2.2 | Events dev gold, D1 + D2 (E24) | 300 | ChatGPT + Claude, resolved by ChatGPT | AI ONLY | `private_notes/…/reference_v6/development/` |
| 2.3 | Tweet sentiment test (E21) | 300 | ChatGPT + Gemini, Claude settled, author reviewed | AI + HUMAN REVIEW | `annotation/e21_tweets_final_labels.csv` |
| 2.4 | Company-relevance samples (E17) | 300 + 96 | Open LLMs, author settled unsure rows | AI + HUMAN REVIEW | `relevance/labels.csv`, `audit_to_label_completed.csv` |
| 2.5 | Events final holdout (E24) | 300 | none yet | UNLABELLED | `gold/e24/final_*_to_label.csv` |
| 2.6 | NWGI sentiment (public) | 20,231 | GPT model | AI (PUBLIC) | `eval/nwgi_*.csv` |

*`private_notes/…` = `private_notes/annotation_uploads/task1/`. In this file "ChatGPT" is the labeller the annotation files call "Codex".*

### 2.1 Credit-event gold set (E23)

| | |
|---|---|
| **Files** | Blank input: `gold/credit/credit_to_label.csv`. Rules: `gold/credit/GUIDE.md`. Sampling info: `gold/credit/sample_info.json`, `gold/credit/sample_meta.csv`. |
| **Rows** | 243 headline–company pairs. |
| **Sampling** | Seed 23. Four strata: S1 gate fired (83, all of them), S2 credit rule but gate silent (60), S3 hard negatives (60), S4 random rest (40). 211 news, 32 tweets. The 300 E3 headlines are excluded. |
| **Columns filled by labellers** | `target_company_match_correct`, `credit_event_relevant`, `event_subtype` (10 values), `materiality` (low/medium/high), `existing_gate_should_fire`, `annotator_rationale`, `adjudication_status`. |
| **Method** | Three AI passes: ChatGPT (research-backed), Claude (first pass, then a best pass with web research), then a final resolution by ChatGPT after reviewing Claude's 43 differing rows. All use the same guide. Web research was allowed for this pass (an amendment to the guide's "headline only" rule). |
| **Current reference** | `reference_v6`: 46 alerts should fire, 197 should not. Subtypes: other 130, downgrade 46, counterparty 27, securitization 26, refinancing 9, affirmation 3, upgrade 2. 11 rows keep an uncertainty flag (`?`) and still have a label. |
| **Agreement** | Independent passes (Claude vs ChatGPT v2): kappa 0.92–0.99 per field, alert-fire field 1.00. |
| **Trust** | AI ONLY. No human or expert validation. Say "AI-resolved reference", never "expert gold". |
| **Purpose** | Measure the frozen credit alert gate (E12, E13): how many alerts it fires correctly (precision) and how many real credit events it misses (recall). |
| **Used in** | Module B trigger gate (`src/riskengine/gate.py`, `nlp/events.py`). E22b showed many alerts were not real credit events, so E23 measures this against labels. |
| **Caveat** | `adjudication_status` is `ai_labelled` on every row. Nothing is marked human-reviewed. |

### 2.2 Events development gold set, D1 + D2 (E24)

| | |
|---|---|
| **Files** | Blank input: `gold/gold_to_label.csv` (300 FNSPID headlines, 2019 – 10 Jun 2020, from E3). Split: `gold/e24/e3_split.csv`. Rules: `gold/e24/GUIDE.md` and `gold/README.md`. Prompts: `gold/e24/prompts/`. |
| **Rows** | 300 = D1 153 + D2 147 (the docs say 150/150; the stored split is 153/147 and was kept). |
| **Sampling** | E3, seed 7: half from headlines with credit-related words, half at random. Split into D1/D2 by the engine's predicted class (seed 24). |
| **Columns** | Input: `doc_id, date, tickers, headline, label` (label blank). AI output: `item_id, label, ambiguous, rationale`. |
| **Classes** | 12: Default, Credit downgrade, Covenant breach, Geopolitical, Macroeconomic, M&A, Product news, Earnings, Analyst rating, Legal/Regulatory, Management change, Other. |
| **Method** | Same three-pass route as 2.1: ChatGPT, Claude (with research), final resolution by ChatGPT. One class per headline, an ambiguity flag, and a rationale of at most 20 words. The E24 guide named Gemini as the second labeller; Claude took that place, and this is logged as a deviation. |
| **Current reference** | `reference_v6`. Other 194, Analyst rating 73, Macroeconomic 9, Earnings 8, Product news 6, M&A 4, Legal/Regulatory 4, Geopolitical 1, Management change 1. 23 rows flagged ambiguous. |
| **Agreement** | Claude vs ChatGPT (independent): kappa 0.77 on D1, 0.76 on D2. |
| **Important gap** | There are **zero** Default, Credit downgrade or Covenant breach examples in the development labels. Recall for those classes cannot be estimated from this set. |
| **First result** | The unchanged classifier scores 0.330 macro-F1 and 46.0% accuracy (138 of 300) on this reference. Always predicting "Other" gives 64.7% accuracy but only 0.066 macro-F1. See `outputs/e24_ai_dev_baseline_v6.md`. |
| **Trust** | AI ONLY. The result is a *development diagnostic*, not a human-validated test. |
| **Purpose** | D1 is for designing event rules. D2 is for choosing the candidate and its thresholds. Then the candidate is frozen. |
| **Used in** | Event classifier (`src/riskengine/nlp/events.py`), E24 evaluation gate. |

### 2.3 Tweet sentiment test set (E21)

| | |
|---|---|
| **Files** | `annotation/e21_tweets_to_label.csv` (blank input) and `annotation/e21_tweets_final_labels.csv` (labelled). |
| **Rows** | 300 tweets: 30 each for TSLA, AAPL, AMZN, META, AMD, NFLX, GOOGL, DIS, BA, KO. Source: equinxx tweets, 30 Sep 2021 – 29 Sep 2022. Sampled with seed 21. |
| **Columns** | `row_id, ticker, text, label, notes` |
| **Labels** | positive 104, neutral 78, negative 69, skip 49 (not about the company, spam, not English). 251 rows are scored. |
| **Method** | ChatGPT and Gemini labelled independently. Claude settled the rows where they disagreed. The author then reviewed and finalised. The original plan was one blind human annotator; this was changed (amendment in E21, 2026-10-11). |
| **Trust** | AI + HUMAN REVIEW. Reported as a **secondary** test. No claim rests on it alone. |
| **Purpose** | Test sentiment on tweets from a later period than the training data. |
| **Used in** | E21 confirmation. Engine model A scored 0.452 macro-F1 on it and the E21 model 0.591, but A stayed because FinEntity (the primary test) said lower. Results: `outputs/e21_results.json`. |

### 2.4 Company-relevance samples (E17)

| | |
|---|---|
| **Files** | `relevance/to_label.csv` (300), `relevance/labels.csv` (300 LLM labels), `relevance/sample_meta.csv`, `relevance/labellers.json`, `relevance/audit_to_label.csv` and `audit_to_label_completed.csv` (96), `relevance/audit_meta.csv`. |
| **Rows** | 300 headline–company pairs: 200 where the company is linked by name (set S1), 100 linked by its own ticker symbol (S2). Plus 96 audit rows. |
| **Question** | "Is this headline about the company?" (Y / N / unsure). |
| **Method** | Three open-weight LLMs voted: Qwen3-8B, Phi-4-mini and OLMo2-7B. OLMo2 said "yes" for 0% of rows, so it was dropped; the other two were used (agreement 0.78). Result: Y 172, unsure 66, N 62. The 66 "unsure" rows were settled by the author (58 N, 8 Y). A 96-row audit file was completed (N 71, Y 25); on 30 audited rows agreement with the LLMs was 23 of 30 (76.7%). |
| **Trust** | AI + HUMAN REVIEW. |
| **Purpose** | Measure how often the company-linking rules keep a headline that is not really about the company. |
| **Used in** | `src/riskengine/ingest/relevance.py`. Rule precision went from 0.645 to 0.849 (95% interval 0.76–0.91). Results in `outputs/relevance_results.json`. |

### 2.5 Events final holdout (E24), not labelled yet

| | |
|---|---|
| **Files** | `gold/e24/final_news_to_label.csv` (200), `gold/e24/final_tweets_to_label.csv` (100), `gold/e24/final_provenance.csv` (300). |
| **Sampling** | Seed 24, simple random sample. News: FNSPID 2017–2018 (earlier period). Tweets: equinxx 2021–22 (later period, different source). News and tweets are never pooled. |
| **Status** | **UNLABELLED on purpose.** They are labelled only after the candidate is frozen, then scored once. |
| **Purpose** | The final, untouched test of the event classifier. |
| **Used in** | E24 final evaluation. Sample record: `outputs/e24_final_sample.json`. |

### 2.6 NWGI sentiment (public set, AI labels)

| | |
|---|---|
| **Files** | `eval/nwgi_train.csv` (16,184), `eval/nwgi_test.csv` (4,047). |
| **Labels** | Written by a GPT model, not by people. |
| **Trust** | AI (PUBLIC). |
| **Used in** | The test file is reported in E18 only as a check against other published results, never as human-verified accuracy. The train file is **not** used for training (E21). |

---

## 3. Human-labelled public benchmarks, `eval/` and `cache/e21_train/`

All labels written by people, as published. Never used to train a model that is also tested on them, except where marked.

| File | Rows | Source | Purpose | Used in |
|---|---:|---|---|---|
| `eval/sent_valid.csv` | 2,388 | Twitter Financial News Sentiment (TFNS), MIT | Main tweet sentiment test | E1, E20, E21 |
| `eval/topic_valid.csv` | 4,117 | Twitter Financial News Topic, MIT | Event-class test (4 mapped classes) | E2 |
| `eval/phrasebank_50Agree.csv` | 4,846 | Financial PhraseBank v1.0 | News sentiment test (50% annotator agreement) | E18 |
| `eval/phrasebank_75Agree.csv` | 3,453 | Financial PhraseBank | Training data | E7, E21 |
| `eval/phrasebank_AllAgree.csv` | 2,264 | Financial PhraseBank | Sentiment test (all annotators agree) | E18 |
| `eval/fiqa_test.csv` | 150 | FiQA-2018 | Test (E18); later folded into training | E18, E21 |
| `eval/fiqa_train.csv` / `fiqa_validation.csv` | 961 / 102 | FiQA-2018 | Training | E7, E21 |
| `eval/sentfin_v1.1.csv` | 10,753 | SEntFiN 1.1 | Was the E20 confirmation set. From E21 on: training and news-dev data | E20, E21 |
| `eval/finentity.json` | 979 | FinEntity (ODC-BY), human entity-level labels | **Primary** fresh news test (851 usable paragraphs) | E21 |
| `cache/e21_train/train.csv` | 19,988 | TFNS 8,386 + SEntFiN 7,050 + PhraseBank 3,447 + FiQA 1,105 | Training data for the E21 model, leak-filtered | E21 |
| `cache/e21_train/dev_tweets.csv` | 937 | TFNS dev split | Tweet dev set | E21 |
| `cache/e21_train/dev_news.csv` | 791 | 10% of SEntFiN, by label (seed 21) | News dev set | E21 |

Leak checks for these sets are recorded in `outputs/leak_check.json` and the E18/E21 manifests.

---

## 4. Raw sources, `raw/`, `samples/`, `prices/`

None of these hold labels.

| File | Rows / size | What it is | Purpose | Used in |
|---|---|---|---|---|
| `raw/fnspid/All_external.csv` | 13.1 M rows, 5.7 GB | FNSPID: full financial news archive | Source for all news slices | `ingest/fetch_fnspid.py`, `impact/fresh.py` |
| `raw/fnspid/universe_news.csv.gz` | 21,551 | News for our 15 stocks | Replay window news | `ingest/news.py` |
| `raw/fnspid/universe_news_named.csv.gz` | 284,478 | Same, with name-linked matches | Longer history (E17) | Module A history |
| `raw/fnspid/fresh_news.csv.gz` | 46,014 | News for other S&P 100 stocks, 2019 – 10 Jun 2020 | Fresh test for impact v2 | E8 |
| `raw/akkerman/stock-market-tweets-data.csv` | 923,673 | 2020 stock tweets (gaps: 30 Apr – 2 May, 10 – 27 May) | Replay window tweets | `ingest/tweets.py` |
| `raw/equinxx/stock_tweets.csv` | 80,793 | Stock tweets, 2021–22 | Fresh tweet tests; source of E21 and E24 tweet samples | E17, E21, E24 |
| `raw/equinxx/stock_yfinance_data.csv` | 6,300 | Daily prices that come with the tweet set | Price check | Module A |
| `raw/sec/<TICKER>_facts.json` | 15 files | SEC company facts (XBRL) | Debt and credit-line figures | Module B |
| `raw/sec/<TICKER>_10k.htm` | 8 files | 10-K filings (AAPL, AMD, AMZN, GOOGL, META, MSFT, NFLX, NVDA) | Credit-line and debt wording | Module B |
| `raw/sec/company_tickers.json` | 1 file | SEC ticker–CIK list | Look up company IDs | Module B |
| `samples/docs_window.jsonl.gz` | 14,169 docs | Committed slice of news for the replay window | Lets a clean clone run the demo without the 5.7 GB file | `pipeline.py` |
| `prices/prices_daily.csv` | 8,080 | Daily close and volume, replay window | Module A returns | `ingest/prices.py` |
| `prices/prices_history.csv` | 53,927 | Longer price history | Impact model, shocks, 10-year Module A | E6, E15, E17 |
| `prices/prices_fresh.csv` | 66,284 | Prices for the fresh S&P 100 stocks | Impact v2 fresh test | E8 |
| `prices/prices_tweets.csv` | 4,704 | Prices for the 2021–22 tweet period | Fresh test of Module A | E17 |

The 15 stocks: AAPL, AMD, AMZN, BA, BAC, DIS, GOOGL, JPM, KO, META, MSFT, NFLX, NVDA, TSLA, XOM.
Replay window: 2020-04-09 to 2020-06-10 (43 trading days). Per-company counts are in `docs/data_coverage.md`.

---

## 5. Reference data, `reference/` and `live/`

| File | Rows | What it is | Purpose | Used in |
|---|---:|---|---|---|
| `reference/universe.csv` | 15 | Ticker, name, sector, cashtags, aliases | Link text to companies | `ingest/linking.py` |
| `reference/book.csv` | 85 | Loan and bond book (rating, notional, exposure, loss rate, years) | The portfolio that Module B stresses | `module_b/book.py` |
| `reference/ratings.csv` | 15 | Credit rating per company, with basis and source | Rating input to Module B | `module_b` |
| `reference/sec_debt.csv` | 15 | Short, long and total debt and maturities from SEC data | Debt input to Module B | `module_b/sources.py` |
| `reference/sec_credit_lines.csv` | 8 | Credit lines found in filings | Liquidity input | `module_b/sources.py` |
| `reference/credit_line_candidates.csv` | 23 | Candidate credit-line sentences from filings | Review list for the file above | `module_b/sources.py` |
| `reference/market_history.csv` | 8,491 | Daily Baa and Aaa spreads, 10-year Treasury, VIX, USD/INR, S&P 500 (FRED, yfinance), 1990 on | Shocks and crisis backtest | E14, E15, E22 |
| `live/gdelt_signals.jsonl` | 102 lines | Live GDELT news signals with sentiment, event and impact | Live page | `live.py`, `ingest/gdelt.py` |

---

## 6. Model-scored and generated files

These are our own model outputs. They are **not** labels and must not be used as ground truth.

### `cache/`: scored headlines and tweets

| File | Rows | What it is | Used in |
|---|---:|---|---|
| `headline_scores.csv.gz` | 19,081 | Sentiment, event class and class probabilities per headline | Impact model training |
| `headline_scores_fresh.csv.gz` | 36,319 | Same, for the fresh S&P 100 news | Impact v2 fresh test |
| `history_texts.csv.gz` / `history_scores.csv.gz` | 171,673 each | Long-history headlines and their sentiment (scored on a Tesla T4 GPU) | Module A over 10 years (E17) |
| `history_signals.csv.gz` | 205,010 | Company-linked signals from the long history | Module A history |
| `tweet_texts.csv.gz` / `tweet_scores.csv.gz` | 45,311 each | 2021–22 tweets and their sentiment | Fresh Module A test |
| `tweet_signals.csv.gz` | 50,186 | Company-linked signals from those tweets | Fresh Module A test |
| `*_meta.json` | 2 files | How the scores were made, with a check of int8 vs fp32 | Reproducibility |

### `models/`

| File | What it is | Used in |
|---|---|---|
| `impact_model.joblib` | Learned impact score (v1) | E6 |
| `impact_model_v2.joblib` | Impact score v2 | E8 |

### `outputs/`

| Group | Files | Purpose |
|---|---|---|
| Sentiment and events | `eval_results.json`, `e18_*`, `e20_*`, `e21_*`, `leak_check.json` | Results, per-row predictions, manifests for E1, E2, E18, E20, E21 |
| Impact | `impact_results.json`, `impact_v2_results.json` | Impact model scores |
| Module A | `module_a_history.json`, `module_a_metrics.csv`, `weights_history.csv`, `signals.jsonl.gz` (22,943 replay signals) | Trading-signal results and portfolio weights |
| Module B | `e22_*`, `module_b_*` (the `module_b_*` files are older versions of `e22_*`) | Crisis and monthly backtests, interest-rate shocks |
| Credit gate | `e22b_*`, `e23_gate_snapshot.json` | Alert tests and the hash of the gate code, taken before E23 labelling |
| Relevance | `relevance_results.json` | Company-linking precision and recall |
| Events (E24) | `e24_final_sample.json`, `e24_ai_dev_baseline_v6.*` | Final sample record; baseline run of the unchanged classifier on the AI reference |

---

## 7. Known issues (review later)

Tick a box when you have decided what to do. Nothing here has been changed yet.
Priority: **High** = fix before reporting or freezing. **Med** = fix before submission. **Low** = note only.

| Done | # | Priority | Issue | Suggested action |
|:---:|:---:|:---:|---|---|
| [ ] | 1 | High | **`gold/credit_labelled.csv` is a placeholder.** It has the 300 dev headlines with `label = N` on every row, and a misleading name. It is not the credit gold set. | Rename it or delete it. |
| [ ] | 2 | High | **No AI labels are stored in `data/`.** `gold/credit/credit_to_label.csv` and `gold/gold_to_label.csv` are blank. The labels are in git-ignored `private_notes/`. | Decide which reference version to copy into `data/gold/`, with provenance. |
| [ ] | 3 | High | **All AI references are non-independent.** The final v6 resolution was made by ChatGPT after seeing Claude's answers. | Call it "AI-derived reference", never "expert gold" or "consensus". |
| [ ] | 4 | High | **No Default, Credit downgrade or Covenant breach examples in the dev labels.** Their recall cannot be measured there. | Measure it on the credit set (2.1) or the final holdout. |
| [ ] | 5 | Med | **Several reference versions exist** (v1 – v6, plus Claude's own passes). The current one is `reference_v6`. Older versions are kept for audit. | Cite v6 and its hash in any report. |
| [ ] | 6 | Med | **Uncertainty counts differ by version.** ChatGPT v4: 17. Claude best pass: 66. v5: 19. v6: 34 (11 credit + 23 dev). | Always say which version a count refers to. |
| [ ] | 7 | Med | **D1/D2 are 153/147, not 150/150** as in the E24 text. | Keep as stored; report the real sizes. |
| [ ] | 8 | Low | **Final holdout text was opened once.** The pack README says the blind final texts were read earlier, but no labels exist for them. | Keep them unlabelled until the candidate is frozen. |
| [ ] | 9 | Low | **`outputs/module_b_*` duplicate `outputs/e22_*`** (older versions of the same backtests). | Keep one set, or note which is current. |
