# AI Risk Engine: From News & Social Sentiment to Portfolio and Credit Risk

**Submission for the S&P Global & Crisil Campus Hackathon 2026** (individual entry)

| | |
|---|---|
| **Candidate** | Anubhav Jha |
| **College email** | anubhav.23bai11112@vitbhopal.ac.in |
| **College** | VIT Bhopal |
| **Demo video** | _Coming soon_ |
| **Slides** | _Coming soon_ (`docs/presentation.pdf`) |

## 1. Overview

An AI/NLP risk engine that reads **financial news** and **social media posts** and, for each company and event, produces:

- a **sentiment score** from -1.0 (very negative) to +1.0 (very positive)
- an **event type**, one of 12: Default, Credit downgrade or Covenant breach (rules), or Geopolitical, Macroeconomic, M&A, Product news, Earnings, Analyst rating, Legal/Regulatory, Management change or Other (model, E16)
- an **impact score** from 1 (minor) to 10 (severe)

These signals drive two modules:

- **Module A: Sentiment-driven rebalancer.** Adjusts the weights of a mock index of 15 S&P 100 stocks as the news changes, and shows the weights over time on a dashboard.
- **Module B: Event-driven credit stress test.** When a high-impact event (impact > 7) hits, it stresses a wholesale bank book of loans, bonds and derivatives. It shows the effect before and after, as spread widening, probability of default and expected loss.

Every part is compared against a simple baseline, so the gains are shown in numbers.

**Live news:** open **Live GDELT** (`/live`) in the dashboard to fetch current financial
headlines and score them with the same sentiment, event and impact engine. The backend
starts polling on the first visit and checks every five minutes; no API key is needed.
It queries the last 24 hours through the [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/),
with the latest 15-minute GKG file as a fallback if DOC is unavailable. Both sources are
filtered to company names in our universe and market-risk topics. This is headline analysis;
GDELT timestamps are observation times, not guaranteed publication times.

`GET /live` returns polling status and up to 500 recent scored headlines, with optional
`ticker` and `min_impact` filters. `POST /live/refresh` schedules a check (at least one minute
between attempts). Signals persist in `data/live/gdelt_signals.jsonl` (gitignored), separate
from replay data and evaluation outputs. Network failures are shown in the dashboard while
the last fetched results remain visible. Live news does not automatically trade or change
the historical Module A/B results.

## 2. Architecture & Tech Stack

_TODO: architecture diagram (`docs/architecture.png`)._

Stack: Python 3.12 with ONNX Runtime (int8 models on CPU) and FastAPI for the engine and both
modules; React 19 + TypeScript + Tailwind CSS + ECharts (Vite) for the dashboard; training on
Kaggle GPUs with PyTorch and Hugging Face Transformers; data as CSV/JSONL; managed with
[uv](https://docs.astral.sh/uv/).

## 3. Dataset Used

Only public data is used, with no confidential client data. Small samples and derived outputs
are kept in `data/`; large raw files are downloaded by the scripts named below.

**Text and prices**

| Source | Used for | Period | Licence |
|---|---|---|---|
| [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) news headlines (`All_external.csv`, titles only) | engine input, impact model, fresh impact test | 2009 – 10 Jun 2020 | CC BY-NC 4.0 |
| [Stock tweets](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction) (equinxx) | social input | Oct 2021 – Sep 2022 | CC0 |
| [Stock market tweets](https://huggingface.co/datasets/StephanAkkerman/stock-market-tweets-data) (Akkerman) | social input, replay window | 2020 | CC BY 4.0 |
| Daily prices, S&P 500 and VIX via [yfinance](https://github.com/ranaroussi/yfinance) | event study, Module A backtest | 2008 – 2020 | Yahoo Finance terms (personal / research use) |

**Model training and testing**

| Source | Used for | Licence |
|---|---|---|
| [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | sentiment training (train file), **test** (valid file) | MIT |
| [Twitter Financial News Topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | topic training (train file), **test** (valid file) | MIT |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (`Sentences_75Agree`) | extra sentiment training | CC BY-NC-SA 3.0 |
| [FiQA-2018 sentiment](https://huggingface.co/datasets/pauri32/fiqa-2018) (train + validation) | extra sentiment training | not stated by the mirror (public FiQA 2018 challenge data) |

**Pretrained models:** [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) (MIT),
[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT),
[FinBERT](https://huggingface.co/ProsusAI/finbert) (baseline; no licence stated on the model card).

**Assumptions and data issues found**

- The equinxx tweet file gives AMZN, MSFT and PG the same 4,080 tweets (F and NOC are also
  identical), with 16,314 exact duplicates. Tweets are linked to companies by the cashtags and
  names in their text, not by that column.
- Training texts that also appear in a test set are removed before training (84 in the sentiment
  data, 999 in the topic data; earlier models were trained with them, see
  [DECISIONS.md](docs/DECISIONS.md) E7 and E10).
- FNSPID news for our stocks ends on 10 Jun 2020, so the replay window is 9 Apr – 10 Jun 2020.
  The 2020 tweet file has gaps (30 Apr – 2 May, 10 – 27 May); news covers those days. See
  [data_coverage.md](docs/data_coverage.md).
- Non-commercial licences (FNSPID, PhraseBank) suit this hackathon; commercial use would need
  other data.

## 4. Quickstart

**Runtime:** Python 3.12

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python run_demo.py
```

Or with pip:

```bash
pip install -r requirements.txt
python run_demo.py
```

The demo replays the stored engine signals for the window and runs Module A; `--rescore`
re-scores the sample with the configured models.

**Dashboard** (React + TypeScript, needs Node 22):

```bash
uv run uvicorn riskengine.api:app --port 8000   # API; docs at http://localhost:8000/docs
cd web && npm install && npm run dev            # dashboard at http://localhost:5173
```

Or as one server (what gets deployed): build once, then the dashboard and the API share one URL.

```bash
cd web && npm install && npm run build && cd ..
uv run uvicorn riskengine.site:app --port 8000   # dashboard at /, API at /api, docs at /api/docs
```

Pages: Overview (scoreboard), Engine (per stock), Analyze (live scoring), Live replay (the window
streamed in time order, with alerts and Module A weights), Module A.

## 5. Key Results & Domain Impact

Every experiment, its verdict and its stored results: [EXPERIMENTS.md](docs/EXPERIMENTS.md).

All test rules were written down before scoring ([DECISIONS.md](docs/DECISIONS.md)); each model
was scored once on a public held-out test set. Speed is measured on CPU (the engine runs on
CPU), batch 64. 95% intervals from 1,000 bootstrap resamples.

### Sentiment (TFNS test, 2,388 tweets) — full table: [RESULTS.md](docs/RESULTS.md)

| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Size | Texts/s |
|---|---|---|---|---|---|---|
| FinBERT off the shelf (baseline) | 0.726 | [0.707, 0.743] | 0.669 | [0.646, 0.689] | 438 MB | 13 |
| FinBERT fine-tuned | 0.893 | [0.880, 0.905] | 0.862 | [0.845, 0.878] | 438 MB | 13 |
| DeBERTa-v3 teacher | **0.905** | [0.892, 0.916] | **0.882** | [0.866, 0.897] | 738 MB | 9 |
| bge-small fine-tuned | 0.885 | [0.873, 0.897] | 0.848 | [0.830, 0.864] | 133 MB | 45 |
| bge-small distilled from teacher | 0.885 | [0.872, 0.898] | 0.854 | [0.837, 0.870] | 133 MB | 43 |
| bge-small fine-tuned, **int8 ONNX** (engine default) | 0.877 | [0.863, 0.889] | 0.836 | [0.818, 0.853] | **34 MB** | **56** |
| bge-small shared encoder, sentiment head | 0.886 | [0.872, 0.898] | 0.855 | [0.837, 0.871] | 133 MB | 46 |

- The engine's sentiment model beats the baseline by **+15 points accuracy**, at **4.3× the
  speed** and **1/13 of the size**.
- The teacher is the most accurate model (0.905). It missed our 0.91 target, and distilling it
  into bge-small did not move accuracy (0.885 → 0.885; macro-F1 0.848 → 0.854).
- int8 costs 0.8 points of accuracy for a 4× smaller, 1.24× faster model; it is within the
  1-point limit we set, so it is the default.

### Sentiment on four more public sets (E18), with a leak check first

Each test text was compared with every training text (exact match, and TF-IDF cosine ≥ 0.9)
**before** scoring. The engine model was trained on tweets only and saw **0 rows** of these
sets, so every number below is a clean held-out score. Dataset revisions, file and model hashes
are frozen in [e18_manifest.json](data/outputs/e18_manifest.json).

| Set | Rows | Engine (int8) macro-F1 | 95% CI | Full precision |
|---|---|---|---|---|
| Financial PhraseBank, all annotators agree | 2,264 | **0.866** | [0.849, 0.881] | 0.880 |
| Financial PhraseBank, ≥ 50% agree | 4,846 | 0.685 | [0.670, 0.700] | 0.702 |
| FiQA-2018 sentiment, test | 150 | 0.573 | [0.486, 0.649] | 0.602 |
| NWGI news, test (labels written by a GPT model) | 4,047 | 0.406 | [0.390, 0.419] | 0.414 |

- **Transfer:** trained only on tweets, the model keeps 0.87 on the news sentences annotators
  fully agree on.
- **Long text is a weakness:** NWGI items are paragraphs and the model reads 64 tokens; the
  neutral class drops to 0.28 F1. NWGI's labels are machine-written, so it is a check, not a
  human benchmark.
- **FiQA is small:** 150 rows give an interval about 16 points wide.
- **In-sample scores are marked, never counted:** FinBERT was trained on PhraseBank, and our
  teacher, distilled and shared-encoder models used PhraseBank and FiQA training rows, so their
  PhraseBank scores are labelled in-sample.
- int8 costs 1–3 macro-F1 points against full precision (paired bootstrap: significant on
  PhraseBank and NWGI, a tie on FiQA).
- **Trained on more than tweets transfers better:** on FiQA and NWGI, our distilled bge-small
  (same size, trained also on PhraseBank and FiQA training rows) beats the engine model by
  0.11 and 0.14 macro-F1 (paired intervals below 0), and off-the-shelf FinBERT beats it on
  NWGI. These test sets were not used to choose a model; a switch would be a new, pre-registered
  experiment. Full table: [DECISIONS.md](docs/DECISIONS.md) E18.
- **E20, the follow-up experiment:** choosing on dev sets (tweets + news) fixed in advance, the
  distilled model gained 0.12 on news but lost 0.035 on tweets, beyond the 0.01 allowed, so the
  engine model stays. On a fresh human-labelled news test (SEntFiN, 7,903 headlines) the
  engine scores 0.745 macro-F1 and the distilled model 0.765 (paired +0.020 [0.013, 0.027]).
  The trade-off between tweets and news is reported, not hidden.
- **E21, one model trained on tweets and news:** chosen on dev, it again lost on tweets, so the
  engine stays. On fresh data the picture flipped: on FinEntity (851 human-labelled Reuters
  paragraphs) the engine wins, 0.765 vs 0.700 macro-F1; on 251 retail tweets from 2021–22
  (labels drafted by AI assistants and reviewed by the author) the new model wins, 0.591 vs
  0.452. Dev sets from the training sources did not predict new sources, and later retail
  tweets are much harder for every model than the TFNS benchmark.
- **Decision:** the engine keeps its current model, which wins on the primary fresh test
  (FinEntity). The E21 model is a promising lead for retail tweets; it is not used, and nothing
  was changed after the confirmation. Generalising across text sources (news vs retail tweets,
  older vs newer data) is left as future research.

### Event type (TFN-topic test, 4,117 tweets, 4 mapped classes)

| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Size | Texts/s |
|---|---|---|---|---|---|---|
| Keyword rules (baseline) | 0.740 | [0.726, 0.754] | 0.505 | [0.481, 0.528] | 0 | 13,193 |
| bge-small topic model | 0.960 | [0.954, 0.966] | 0.907 | [0.889, 0.923] | 133 MB | 39 |
| FinBERT topic model | **0.966** | [0.961, 0.972] | **0.933** | [0.919, 0.946] | 438 MB | 11 |
| bge-small shared encoder, topic head | 0.950 | [0.943, 0.956] | 0.900 | [0.882, 0.915] | 133 MB | 41 |
| bge-small topic model, int8 ONNX (engine until E16) | 0.951 | [0.944, 0.957] | 0.886 | [0.866, 0.903] | **34 MB** | 44 |

- Geopolitical F1 rises from 0.14 (keywords) to 0.93.
- The shared encoder gives sentiment and event type in **one pass** (half the compute of two
  models). Its topic head is 1.04 points below the single bge-small model, just outside our
  1-point limit. The single-task topic models were trained with 999 texts that also appear in
  the test set; the shared encoder was not (see below).

- The engine uses the int8 bge-small topic model: 0.97 points below its torch version (inside
  our 1-point limit) and 1.6 points below FinBERT, for 1/13 of FinBERT's size
  ([DECISIONS.md](docs/DECISIONS.md) E11).

### Event type, 9 learned classes (same test, E16)

With 4 classes the model put product, earnings, analyst, legal and personnel news all in
"Other". E16 maps all 20 TFN topics to 9 learned classes (the 3 credit classes stay rules), fixed
before training. A new 9-class bge-small model was trained without the 999 leaked texts.

| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Size | Texts/s |
|---|---|---|---|---|---|---|
| Keyword rules (baseline) | 0.419 | [0.404, 0.435] | 0.372 | [0.349, 0.393] | 0 | 2,423 |
| Old engine model (20 topics, remapped) | 0.897 | [0.888, 0.906] | 0.813 | [0.796, 0.830] | 34 MB | 47 |
| 9-class bge-small | **0.904** | [0.895, 0.913] | **0.861** | [0.842, 0.877] | 133 MB | 38 |
| 9-class bge-small, **int8 ONNX** (engine default) | 0.897 | [0.888, 0.905] | 0.841 | [0.822, 0.858] | **34 MB** | 46 |

| F1 per class (95% CI for the engine model) | Keywords | Old engine model | 9-class | 9-class int8 |
|---|---|---|---|---|
| Product news | 0.122 | 0.903 | 0.900 | **0.892** [0.876, 0.906] |
| Earnings | 0.647 | 0.948 | 0.960 | **0.949** [0.932, 0.964] |
| Analyst rating | 0.346 | 0.214 | 0.561 | **0.465** [0.349, 0.585] |
| Legal/Regulatory | 0.490 | 0.848 | 0.869 | **0.826** [0.768, 0.879] |
| Management change | 0.161 | 0.918 | 0.951 | **0.941** [0.908, 0.973] |
| Geopolitical | 0.145 | 0.916 | 0.913 | **0.910** [0.884, 0.934] |
| Macroeconomic | 0.376 | 0.917 | 0.922 | **0.911** [0.898, 0.924] |
| M&A | 0.571 | 0.751 | 0.763 | **0.766** [0.708, 0.817] |
| Other | 0.490 | 0.902 | 0.907 | **0.906** [0.895, 0.916] |

- **E16 passed.** Product news F1 is 0.892 (rule: ≥ 0.85). Geopolitical and Macro are each 0.6
  points below the old model, and M&A is 1.5 points above it (rule: lose at most 2 points).
  int8 is 0.7 points of accuracy below full precision (inside the 1-point limit), so it is the
  engine default.
- **Most of the product-news gain comes from the new mapping, not the new model.** The old model
  already scores 0.903 on Product news once its 20 topics are remapped. The new model mainly
  adds **Analyst rating** (0.21 → 0.47) and +2.8 points of macro-F1, and it was trained
  without the leaked texts.
- **Weak spots:**
  - Analyst rating finds only 32% of analyst texts. There are 73 in the test, and the class is
    small in training.
  - int8 costs Legal/Regulatory 4.3 points.
  - A plain "Microsoft beats earnings estimates" is called Other. TFN earnings tweets usually
    carry the numbers. This is kept as a known miss in the tests.
- **Not comparable:** the 9-class macro-F1 (0.841) cannot be compared with the 4-class 0.886
  above. These are different classes, and Macro now also includes oil, metals, currencies and
  Treasuries.
- "Product news" is TFN's "Company | Product News": any company or product news, not only
  launches.

### Demo window re-scored with our models (14,169 texts, 9 Apr – 10 Jun 2020)

| | Baselines (FinBERT + keywords) | Our int8 models | + credit rules and gate (E12, E13) |
|---|---|---|---|
| Time to score on CPU | — | 9 min 54 s, 1.0 GB RAM | 11 min 43 s, 1.0 GB RAM |
| Texts tagged M&A / Macro / Geopolitical | 245 / 229 / 220 | 612 / 186 / 67 | 612 / 186 / 67 |
| Texts tagged Credit downgrade / Default | — | 69 / 17 | 7 / 10 |
| High-impact texts (> 7) | 48 | 41 | 6 |
| Credit alerts sent to Module B | — | — | 4 (3 real) |
| Module A sentiment tilt: return / Sharpe | 23.4% / 4.41 | 24.1% / 4.55 | 24.4% / 4.61 |
| Equal weight (benchmark): return / Sharpe | 25.5% / 4.88 | 25.5% / 4.88 | 25.5% / 4.88 |

- Same event class on 89.7% of texts; sentiment correlation 0.61 between the two runs.
- The tilt improves a little with our models but still trails equal weight in this 2-month window.
- **Re-scored with the 9-class model (E16):** 11 min 3 s, 1.0 GB RAM. Texts per class, before → after:

  | Class | Before | After |
  |---|---|---|
  | Other | 13,286 | 10,228 |
  | Product news | 0 | 1,723 |
  | M&A | 612 | 802 |
  | Earnings | 0 | 590 |
  | Macroeconomic | 186 | 410 |
  | Analyst rating | 0 | 217 |
  | Geopolitical | 67 | 82 |
  | Legal/Regulatory | 0 | 68 |
  | Management change | 0 | 31 |
  | Default / Credit downgrade / Covenant breach | 10 / 7 / 1 | 10 / 7 / 1 |

- **Launch texts:** of the 134 texts the old keyword rule called a product launch, 96 are now
  Product news, 37 Other and 1 Earnings.
- **Credit alerts are unchanged:** the same 4 alerts with the same CET1 results. The credit rules
  override the model, and the credit severities did not change. Module A is also unchanged
  (24.4% / 4.61); its tilt uses sentiment only.
- **Analyst rating texts and the credit gate:** none of the 217 would pass the gate even if
  treated as a credit downgrade (172 have impact ≤ 7, 55 are not about the obligor, 44 name a
  bank only as the analyst). The gate stays as fixed in E13.
- **One new false high-impact text:** "Fitch Affirms Ratings On Morgan Stanley, Citi, …" is now
  called M&A and gets impact 8 (because an agency is named), so high-impact texts go from 6 to 7.
  It is not a credit class, so it cannot reach Module B.

### Module B: interest-rate shock (E22)

Treasury yields now move with each crisis episode (a parallel shift of the 10-year change),
repricing the fixed-rate bonds and the pay-fixed swaps. In GFC 2008, euro 2011 and COVID 2020
yields fell, so the bonds gained and offset part of the credit loss (COVID: CET1 4.22% →
7.05%, still below the 8% floor). A controlled +200 bp rise alone takes CET1 from 13% to
6.18%. The bank's funding side is not modelled, so crisis-time rate gains are not offset by
liabilities, and one rate is used for every tenor: a controlled stress, not a forecast. Rules,
checks and full results: [DECISIONS.md](docs/DECISIONS.md) E22.

**Does the alert itself predict anything (E22b)?** Running the same credit rules and gate over
ten years of news headlines (2010 – Apr 2020) gave 79 alerts, 40 of them independent. The
company's stock then lagged the S&P 500 in 55% of cases against a 48% base rate (binomial
p = 0.23): **not significant, a fail**. Module B's predicted loss size was unrelated to the
reaction. Reading the alerts showed why in part: many history "downgrades" on JPM and BAC are
securitisations that carry the bank's name, not the bank's own rating. The stock return is a
proxy for credit stress, not a credit loss.

### Credit alerts: which news may trigger Module B ([DECISIONS.md](docs/DECISIONS.md) E12, E13)

A hand check of the 41 high-impact texts above found **1 real credit event**. The rest were
stock-analyst rating changes ("Goldman Downgrades Apple to Sell"), banks linked as the analyst
("iRobot Downgraded by JPMorgan" → JPM), and noise ("Google pays Apple to be the default search
engine"). Three fixes, all rules, written down before the code:

- **Who is the news about:** a bank named only as the analyst of a stock rating is tagged
  `analyst`, kept on the dashboard, and ignored by both modules. All 126 tagged links in the
  demo were read: 126 correct. In 30 bank rating texts left as subject, 6 were misses (none
  high-impact).
- **What is a credit event:** credit downgrade needs a rating agency (Moody's, Fitch, S&P, DBRS;
  not the S&P index) or a credit word (junk, credit rating, bonds…); stock ratings are flagged
  "analyst rating" and never credit; "default" needs a debt word next to it. +1 impact when an
  agency is named.
- **Credit gate:** Module B fires only for a credit event on the company the news is
  about (first cashtag), with impact > 7, confirmed by a named agency or a second text within
  24 hours. Every row gets the reason it passed or was blocked (`GET /gate`).

| Real credit events on our stocks in the window | Impact | Before: impact > 7 | After: gate |
|---|---|---|---|
| Moody's downgrades Boeing to Baa2 (10 Apr) | 8 | missed (was 7) | fired |
| S&P lowers Disney's credit rating (23 Apr) | 8 | missed (was 7) | fired |
| Fitch downgrades Disney's IDR to A- (6 May) | 9 | fired | fired |
| **Alerts in total** | | **41 (1 real)** | **4 (3 real)** |

- The one false alert: "India Stocks, Rupee Shrug Off Moody's Rate Cut … $BAC $MCO" fires on BAC
  (India's sovereign rating; $BAC is only the first cashtag). Left as it is: the rules are frozen.
- These demo numbers are a **design check**, not a test: the texts were read while writing the
  rules, and the rules were amended once after the first check (E13). The test is the hand
  labelled gold set (E3), scored once; a learned credit model will be compared with these rules.

### Impact score 1–10 (does it rank real price moves?) — [IMPACT_V2.md](docs/IMPACT_V2.md)

Spearman correlation between impact and the stock's real abnormal move over two days.
**Fresh** = 71 other S&P 100 stocks (those with FNSPID news and prices), never used for fitting or choosing.

| Model | Fresh 2019 (n = 26,964) | Fresh 2020 (n = 15,098) |
|---|---|---|
| Fixed formula (old baseline) | 0.038 | −0.041 |
| Volatility only (baseline) | 0.145 | **0.307** |
| Learned v2, ridge (chosen on validation) | **0.294** | −0.041 |
| Learned v2, gradient boosting (not chosen) | 0.222 | 0.275 |

- In 2019 the learned score doubles the volatility baseline: headlines it rates ≥ 8 moved their
  stock 3.1% on average, against 1.1% for those rated ≤ 3.
- In the COVID crash (VIX up to 83, far above anything in training) the linear model
  extrapolated badly: 20% of its predictions fell below zero and were floored, so it lost to
  volatility alone. Our pre-set success rule (beat volatility in both years) is **not met**.

### Problems found and reported, not hidden

- **Test leaks:** 84 sentiment and 999 topic training texts also appear in the public test sets.
  New models were trained without them; older scores are kept, marked.
- **Dataset bug:** the Kaggle tweet file gives AMZN, MSFT and PG the same tweets.
- **False credit alerts:** 40 of the first 41 high-impact alerts were not credit events (see
  "Credit alerts" above); now 1 of 4.
- **E16 weak spots:** Analyst rating F1 0.47 (it finds only 32% of analyst texts); int8 costs
  Legal/Regulatory 4.3 points; "Microsoft beats earnings estimates" without numbers is called
  Other. Most of the Product news gain came from the new mapping, not the new model.
- **E21 limits:** the 2021–22 tweet labels are AI-assisted (ChatGPT, Gemini, Claude) and
  reviewed by one person, not a blind human test; the engine scores only 0.45 on them.
- **E18 limits:** weak on long news paragraphs (NWGI 0.41 macro-F1, below off-the-shelf
  FinBERT's 0.55) and on PhraseBank's positive class (0.50 F1 at ≥ 50% agreement); FiQA's 150
  rows are too few for a tight estimate.
- **Code bug:** pandas 3 stores time in microseconds; one v1 impact feature counted the last
  ~1,000 days instead of 24 hours. Fixed and tested; v1 results are kept as reported.

## License

[MIT](LICENSE)
