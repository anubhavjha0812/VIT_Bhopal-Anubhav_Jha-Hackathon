# AI Risk Engine: From News & Social Sentiment to Portfolio and Credit Risk

**Submission for the S&P Global & Crisil Campus Hackathon 2026** (individual entry)

| | |
|---|---|
| **Candidate** | Anubhav Jha |
| **College email** | anubhav.23bai11112@vitbhopal.ac.in |
| **College** | VIT Bhopal |
| **Demo video** | _Coming soon_ |
| **Slides** | _Coming soon_ (`docs/presentation.pdf`) |

> 🚧 **Work in progress.** Built step by step during the hackathon; each section is filled in as its part is finished.

## 1. Overview

An AI/NLP risk engine that reads **financial news** and **social media posts** and, for each company and event, produces:

- a **sentiment score** from -1.0 (very negative) to +1.0 (very positive)
- an **event type**: Geopolitical, Macroeconomic, Credit downgrade, Default, Covenant breach, M&A, Product launch or Other
- an **impact score** from 1 (minor) to 10 (severe)

These signals drive two modules:

- **Module A: Sentiment-driven rebalancer.** Adjusts the weights of a mock index of 15 S&P 100 stocks as the news changes, and shows the weights over time on a dashboard.
- **Module B: Event-driven credit stress test.** When a high-impact event (impact > 7) hits, it stresses a wholesale bank book of loans, bonds and derivatives. It shows the effect before and after, as spread widening, probability of default and expected loss.

Every part is compared against a simple baseline, so the gains are shown in numbers.

## 2. Architecture & Tech Stack

_TODO: architecture diagram (`docs/architecture.png`)._

Planned stack: Python 3.12, PyTorch (CPU), FastAPI, Streamlit, SQLite/JSONL; managed with [uv](https://docs.astral.sh/uv/).

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

_The demo currently loads the config and shows which version of each part is active; the pipeline is added step by step._

## 5. Key Results & Domain Impact

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

### Event type (TFN-topic test, 4,117 tweets, 4 mapped classes)

| Model | Accuracy | 95% CI | Macro-F1 | 95% CI | Size | Texts/s |
|---|---|---|---|---|---|---|
| Keyword rules (baseline) | 0.740 | [0.726, 0.754] | 0.505 | [0.481, 0.528] | 0 | 13,193 |
| bge-small topic model | 0.960 | [0.954, 0.966] | 0.907 | [0.889, 0.923] | 133 MB | 39 |
| FinBERT topic model | **0.966** | [0.961, 0.972] | **0.933** | [0.919, 0.946] | 438 MB | 11 |
| bge-small shared encoder, topic head | 0.950 | [0.943, 0.956] | 0.900 | [0.882, 0.915] | 133 MB | 41 |

- Geopolitical F1 rises from 0.14 (keywords) to 0.93.
- The shared encoder gives sentiment and event type in **one pass** (half the compute of two
  models). Its topic head is 1.04 points below the single bge-small model, just outside our
  1-point limit. The single-task topic models were trained with 999 texts that also appear in
  the test set; the shared encoder was not (see below).

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
- **Code bug:** pandas 3 stores time in microseconds; one v1 impact feature counted the last
  ~1,000 days instead of 24 hours. Fixed and tested; v1 results are kept as reported.

## License

[MIT](LICENSE)