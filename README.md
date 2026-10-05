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

_TODO._ Only public or synthetic data is used, with no confidential client data. Small samples are kept in `data/`, and each source and licence will be listed here.

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

_TODO: results against each baseline._

## License

[MIT](LICENSE)