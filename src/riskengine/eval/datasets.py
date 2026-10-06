"""Public test sets (MIT licence), downloaded once and kept small in data/eval/.

Mappings follow docs/DECISIONS.md (E1, E2) and must not change after scoring.
"""

import pandas as pd

from riskengine.nlp.events import TOPIC_TO_EVENT  # same table the engine uses
from riskengine.paths import DATA

EVAL_DIR = DATA / "eval"
HF = "https://huggingface.co/datasets/zeroshot/{repo}/resolve/main/{file}"

TFNS_LABELS = {0: "negative", 1: "positive", 2: "neutral"}  # Bearish, Bullish, Neutral

TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News",
    "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
    "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO",
    "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics",
    "Personnel Change", "Stock Commentary", "Stock Movement",
]  # fmt: skip
EVENT_TEST_CLASSES = ["Macroeconomic", "M&A", "Geopolitical", "Other"]


def _cached(repo: str, file: str) -> pd.DataFrame:
    path = EVAL_DIR / file
    if not path.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(HF.format(repo=repo, file=file)).to_csv(path, index=False)
    return pd.read_csv(path)


def tfns_test() -> pd.DataFrame:
    """2,388 finance tweets: text, label (positive / negative / neutral)."""
    df = _cached("twitter-financial-news-sentiment", "sent_valid.csv")
    return pd.DataFrame({"text": df["text"], "label": df["label"].map(TFNS_LABELS)})


def topic_test() -> pd.DataFrame:
    """4,117 finance tweets: text, topic, label (our event class via TOPIC_TO_EVENT)."""
    df = _cached("twitter-financial-news-topic", "topic_valid.csv")
    topic = df["label"].map(dict(enumerate(TOPIC_NAMES)))
    return pd.DataFrame(
        {"text": df["text"], "topic": topic, "label": topic.map(TOPIC_TO_EVENT).fillna("Other")}
    )
