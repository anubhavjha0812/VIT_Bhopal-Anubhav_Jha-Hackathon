"""Public test sets (MIT licence), downloaded once and kept small in data/eval/.

Mappings follow docs/DECISIONS.md (E1, E2) and must not change after scoring.
"""

import pandas as pd

from riskengine.nlp.events import LEARNED_CLASSES, TOPIC_TO_EVENT, TOPIC_TO_EVENT_E2
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
EVENT_TEST_CLASSES = ["Macroeconomic", "M&A", "Geopolitical", "Other"]  # E2
EVENT9_TEST_CLASSES = LEARNED_CLASSES  # E16


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
    """4,117 finance tweets: text, topic, label (E2, 4 classes), label9 (E16, 9 classes)."""
    df = _cached("twitter-financial-news-topic", "topic_valid.csv")
    topic = df["label"].map(dict(enumerate(TOPIC_NAMES)))
    return pd.DataFrame(
        {
            "text": df["text"],
            "topic": topic,
            "label": topic.map(TOPIC_TO_EVENT_E2).fillna("Other"),
            "label9": topic.map(TOPIC_TO_EVENT),
        }
    )


# E18: more public sentiment test sets, pinned to the dataset revisions that were scored
E18_SOURCES = {  # repo: (revision, licence)
    "takala/financial_phrasebank": ("8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964", "CC BY-NC-SA 3.0"),
    "pauri32/fiqa-2018": ("74b7dd5d4fc61e0fa5d979bef2c53111ea69c58a", "not stated"),
    "oliverwang15/news_with_gpt_instructions": (
        "b7c333378679b2a74d7248b407c06021d3f26adb",
        "not stated",
    ),
}
PHRASEBANK_ZIP = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/"
    f"{E18_SOURCES['takala/financial_phrasebank'][0]}/data/FinancialPhraseBank-v1.0.zip"
)
FIQA_CSV = (
    "https://huggingface.co/datasets/pauri32/fiqa-2018/resolve/"
    f"{E18_SOURCES['pauri32/fiqa-2018'][0]}/{{split}}.csv"
)
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset={repo}&config=default&split={split}"
FIQA_LABELS = {0: "positive", 1: "neutral", 2: "negative"}  # as E7


def phrasebank_lines(level: str) -> pd.DataFrame:
    """One PhraseBank file (e.g. 50Agree, 75Agree, AllAgree): text, label."""
    path = EVAL_DIR / f"phrasebank_{level}.csv"
    if not path.exists():
        import io
        import urllib.request
        import zipfile

        raw = urllib.request.urlopen(PHRASEBANK_ZIP).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(f"FinancialPhraseBank-v1.0/Sentences_{level}.txt").decode("latin-1")
        rows = [line.rsplit("@", 1) for line in text.splitlines() if "@" in line]
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return pd.read_csv(path)


def fiqa_split(split: str) -> pd.DataFrame:
    path = EVAL_DIR / f"fiqa_{split}.csv"
    if not path.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(FIQA_CSV.format(split=split)).to_csv(path, index=False)
    df = pd.read_csv(path)
    return pd.DataFrame({"text": df["sentence"], "label": df["label"].map(FIQA_LABELS)})


def _rows(repo: str, split: str) -> pd.DataFrame:
    """A whole split through the HF rows API, 100 rows a page (no parquet reader needed)."""
    import json
    import time
    import urllib.error
    import urllib.request

    url, out = ROWS_API.format(repo=repo, split=split), []
    while True:
        for wait in (0, 5, 20, 60, 120, 300):  # the API rate-limits long reads (HTTP 429)
            time.sleep(wait)
            try:
                page = json.load(urllib.request.urlopen(f"{url}&offset={len(out)}&length=100"))
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
        else:
            raise RuntimeError(f"{repo} {split}: still rate-limited at row {len(out)}")
        out += [r["row"] for r in page["rows"]]
        if len(out) >= page["num_rows_total"] or not page["rows"]:
            return pd.DataFrame(out)


def nwgi_test() -> pd.DataFrame:
    """4,047 news paragraphs; GPT-written labels folded to 3 classes (E18)."""
    path = EVAL_DIR / "nwgi_test.csv"
    if not path.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        _rows("oliverwang15/news_with_gpt_instructions", "test")[["news", "label"]].to_csv(
            path, index=False
        )
    df = pd.read_csv(path)
    return pd.DataFrame({"text": df["news"], "label": df["label"].str.split().str[-1]})


E18_FILES = {  # test set -> (cached file, source repo)
    "PhraseBank 50Agree": ("phrasebank_50Agree.csv", "takala/financial_phrasebank"),
    "PhraseBank AllAgree": ("phrasebank_AllAgree.csv", "takala/financial_phrasebank"),
    "FiQA-2018 test": ("fiqa_test.csv", "pauri32/fiqa-2018"),
    "NWGI test": ("nwgi_test.csv", "oliverwang15/news_with_gpt_instructions"),
}


def e18_tests() -> dict[str, pd.DataFrame]:
    return {
        "PhraseBank 50Agree": phrasebank_lines("50Agree"),
        "PhraseBank AllAgree": phrasebank_lines("AllAgree"),
        "FiQA-2018 test": fiqa_split("test"),
        "NWGI test": nwgi_test(),
    }
