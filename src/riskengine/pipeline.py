"""Engine pipeline: window docs -> sentiment, event, impact per (doc, company) -> signals file.

Inputs come from the committed sample (data/samples/docs_window.jsonl.gz) so a clean clone
runs without the raw downloads. `build_sample` recreates it from data/raw/.
"""

import logging
from collections.abc import Callable, Sequence

import pandas as pd

from riskengine.config import Config
from riskengine.ingest.coverage import in_window
from riskengine.ingest.news import load_fnspid
from riskengine.ingest.tweets import load_akkerman
from riskengine.nlp.events import get_event_classifier
from riskengine.nlp.impact import impact_formula
from riskengine.nlp.sentiment import SentimentScorer, get_scorer
from riskengine.paths import DATA

log = logging.getLogger(__name__)

SAMPLE_PATH = DATA / "samples" / "docs_window.jsonl.gz"
SIGNALS_PATH = DATA / "outputs" / "signals.jsonl.gz"
MAX_TWEETS_PER_STOCK_DAY = 20  # caps CPU time for FinBERT (~12-47 texts/s)


def build_sample(config: Config, max_tweets: int = MAX_TWEETS_PER_STOCK_DAY) -> pd.DataFrame:
    """All window news + up to `max_tweets` random tweets per stock per day (seed 0)."""
    tweets = in_window(load_akkerman()[0], config)
    news = in_window(load_fnspid()[0], config)

    per_stock = tweets.explode("tickers")
    per_stock["day"] = per_stock["published_at"].dt.date
    picked = (
        per_stock.groupby(["tickers", "day"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_tweets), random_state=0))
        .loc[:, "doc_id"]
    )
    tweets = tweets[tweets["doc_id"].isin(set(picked))]

    docs = pd.concat([news, tweets]).sort_values("published_at").reset_index(drop=True)
    docs["tickers"] = docs["tickers"].map(list)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs.to_json(SAMPLE_PATH, orient="records", lines=True, date_format="iso")
    log.info("sample: %d news + %d tweets -> %s", len(news), len(tweets), SAMPLE_PATH)
    return docs


def load_sample() -> pd.DataFrame:
    docs = pd.read_json(SAMPLE_PATH, lines=True, dtype={"doc_id": str})
    docs["published_at"] = pd.to_datetime(docs["published_at"], utc=True)
    return docs


def score_docs(
    docs: pd.DataFrame,
    config: Config,
    scorer: SentimentScorer,
    classify_events: Callable[[Sequence[str]], list[str]] | None = None,
) -> pd.DataFrame:
    """One row per (doc, company): sentiment, event, impact and which models produced them."""
    if config.impact != "formula":
        raise NotImplementedError("only the baseline impact formula exists so far")
    classify_events = classify_events or get_event_classifier(config.events)
    docs = docs.copy()
    texts = docs["text"].tolist()
    docs["sentiment"] = [round(s, 4) for s in scorer.score(texts)]
    docs["event"] = classify_events(texts)
    docs["impact"] = [
        impact_formula(e, s, src)
        for e, s, src in zip(docs["event"], docs["sentiment"], docs["source"], strict=True)
    ]
    docs["model"] = f"{scorer.name}+{config.events}+{config.impact}"
    rows = docs.explode("tickers").rename(columns={"tickers": "ticker"})
    cols = ["doc_id", "published_at", "source", "ticker", "text", "sentiment", "event"]
    return rows[[*cols, "impact", "model"]].reset_index(drop=True)


def run(config: Config, docs: pd.DataFrame | None = None) -> pd.DataFrame:
    docs = load_sample() if docs is None else docs
    log.info("scoring %d docs with %s", len(docs), config.sentiment)
    signals = score_docs(docs, config, get_scorer(config.sentiment))
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals.to_json(SIGNALS_PATH, orient="records", lines=True, date_format="iso")
    return signals


def load_signals(path=SIGNALS_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"doc_id": str})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
