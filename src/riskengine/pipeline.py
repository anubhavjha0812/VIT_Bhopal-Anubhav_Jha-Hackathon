"""Engine pipeline: window docs -> sentiment, event, impact per (doc, company) -> signals file.

Inputs come from the committed sample (data/samples/docs_window.jsonl.gz) so a clean clone
runs without the raw downloads. `build_sample` recreates it from data/raw/.
"""

import logging
from collections.abc import Callable, Sequence
from typing import cast

import pandas as pd

from riskengine.config import Config
from riskengine.ingest.coverage import in_window
from riskengine.ingest.linking import Linker
from riskengine.ingest.news import load_fnspid
from riskengine.ingest.tweets import load_akkerman
from riskengine.nlp.event_engine import get_engine_classifier
from riskengine.nlp.events import is_analyst_rating, names_agency
from riskengine.nlp.impact import impact_formula
from riskengine.nlp.sentiment import EntitySentimentScorer, SentimentScorer, get_scorer
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


def _impacts(rows: pd.DataFrame) -> list[int]:
    return [
        impact_formula(e, s, src, names_agency(t))
        for e, s, src, t in zip(
            rows["event"], rows["sentiment"], rows["source"], rows["text"], strict=True
        )
    ]


def _entity_scores(
    rows: pd.DataFrame, scorer: EntitySentimentScorer, linker: Linker
) -> list[float]:
    texts = rows["text"].tolist()
    entities = [linker.company_name(ticker) for ticker in rows["ticker"]]
    named = [i for i, entity in enumerate(entities) if entity is not None]
    scores = [0.0] * len(rows)
    if named:
        values = scorer.score_entities([texts[i] for i in named], [entities[i] for i in named])
        for i, value in zip(named, values, strict=True):
            scores[i] = round(value, 4)
    missing = [i for i, entity in enumerate(entities) if entity is None]
    if missing:
        # One headline fallback per distinct text, copied to each unnamed company row.
        headlines = list(dict.fromkeys(texts[i] for i in missing))
        fallback = dict(zip(headlines, scorer.score(headlines), strict=True))
        for i in missing:
            scores[i] = round(fallback[texts[i]], 4)
    return scores


def score_docs(
    docs: pd.DataFrame,
    config: Config,
    scorer: SentimentScorer,
    classify_events: Callable[[Sequence[str]], list[str]] | None = None,
    linker: Linker | None = None,
) -> pd.DataFrame:
    """One row per (doc, company): sentiment, event, impact and which models produced them."""
    classify_events = classify_events or get_engine_classifier(config.events)
    linker = linker or Linker.from_csv()
    docs = docs.copy()
    texts = docs["text"].tolist()
    if config.sentiment != "entity":
        docs["sentiment"] = [round(s, 4) for s in scorer.score(texts)]
    docs["event"] = classify_events(texts)
    docs["analyst_rating"] = [is_analyst_rating(t) for t in texts]
    if config.sentiment != "entity":
        docs["impact"] = _impacts(docs)
    docs["model"] = f"{scorer.name}+{config.events}+{config.impact}"
    rows = docs.explode("tickers").rename(columns={"tickers": "ticker"})
    if config.sentiment == "entity":
        rows["sentiment"] = _entity_scores(rows, cast(EntitySentimentScorer, scorer), linker)
        rows["impact"] = _impacts(rows)
    # A bank that only gives a stock rating is kept but is not the subject (DECISIONS E12).
    rows["role"] = [
        "analyst" if tk in linker.analyst_only(text) else "subject"
        for tk, text in zip(rows["ticker"], rows["text"], strict=True)
    ]
    cols = ["doc_id", "published_at", "source", "ticker", "role", "text", "sentiment", "event"]
    out = rows[[*cols, "analyst_rating", "impact", "model"]].reset_index(drop=True)
    if config.impact == "learned":
        # E28: news rows with price history take the learned 1-10 score as `impact`; tweets and
        # stocks without prices keep the formula (kept as `impact_formula` for every row).
        from riskengine.impact import learned  # sklearn is only needed in this mode

        out = learned.apply(out)
    return out


def run(config: Config, docs: pd.DataFrame | None = None) -> pd.DataFrame:
    docs = load_sample() if docs is None else docs
    log.info("scoring %d docs with %s", len(docs), config.sentiment)
    signals = score_docs(docs, config, get_scorer(config.sentiment))
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals.to_json(SIGNALS_PATH, orient="records", lines=True, date_format="iso")
    return signals


def subjects(signals: pd.DataFrame) -> pd.DataFrame:
    """Rows where the company is what the text is about; files without `role` count as all."""
    if "role" not in signals:
        return signals
    return signals[signals["role"] == "subject"]


def load_signals(path=SIGNALS_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"doc_id": str})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
