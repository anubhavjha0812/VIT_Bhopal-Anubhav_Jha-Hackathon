"""Score every historical headline once with our sentiment and topic models (slow on CPU, cached).

uv run python -m riskengine.impact.scores
"""

import logging

import pandas as pd
import torch

from riskengine.ingest.news import load_fnspid
from riskengine.nlp.classifier import TextClassifier
from riskengine.nlp.events import CREDIT_CLASSES, OWN_TOPIC, classify_keywords, topic_to_event
from riskengine.nlp.sentiment import OWN_SENTIMENT, TransformerSentiment
from riskengine.paths import DATA

log = logging.getLogger(__name__)
SCORES_PATH = DATA / "cache" / "headline_scores.csv.gz"


def score_headlines() -> pd.DataFrame:
    news = load_fnspid()[0]
    texts = news["text"].tolist()

    log.info("sentiment for %d headlines", len(texts))
    sentiment = TransformerSentiment(OWN_SENTIMENT, name="own").score(texts)

    log.info("topics for %d headlines", len(texts))
    topics = TextClassifier(str(OWN_TOPIC))
    probs = torch.cat(list(topics.probs(texts))).numpy()
    p = pd.DataFrame(probs, columns=[f"p_{t}" for t in topics.labels], index=news.index)

    model_event = [topic_to_event(topics.labels[i]) for i in probs.argmax(1)]
    rules = [classify_keywords(t) for t in texts]
    event = [r if r in CREDIT_CLASSES else m for r, m in zip(rules, model_event, strict=True)]

    out = pd.concat([news[["doc_id"]], p], axis=1)
    out.insert(1, "sentiment", [round(s, 4) for s in sentiment])
    out.insert(2, "event", event)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORES_PATH, index=False, float_format="%.4f")
    return out


def load_scores() -> pd.DataFrame:
    return pd.read_csv(SCORES_PATH, dtype={"doc_id": str})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    df = score_headlines()
    log.info("wrote %d rows -> %s", len(df), SCORES_PATH)
