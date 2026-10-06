"""Sentiment in [-1, 1] = P(positive) - P(negative).

- finbert: off-the-shelf FinBERT (ProsusAI/finbert), the baseline.
- own: our model fine-tuned on Twitter Financial News Sentiment (notebooks/train_classifiers.py).
"""

from collections.abc import Sequence
from typing import Protocol

from riskengine.nlp.classifier import TextClassifier
from riskengine.paths import MODELS

FINBERT = "ProsusAI/finbert"
OWN_SENTIMENT = MODELS / "sentiment"


class SentimentScorer(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[float]: ...


class TransformerSentiment(TextClassifier):
    def __init__(self, model: str = FINBERT, name: str = "finbert", **kw):
        super().__init__(str(model), **kw)
        self.name = name
        self.labels = [label.lower() for label in self.labels]
        self.pos, self.neg = self.labels.index("positive"), self.labels.index("negative")

    def score(self, texts: Sequence[str]) -> list[float]:
        return [s for p in self.probs(texts) for s in (p[:, self.pos] - p[:, self.neg]).tolist()]

    def classify(self, texts: Sequence[str]) -> list[str]:
        """Most likely label per text: positive, negative or neutral."""
        return self.predict(texts)


FinBert = TransformerSentiment  # the baseline is the default model


def get_scorer(kind: str) -> SentimentScorer:
    if kind == "finbert":
        return TransformerSentiment()
    if kind == "own":
        if not OWN_SENTIMENT.exists():
            raise FileNotFoundError(f"own sentiment model not found at {OWN_SENTIMENT}")
        return TransformerSentiment(OWN_SENTIMENT, name="own")
    raise NotImplementedError(f"sentiment model {kind!r} is not built yet")
