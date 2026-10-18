"""Sentiment in [-1, 1] = P(positive) - P(negative).

- finbert: off-the-shelf FinBERT (ProsusAI/finbert), the baseline.
- own: our model fine-tuned on Twitter Financial News Sentiment (notebooks/train_classifiers.py),
  int8 ONNX in models/sentiment (DECISIONS E9, E11).
- entity: E27 int8 ONNX with (company name, headline) pairs; own is the headline fallback.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from riskengine.nlp.classifier import TextClassifier
from riskengine.paths import MODELS

FINBERT = "ProsusAI/finbert"
OWN_SENTIMENT = MODELS / "sentiment"
ENTITY_SENTIMENT = MODELS / "e27-entity-bge-small-en-v1.5-int8"


class SentimentScorer(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[float]: ...


class EntitySentimentScorer(SentimentScorer, Protocol):
    def score_entities(self, texts: Sequence[str], entities: Sequence[str]) -> list[float]: ...


class EntitySentiment:
    """E27's evaluated pair inference, with a lazy own-model headline fallback."""

    name = "entity"

    def __init__(self, model_dir: Path = ENTITY_SENTIMENT, fallback: SentimentScorer | None = None):
        # Import lazily: the evaluator also imports OWN_SENTIMENT from this module.
        from riskengine.eval.e27_entity_eval import EntityEvaluationProbs

        if not (model_dir / "model.onnx").is_file():
            raise FileNotFoundError(f"entity sentiment model not found at {model_dir}")
        self.probs = EntityEvaluationProbs(model_dir)
        self._fallback = fallback

    def score(self, texts: Sequence[str]) -> list[float]:
        """Headline-only score for rows without a known company name."""
        if not len(texts):
            return []
        if self._fallback is None:
            self._fallback = get_scorer("own")
        return self._fallback.score(texts)

    def score_entities(self, texts: Sequence[str], entities: Sequence[str]) -> list[float]:
        pairs = list(zip(entities, texts, strict=True))
        if not pairs:
            return []
        names, p = self.probs(pairs)
        labels = [label.lower() for label in names]
        if len(labels) != 3 or set(labels) != {"negative", "neutral", "positive"}:
            raise ValueError("expected negative/neutral/positive entity sentiment labels")
        p = np.asarray(p)
        if p.shape != (len(pairs), 3) or not np.isfinite(p).all():
            raise ValueError("invalid entity sentiment probabilities")
        return (p[:, labels.index("positive")] - p[:, labels.index("negative")]).tolist()


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
    if kind == "entity":
        return EntitySentiment()
    raise NotImplementedError(f"sentiment model {kind!r} is not built yet")
