"""Sentiment in [-1, 1] = P(positive) - P(negative).

Baseline: off-the-shelf FinBERT (ProsusAI/finbert), CPU only.
"""

from collections.abc import Sequence
from typing import Protocol

FINBERT = "ProsusAI/finbert"


class SentimentScorer(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[float]: ...


class FinBert:
    name = "finbert"

    def __init__(self, model: str = FINBERT, batch_size: int = 64, max_length: int = 64):
        # Imported here, not at the top: replaying stored signals never pays torch's load time.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos, self.neg = labels["positive"], labels["negative"]
        self.batch_size, self.max_length = batch_size, max_length

    def score(self, texts: Sequence[str]) -> list[float]:
        out: list[float] = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    list(texts[i : i + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                p = self.model(**batch).logits.softmax(-1)
                out.extend((p[:, self.pos] - p[:, self.neg]).tolist())
        return out


def get_scorer(kind: str) -> SentimentScorer:
    if kind == "finbert":
        return FinBert()
    raise NotImplementedError(f"sentiment model {kind!r} is not built yet")
