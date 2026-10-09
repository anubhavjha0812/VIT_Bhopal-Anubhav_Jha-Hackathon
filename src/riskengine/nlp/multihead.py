"""One encoder, two heads: sentiment and TFN topic from a single pass (DECISIONS E10).

Loads a folder written by notebooks/teacher/train_teacher.py --stage multitask:
encoder weights + tokenizer, heads.pt and heads.json (label names per head).
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path


class MultiHeadModel:
    def __init__(self, path: str | Path, batch_size: int = 64, max_length: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.encoder = AutoModel.from_pretrained(path).float().eval()
        names = json.loads((path / "heads.json").read_text())
        self.sentiment_labels = [n.lower() for n in names["sentiment"]]
        self.topic_labels = names["topic"]
        heads = torch.load(path / "heads.pt", map_location="cpu")
        h = self.encoder.config.hidden_size
        self.heads = {}
        for task, labels in (("sentiment", self.sentiment_labels), ("topic", self.topic_labels)):
            layer = torch.nn.Linear(h, len(labels))
            layer.load_state_dict(heads[task])
            self.heads[task] = layer.eval()
        params = [
            *self.encoder.parameters(),
            *(p for m in self.heads.values() for p in m.parameters()),
        ]
        self.size_mb = sum(p.numel() * p.element_size() for p in params) / 1e6
        self.batch_size, self.max_length = batch_size, max_length

    def probs(self, texts: Sequence[str]) -> Iterator[tuple]:
        """Yield (sentiment probs, topic probs) per batch, both from one encoder pass."""
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    list(texts[i : i + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                x = self.encoder(**batch).last_hidden_state[:, 0]
                yield self.heads["sentiment"](x).softmax(-1), self.heads["topic"](x).softmax(-1)

    def predict(self, texts: Sequence[str]) -> tuple[list[str], list[str]]:
        """(sentiment label, topic label) per text."""
        sent, topic = [], []
        for s, t in self.probs(texts):
            sent += [self.sentiment_labels[i] for i in s.argmax(-1).tolist()]
            topic += [self.topic_labels[i] for i in t.argmax(-1).tolist()]
        return sent, topic

    def score(self, texts: Sequence[str]) -> list[float]:
        """Sentiment score P(positive) - P(negative), as TransformerSentiment.score."""
        pos, neg = self.sentiment_labels.index("positive"), self.sentiment_labels.index("negative")
        return [v for s, _ in self.probs(texts) for v in (s[:, pos] - s[:, neg]).tolist()]
