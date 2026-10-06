"""Batched CPU inference for any Hugging Face sequence classifier (local folder or hub id)."""

from collections.abc import Iterator, Sequence


class TextClassifier:
    def __init__(self, model: str, batch_size: int = 64, max_length: int = 64):
        # Imported here, not at the top: replaying stored signals never pays torch's load time.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.batch_size, self.max_length = batch_size, max_length

    def probs(self, texts: Sequence[str]) -> Iterator:
        """Yield one (batch, n_labels) probability tensor per batch."""
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    list(texts[i : i + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                yield self.model(**batch).logits.softmax(-1)

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Most likely label per text."""
        return [self.labels[i] for p in self.probs(texts) for i in p.argmax(-1).tolist()]
