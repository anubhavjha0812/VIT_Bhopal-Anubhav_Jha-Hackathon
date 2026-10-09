"""Batched CPU inference for any Hugging Face sequence classifier (local folder or hub id).

A local folder with a `model.onnx` (from nlp/onnx_int8.py) runs on onnxruntime instead of torch.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path


class TextClassifier:
    def __init__(self, model: str, batch_size: int = 64, max_length: int = 64):
        # Imported here, not at the top: replaying stored signals never pays torch's load time.
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        onnx_file = Path(model) / "model.onnx"
        if onnx_file.exists():
            import onnxruntime as ort

            self.model = None
            self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
            self.inputs = {i.name for i in self.session.get_inputs()}
            self.size_mb = onnx_file.stat().st_size / 1e6
            id2label = AutoConfig.from_pretrained(model).id2label
        else:
            self.session = None
            self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
            # weights in memory (parameters x bytes each), independent of the disk cache
            self.size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
            id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.batch_size, self.max_length = batch_size, max_length

    def _logits(self, batch):
        if self.session is None:
            return self.model(**batch).logits
        feed = {k: v.numpy() for k, v in batch.items() if k in self.inputs}
        return self._torch.from_numpy(self.session.run(None, feed)[0])

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
                yield self._logits(batch).softmax(-1)

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Most likely label per text."""
        return [self.labels[i] for p in self.probs(texts) for i in p.argmax(-1).tolist()]
