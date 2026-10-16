"""Batch-independent E24 inference and probability-cache validation.

Each ONNX invocation contains exactly one headline, padded only to its own token length.
Callers may group or reorder requests without changing that headline's input tensor.
Production TextClassifier defaults are deliberately separate from this evaluation method.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

METHOD = "onnx_cpu_per_row_dynamic_padding_v1"
MAX_LENGTH = 64
THREADS = 1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference_metadata(model_dir: Path | None = None) -> dict:
    """All numerical settings and model/tokenizer assets required to replay inference."""
    import onnxruntime as ort
    import torch
    import transformers

    from riskengine.nlp.events import OWN_TOPIC

    model_dir = Path(model_dir or OWN_TOPIC)
    assets = [
        p
        for p in model_dir.iterdir()
        if p.name == "model.onnx" or p.suffix in {".json", ".txt", ".model"}
    ]
    return {
        "method": METHOD,
        "batch_size": 1,
        "padding": "longest_in_single_row",
        "truncation": True,
        "max_length": MAX_LENGTH,
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "intra_op_num_threads": THREADS,
        "inter_op_num_threads": THREADS,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "softmax": "torch_float32_softmax_last_dimension",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_sha256": _sha(model_dir / "model.onnx"),
        "assets_sha256": {p.name: _sha(p) for p in sorted(assets)},
    }


def validate_inference_metadata(stored: dict | None, expected: dict) -> None:
    if stored != expected:
        raise ValueError("E24 inference method/settings mismatch; regenerate the probability cache")


class EvaluationProbs:
    """A probability callable whose ONNX batch size cannot be changed by callers."""

    def __init__(self, model_dir: Path | None = None):
        import onnxruntime as ort
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from riskengine.nlp.events import OWN_TOPIC

        model_dir = Path(model_dir or OWN_TOPIC)
        self.inference_metadata = inference_metadata(model_dir)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.labels = [config.id2label[i] for i in range(len(config.id2label))]
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = THREADS
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def __call__(self, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
        rows = []
        with self._torch.inference_mode():
            for text in texts:
                tokens = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="np",
                )
                feed = {k: v for k, v in tokens.items() if k in self.inputs}
                logits = self.session.run(None, feed)[0]
                rows.append(self._torch.from_numpy(logits).softmax(-1).numpy())
        p = np.concatenate(rows) if rows else np.zeros((0, len(self.labels)), dtype=np.float32)
        return list(self.labels), p


def save_probability_cache(
    path: Path,
    ids: Sequence[str],
    names: Sequence[str],
    p: np.ndarray,
    metadata: dict,
    **extra,
) -> dict:
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(f"preserve the previous cache before regenerating: {path}")
    np.savez_compressed(
        path, item_id=np.array(ids, dtype=str), labels=np.array(names), probabilities=p
    )
    record = extra | {"inference": metadata, "cache_sha256": _sha(path)}
    meta_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_probability_cache(
    path: Path,
    ids: Sequence[str],
    expected_metadata: dict,
) -> tuple[list[str], np.ndarray, dict]:
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text())
    validate_inference_metadata(record.get("inference"), expected_metadata)
    if _sha(path) != record.get("cache_sha256"):
        raise ValueError("E24 probability cache hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        if saved["item_id"].tolist() != list(ids):
            raise ValueError("E24 probability cache row ids/order mismatch")
        names, p = saved["labels"].tolist(), saved["probabilities"].copy()
    if (
        p.shape != (len(ids), len(names))
        or len(set(names)) != len(names)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(axis=1), 1)
    ):
        raise ValueError("invalid E24 cached probability matrix")
    return names, p, record
