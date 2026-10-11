"""E18: every sentiment model on the four extra public test sets, one prediction per row
(rules: docs/DECISIONS.md, E18). Only predictions are made here; metrics, intervals and the
leak split are computed in the repo (uv run python -m riskengine.eval --e18-preds FILE).

Same inference as the engine (riskengine.nlp.classifier): max 64 tokens, batch 64, argmax.
Torch models run on the GPU if there is one; the int8 ONNX model runs on the CPU, as in the
engine. Model weights come from the training kernels' outputs and must match the sha256 in
expected_hashes.json (the files the repo scored), or the run stops.

    python score_e18.py [--root /kaggle/input] [--out /kaggle/working]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN, BATCH = 64, 64
TESTS = {
    "PhraseBank 50Agree": "phrasebank_50Agree.csv",
    "PhraseBank AllAgree": "phrasebank_AllAgree.csv",
    "FiQA-2018 test": "fiqa_test.csv",
    "NWGI test": "nwgi_test.csv",
}
FOLDERS = [  # (row name in results, folder)
    ("sentiment-bge-small-en-v1.5-int8", "sentiment-bge-small-en-v1.5-int8"),
    ("sentiment-bge-small-en-v1.5", "sentiment-bge-small-en-v1.5"),
    ("sentiment-finbert", "sentiment-finbert"),
    ("sentiment-distil-bge-small-en-v1.5", "sentiment-distil-bge-small-en-v1.5"),
    ("sentiment-teacher-deberta-v3-base", "sentiment-teacher-deberta-v3-base"),
    ("multihead-bge-small-en-v1.5 (sentiment head)", "multihead-bge-small-en-v1.5"),
]
FIQA_LABELS = {0: "positive", 1: "neutral", 2: "negative"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dir(root: Path, name: str) -> Path:
    hits = sorted(p for p in root.rglob(name) if p.is_dir() and (p / "config.json").exists())
    if not hits:
        raise FileNotFoundError(f"model folder {name} not found under {root}")
    return hits[0]


def find_file(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def load_tests(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name, file in TESTS.items():
        df = pd.read_csv(find_file(root, file))
        if file.startswith("fiqa"):
            df = pd.DataFrame({"text": df["sentence"], "label": df["label"].map(FIQA_LABELS)})
        elif file.startswith("nwgi"):
            df = pd.DataFrame({"text": df["news"], "label": df["label"].str.split().str[-1]})
        out[name] = df
    return out


def batches(tok, texts, device):
    for i in range(0, len(texts), BATCH):
        yield tok(list(texts[i : i + BATCH]), padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)  # fmt: skip


def predictor(folder: Path, device: str):
    """(predict(texts) -> labels, where it ran)."""
    tok = AutoTokenizer.from_pretrained(folder)
    if (folder / "model.onnx").exists():
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            import subprocess
            import sys

            pip = [sys.executable, "-m", "pip", "install", "-q", "onnxruntime"]
            subprocess.run(pip, check=True)
            import onnxruntime as ort
        sess = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        names = {i.name for i in sess.get_inputs()}
        id2label = AutoConfig.from_pretrained(folder).id2label
        labels = [id2label[i].lower() for i in range(len(id2label))]

        def run(texts):
            out = []
            for b in batches(tok, texts, "cpu"):
                logits = sess.run(None, {k: v.numpy() for k, v in b.items() if k in names})[0]
                out += [labels[i] for i in logits.argmax(-1)]
            return out

        return run, "int8 ONNX, CPU"
    if (folder / "heads.pt").exists():  # E10 shared encoder: sentiment head only
        enc = AutoModel.from_pretrained(folder).float().eval().to(device)
        names = json.loads((folder / "heads.json").read_text())
        labels = [n.lower() for n in names["sentiment"]]
        head = torch.nn.Linear(enc.config.hidden_size, len(labels))
        head.load_state_dict(torch.load(folder / "heads.pt", map_location="cpu")["sentiment"])
        head = head.eval().to(device)

        def run(texts):
            out = []
            for b in batches(tok, texts, device):
                x = enc(**b).last_hidden_state[:, 0]
                out += [labels[i] for i in head(x).argmax(-1).tolist()]
            return out

        return run, f"fp32, {device}"
    model = AutoModelForSequenceClassification.from_pretrained(folder).float().eval().to(device)
    labels = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]

    def run(texts):
        out = []
        for b in batches(tok, texts, device):
            out += [labels[i] for i in model(**b).logits.argmax(-1).tolist()]
        return out

    return run, f"fp32, {device}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", torch.cuda.get_device_name() if device == "cuda" else "cpu", flush=True)
    expected = json.loads(find_file(args.root, "expected_hashes.json").read_text())
    tests = load_tests(args.root)
    print({k: len(v) for k, v in tests.items()}, flush=True)

    folders = {row: find_dir(args.root, f) for row, f in FOLDERS}
    for key, want in expected.items():  # the exact weights the repo scored, or stop
        folder, file = key.split("/")
        got = sha256(folders[next(r for r, f in FOLDERS if f == folder)] / file)
        if got != want:
            raise SystemExit(f"{key}: sha256 {got} != expected {want}")
    print("all model hashes match", flush=True)

    rows, meta = [], {"device": device, "models": {}}
    plan = [(row, folder) for row, folder in folders.items()] + [("finbert (baseline)", None)]
    for row, folder in plan:
        run, where = predictor(folder or Path("ProsusAI/finbert"), device)
        t0 = time.time()
        for name, df in tests.items():
            with torch.inference_mode():  # the model and the call both outside autograd
                pred = run(df["text"].astype(str).tolist())
            rows.append(pd.DataFrame({"model": row, "test_set": name, "row": np.arange(len(df)),
                                      "label": df["label"], "pred": pred}))  # fmt: skip
        meta["models"][row] = {"ran_on": where, "seconds": round(time.time() - t0, 1)}
        print(row, meta["models"][row], flush=True)
        del run
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.out / "e18_predictions.csv.gz", index=False)
    (args.out / "e18_predictions_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
