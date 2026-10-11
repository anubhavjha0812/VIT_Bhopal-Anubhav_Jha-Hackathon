"""Sentiment for the E17 history headlines (rules: docs/DECISIONS.md, E17).

Uses the engine's fine-tuned bge-small sentiment model. With a GPU it runs the full-precision
model on the GPU (minutes); without one it runs the engine's int8 ONNX copy on the CPU (about an
hour). Either way it also scores a random 5,000 headlines with the int8 copy, so the two can be
compared. Score = P(positive) - P(negative), as in the engine.

    python score_history.py [--texts tweet_texts.csv.gz] [--root DIR] [--out DIR]
The texts file (text_id, text) and the sentiment-fp32/ and sentiment-int8/ model folders are
found anywhere under ROOT (on Kaggle: the attached private datasets). Output: <name>_scores.csv.gz.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN, CHECK, SEED = 64, 5000, 0


def labels_of(folder: Path) -> tuple[int, int]:
    id2label = AutoConfig.from_pretrained(folder).id2label
    names = [id2label[i].lower() for i in range(len(id2label))]
    return names.index("positive"), names.index("negative")


def score_torch(folder: Path, texts: list[str], device: str, batch: int = 512) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).to(device).eval()
    pos, neg = labels_of(folder)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)  # fmt: skip
            p = model(**enc).logits.float().softmax(-1)
            out.append((p[:, pos] - p[:, neg]).cpu().numpy())
    return np.concatenate(out)


def score_int8(folder: Path, texts: list[str], batch: int = 64) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:  # not in the Kaggle image
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
        import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(folder)
    sess = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    pos, neg = labels_of(folder)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="np")  # fmt: skip
        logits = sess.run(None, {k: v for k, v in enc.items() if k in names})[0]
        p = torch.from_numpy(logits).softmax(-1).numpy()
        out.append(p[:, pos] - p[:, neg])
    return np.concatenate(out)


def find(root: Path, name: str, parent: str | None = None) -> Path:
    """First file called `name` (or `name` without .gz) under root, inside a folder `parent`."""
    names = {name, name.removesuffix(".gz")}
    hits = sorted(
        p for p in root.rglob("*")
        if p.name in names and (parent is None or parent in {q.name for q in p.parents})
    )  # fmt: skip
    if not hits:
        raise FileNotFoundError(f"{name} (in {parent}) not found under {root}")
    return hits[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--texts", default=None, help="default: tweets if attached, else history")
    p.add_argument("--root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--limit", type=int, help="first N headlines only (smoke test)")
    args = p.parse_args()

    for f in sorted(args.root.rglob("*"))[:60]:  # what the attached datasets look like
        print("input:", f.relative_to(args.root), flush=True)
    tweets = any(p.name.startswith("tweet_texts") for p in args.root.rglob("*"))
    name = args.texts or ("tweet_texts.csv.gz" if tweets else "history_texts.csv.gz")
    fp32 = find(args.root, "config.json", "sentiment-fp32").parent
    int8 = find(args.root, "model.onnx", "sentiment-int8").parent
    df = pd.read_csv(find(args.root, name))
    if args.limit:
        df = df.head(args.limit)
    texts = df["text"].astype(str).tolist()
    t0 = time.time()
    gpu = torch.cuda.is_available()
    how = f"fp32 on {torch.cuda.get_device_name()}" if gpu else "int8 ONNX on CPU"
    print(f"scoring {len(texts)} headlines with {how}", flush=True)  # shows in the log at once
    if gpu:
        df["sentiment"] = score_torch(fp32, texts, "cuda")
    else:
        df["sentiment"] = score_int8(int8, texts)
    minutes = (time.time() - t0) / 60
    args.out.mkdir(parents=True, exist_ok=True)
    stem = name.removesuffix("_texts.csv.gz")
    df[["text_id", "sentiment"]].round(4).to_csv(args.out / f"{stem}_scores.csv.gz", index=False)

    check = df.sample(min(CHECK, len(df)), random_state=SEED)
    check_int8 = score_int8(int8, check["text"].astype(str).tolist())
    main_s = check["sentiment"].to_numpy()
    sign = lambda x: np.sign(np.where(np.abs(x) < 1e-9, 0, x))  # noqa: E731
    meta = {
        "texts_file": name,
        "headlines": len(df),
        "scored_with": how,
        "minutes": round(minutes, 1),
        "check_vs_int8": {
            "n": len(check),
            "pearson": round(float(np.corrcoef(main_s, check_int8)[0, 1]), 4),
            "mean_abs_diff": round(float(np.abs(main_s - check_int8).mean()), 4),
            "same_sign": round(float((sign(main_s) == sign(check_int8)).mean()), 4),
        },
    }
    (args.out / f"{stem}_scores_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
