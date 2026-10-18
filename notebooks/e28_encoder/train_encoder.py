"""E28 G: frozen encoders + train-only TF-IDF + logistic regression (Kaggle script).

Inputs: train.csv, valid.csv and blind eval_texts.csv (id,text,set; D1/D2/R3 only).
No evaluation labels are loaded. Selection and temperature use valid.csv only.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

CLASSES = [
    "M&A",
    "Geopolitical",
    "Macroeconomic",
    "Product news",
    "Earnings",
    "Analyst rating",
    "Legal/Regulatory",
    "Management change",
    "Other",
]
ENCODERS = ["intfloat/multilingual-e5-large", "intfloat/e5-base-v2", "BAAI/bge-small-en-v1.5"]
SEED, MAX_LENGTH = 28, 64
OUT = Path("/kaggle/working/e28_encoder")


def find(name):
    hits = sorted(Path("/kaggle/input").rglob(name))
    if len(hits) != 1:
        raise ValueError(f"need exactly one {name}, found {len(hits)}")
    return hits[0]


def inputs():
    paths = {name: find(name + ".csv") for name in ("train", "valid", "eval_texts")}
    frames = {
        name: pd.read_csv(path, dtype=str, keep_default_na=False) for name, path in paths.items()
    }
    for split in ("train", "valid"):
        frame = frames[split]
        if not {"id", "text", "label"}.issubset(frame) or frame.empty:
            raise ValueError(f"invalid {split} input")
        if not frame["label"].isin(CLASSES).all() or frame["text"].str.strip().eq("").any():
            raise ValueError(f"invalid {split} labels/text")
    if set(frames["train"]["label"]) != set(CLASSES):
        raise ValueError("training must contain all nine learned classes")
    evaluation = frames["eval_texts"]
    if set(evaluation) != {"id", "text", "set"} or evaluation.empty:
        raise ValueError("eval_texts.csv must contain only id,text,set")
    if not evaluation["set"].isin(["D1", "D2", "R3"]).all():
        raise ValueError("only D1/D2/R3; E24 finals are forbidden")
    if evaluation.duplicated(["set", "id"]).any() or evaluation["text"].str.strip().eq("").any():
        raise ValueError("invalid evaluation ids/text")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return frames, hashes


def temperature_scale(probabilities, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, labels):
    result = minimize_scalar(
        lambda t: nll(labels, temperature_scale(probabilities, t)),
        bounds=(0.05, 20),
        method="bounded",
    )
    return float(result.x)


def nll(labels, probabilities):
    # Integer labels preserve the explicit probability-column order in sklearn.
    y = [CLASSES.index(label) for label in labels]
    return float(log_loss(y, probabilities, labels=list(range(len(CLASSES)))))


def ordered_probabilities(model, features):
    p = model.predict_proba(features)
    if set(model.classes_) != set(CLASSES):
        raise ValueError("head is missing a learned class")
    return p[:, [list(model.classes_).index(label) for label in CLASSES]]


def encode(encoder, texts, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    prefix = "query: " if encoder.startswith("intfloat/") else ""
    start, pieces = time.perf_counter(), []
    # Conservative batch size also fits multilingual-e5-large on the usual Kaggle GPU.
    with torch.inference_mode():
        for offset in range(0, len(texts), 16):
            tokens = tokenizer(
                [prefix + text for text in texts[offset : offset + 16]],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            hidden = model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pieces.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    elapsed = time.perf_counter() - start
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(pieces), {
        "seconds": elapsed,
        "seconds_per_1000_texts": elapsed * 1000 / len(texts),
    }


def save_probabilities(folder, evaluation, p, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "probabilities.npz"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("preserve the earlier probability export")
    np.savez_compressed(
        path,
        ids=evaluation["id"].to_numpy(dtype=str),
        sets=evaluation["set"].to_numpy(dtype=str),
        labels=np.array(CLASSES),
        text_sha256=np.array([hashlib.sha256(t.encode()).hexdigest() for t in evaluation["text"]]),
        probabilities=p,
    )
    metadata |= {
        "class_order": CLASSES,
        "calibration_split": "valid",
        "calibrated": True,
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "Exploratory development only; no fresh test or E24 finals.",
        "labels": "AI-derived training labels; evaluation input has no labels.",
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    frames, hashes = inputs()
    train, valid, evaluation = (frames[name] for name in ("train", "valid", "eval_texts"))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=100000, dtype=np.float32
    )
    text_features = {"train": vectorizer.fit_transform(train["text"])}
    text_features |= {
        name: vectorizer.transform(frame["text"])
        for name, frame in frames.items()
        if name != "train"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_texts = pd.concat([train["text"], valid["text"], evaluation["text"]]).tolist()
    for encoder in ENCODERS:
        embeddings, timing = encode(encoder, all_texts, device)
        n, v = len(train), len(valid)
        vectors = {
            "train": embeddings[:n],
            "valid": embeddings[n : n + v],
            "eval_texts": embeddings[n + v :],
        }
        best, head, chosen_x = None, None, None
        start = time.perf_counter()
        for alpha in (0.5, 1.0, 2.0):
            x = {
                name: sparse.hstack([text_features[name], alpha * vector], format="csr")
                for name, vector in vectors.items()
            }
            for c in (0.5, 1.0, 3.0):
                model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=SEED)
                model.fit(x["train"], train["label"])
                p = ordered_probabilities(model, x["valid"])
                loss = nll(valid["label"], p)
                if best is None or loss < best["valid_log_loss_raw"]:
                    best = {"C": c, "alpha": alpha, "valid_log_loss_raw": float(loss)}
                    head, chosen_x = model, x
        valid_p = ordered_probabilities(head, chosen_x["valid"])
        temperature = fit_temperature(valid_p, valid["label"])
        p = temperature_scale(ordered_probabilities(head, chosen_x["eval_texts"]), temperature)
        calibrated = temperature_scale(valid_p, temperature)
        metadata = {
            "variant": "G",
            "encoder": encoder,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "embedding": "attention-mask mean pooling; L2 norm; query: prefix for e5",
            "chosen": best,
            "temperature": temperature,
            "input_sha256": hashes,
            "valid_macro_f1": float(
                f1_score(
                    valid["label"],
                    np.array(CLASSES)[calibrated.argmax(1)],
                    labels=CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "valid_log_loss_calibrated": nll(valid["label"], calibrated),
            "encode_timing": timing,
            "head_and_calibration_seconds": time.perf_counter() - start,
            "machine": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        }
        save_probabilities(OUT / ("G_" + encoder.split("/")[-1]), evaluation, p, metadata)
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
