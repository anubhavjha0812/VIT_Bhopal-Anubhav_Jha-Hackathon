"""E25 student: a bge-small event model trained on two teachers' agreed labels.

Design: docs/E25_TEACHER_STUDENT.md. Input (private Kaggle dataset): train.csv and valid.csv from
`python -m riskengine.eval.e25_student_data` (columns id, text, label, source and p_<class> for the
9 learned classes). Twitter Financial News Topic (TFN) tweets for all 9 classes are mixed in
here, so tweet style is not a hint for any class (hard labels, at most TFN_CAP per class, train
rows only, leaks dropped as in notebooks/events9/train_events.py). Never reads D1, D2, E23 or
the finals.

Loss = class-weighted (cross-entropy to the hard label + KL to the teachers' soft label). Weights
are 1 / sqrt(class count). Seeds 42, 43, 44; each run keeps its best epoch by macro-F1 on valid.csv
(the pool validation slice only); the best run overall is exported to int8 ONNX.
Output: /kaggle/working/models/e25-student-bge-small-en-v1.5[-int8] and summary.json.
"""

import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BACKBONE = "BAAI/bge-small-en-v1.5"
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
]  # fmt: skip  same order as riskengine.nlp.events.LEARNED_CLASSES
MAX_LEN, BATCH, LR, EPOCHS, WARMUP = 64, 32, 5e-5, 4, 0.1
SEEDS = [42, 43, 44]
TFN_CAP, TFN_SEED, TFN_DEV_FRAC = 600, 25, 0.1
OPSET = 17
OUT = Path("/kaggle/working")
TOPIC = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic/resolve/main/{}"
TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News",
    "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
    "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
    "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]  # fmt: skip
TOPIC_TO_EVENT = {  # the E16 table (riskengine.nlp.events.TOPIC_TO_EVENT)
    "Politics": "Geopolitical",
    "Fed | Central Banks": "Macroeconomic", "Macro": "Macroeconomic",
    "Treasuries | Corporate Debt": "Macroeconomic", "Energy | Oil": "Macroeconomic",
    "Gold | Metals | Materials": "Macroeconomic", "Currencies": "Macroeconomic",
    "M&A | Investments": "M&A", "IPO": "M&A", "Company | Product News": "Product news",
    "Earnings": "Earnings", "Dividend": "Earnings", "Analyst Update": "Analyst rating",
    "Legal | Regulation": "Legal/Regulatory", "Personnel Change": "Management change",
    "General News | Opinion": "Other", "Markets": "Other", "Stock Commentary": "Other",
    "Stock Movement": "Other", "Financials": "Other",
}  # fmt: skip


def find(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm(text: str) -> str:
    text = re.sub(r"https?://\S+", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tfn_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """TFN train rows for every class (capped) and the TFN dev rows (report only).

    Same E5 dev split and leak drop as notebooks/events9/train_events.py; topic_valid is only used
    to drop texts that match it."""
    raw = pd.read_csv(TOPIC.format("topic_train.csv"))
    df = pd.DataFrame({"text": raw["text"], "name": raw["label"].map(dict(enumerate(TOPIC_NAMES)))})
    _, dev = train_test_split(
        np.arange(len(df)), test_size=TFN_DEV_FRAC, stratify=df["name"], random_state=42
    )
    df["is_dev"] = np.isin(np.arange(len(df)), dev)
    test = set(pd.read_csv(TOPIC.format("topic_valid.csv"), usecols=["text"])["text"].map(norm))
    key = df["text"].map(norm)
    df = df[~key.isin(test)]
    key = key[~key.isin(test)]
    df = df[~df.groupby(key)["name"].transform("nunique").gt(1)]
    df = df[~key[df.index].duplicated()].reset_index(drop=True)
    df["label"] = df["name"].map(TOPIC_TO_EVENT)
    train = df[~df["is_dev"]]
    train = (
        train.groupby("label", group_keys=False)
        .sample(n=None, frac=1.0, random_state=TFN_SEED)
        .groupby("label")
        .head(TFN_CAP)
    )
    info = {"tfn_train": train["label"].value_counts().to_dict(),
            "tfn_dev": df[df["is_dev"]]["label"].value_counts().to_dict()}  # fmt: skip
    train = train.assign(source="tfn", id="tfn")[["id", "text", "label", "source"]]
    dev_rows = df[df["is_dev"]].assign(source="tfn_dev", id="tfn_dev")
    return train.reset_index(drop=True), dev_rows[["id", "text", "label", "source"]], info


def soft_matrix(df: pd.DataFrame) -> np.ndarray:
    """Teacher soft labels where present; a one-hot vector for the hard-label TFN rows."""
    p = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    cols = [f"p_{c}" for c in CLASSES]
    has = df[cols].notna().all(axis=1).to_numpy() if cols[0] in df else np.zeros(len(df), bool)
    if has.any():
        p[has] = df.loc[has, cols].to_numpy(dtype=np.float32)
    for i in np.flatnonzero(~has):
        p[i, CLASSES.index(df["label"].iloc[i])] = 1.0
    return p


def encode(tok, texts):
    return tok(list(texts), padding="max_length", truncation=True, max_length=MAX_LEN,
               return_tensors="pt")  # fmt: skip


@torch.no_grad()
def predict(model, enc, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(enc["input_ids"]), 256):
        b = {k: v[i : i + 256].to(device) for k, v in enc.items()}
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**b).logits.float().argmax(-1).cpu())
    return torch.cat(out).numpy()


def macro(y, p) -> float:
    present = sorted(set(y.tolist()))  # classes with at least one validation row
    return float(f1_score(y, p, labels=present, average="macro", zero_division=0))


def per_class(y, p) -> dict:
    out = {}
    for i, c in enumerate(CLASSES):
        tp = int(((y == i) & (p == i)).sum())
        n, k = int((y == i).sum()), int((p == i).sum())
        out[c] = {"support": n, "predicted": k,
                  "precision": round(tp / k, 4) if k else None,
                  "recall": round(tp / n, 4) if n else None}  # fmt: skip
    return out


def run(seed, data, device):
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)},
    ).to(device)  # fmt: skip
    enc, y, soft, cw = data["enc"], data["y"], data["soft"], data["class_w"]
    n = len(y)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * ((n + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    g = torch.Generator().manual_seed(seed)
    best, history, state = None, [], None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = order[i : i + BATCH]
            b = {k: v[j].to(device) for k, v in enc.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**b).logits.float()
            yb, sb = y[j].to(device), soft[j].to(device)
            logp = torch.log_softmax(logits, -1)
            ce = torch.nn.functional.nll_loss(logp, yb, reduction="none")
            kl = torch.nn.functional.kl_div(logp, sb, reduction="none").sum(-1)
            w = cw.to(device)[yb]
            loss = ((ce + kl) * w).sum() / w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        pv = predict(model, data["valid_enc"], device)
        score = macro(data["valid_y"], pv)
        acc = float((pv == data["valid_y"]).mean())
        history.append({"epoch": epoch, "valid_macro_f1": round(score, 4),
                        "valid_acc": round(acc, 4)})  # fmt: skip
        print("seed", seed, history[-1], flush=True)
        if best is None or score > best["valid_macro_f1"]:
            best = history[-1] | {}
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"seed": seed, "best": best, "history": history}


def export_int8(model_dir: Path, out: Path) -> Path:
    """Same export as riskengine.nlp.onnx_int8.export_int8 (ONNX opset 17, int8 weights)."""
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"],
                       check=True)  # fmt: skip
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().eval()
    model.config.return_dict = False
    sample = tok(["a sample headline"], return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]
    axes = {k: {0: "batch", 1: "tokens"} for k in names}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        torch.onnx.export(
            model, tuple(sample[k] for k in names), str(fp32), input_names=names,
            output_names=["logits"], dynamic_axes={**axes, "logits": {0: "batch"}},
            opset_version=OPSET, dynamo=False,
        )  # fmt: skip
        quantize_dynamic(str(fp32), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(out)
    model.config.return_dict = True
    model.config.save_pretrained(out)
    if (model_dir / "metrics.json").exists():
        shutil.copy(model_dir / "metrics.json", out / "source_metrics.json")
    return out / "model.onnx"


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name() if device.type == "cuda" else "cpu")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    train, valid = pd.read_csv(find("train.csv")), pd.read_csv(find("valid.csv"))
    tfn_train, tfn_dev, tfn_info = tfn_rows()
    train = pd.concat([train, tfn_train], ignore_index=True)
    lab = {c: i for i, c in enumerate(CLASSES)}
    y = torch.tensor(train["label"].map(lab).to_numpy())
    counts = np.bincount(y.numpy(), minlength=len(CLASSES)).astype(np.float64)
    cw = 1 / np.sqrt(np.maximum(counts, 1))
    cw = torch.tensor(cw * len(y) / (cw[y.numpy()].sum()), dtype=torch.float)  # mean weight 1
    data = {
        "enc": encode(tok, train["text"].astype(str)), "y": y,
        "soft": torch.tensor(soft_matrix(train)), "class_w": cw,
        "valid_enc": encode(tok, valid["text"].astype(str)),
        "valid_y": valid["label"].map(lab).to_numpy(),
    }  # fmt: skip
    rows = {"train": len(train), "valid": len(valid),
            "train_by_class": train["label"].value_counts().to_dict(),
            "train_by_source": train["source"].value_counts().to_dict(), **tfn_info}  # fmt: skip
    print(rows, flush=True)
    runs, best_model, best_run = [], None, None
    for seed in SEEDS:
        model, info = run(seed, data, device)
        runs.append(info)
        if best_run is None or info["best"]["valid_macro_f1"] > best_run["best"]["valid_macro_f1"]:
            best_run, best_model = info, model.cpu()
        else:
            del model
        torch.cuda.empty_cache()
    fp32 = OUT / "models" / "e25-student-bge-small-en-v1.5"
    fp32.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(fp32)
    tok.save_pretrained(fp32)
    (fp32 / "metrics.json").write_text(json.dumps(best_run, indent=2))
    # reports: pool validation per class and the TFN dev rows (report only, never used to choose)
    best_model.to(device)
    pv = predict(best_model, data["valid_enc"], device)
    dev_y = tfn_dev["label"].map(lab).to_numpy()
    pd_ = predict(best_model, encode(tok, tfn_dev["text"].astype(str)), device)
    reports = {
        "valid_per_class": per_class(data["valid_y"], pv),
        "tfn_dev_per_class": per_class(dev_y, pd_),
        "tfn_dev_macro_f1_9": round(float(f1_score(dev_y, pd_, labels=list(range(len(CLASSES))),
                                                   average="macro", zero_division=0)), 4),
    }  # fmt: skip
    int8 = export_int8(fp32, OUT / "models" / "e25-student-bge-small-en-v1.5-int8")
    summary = {"chosen": best_run, "runs": runs, "rows": rows, "reports": reports,
               "int8_mb": round(int8.stat().st_size / 1e6, 1),
               "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"chosen": best_run["seed"], "best": best_run["best"],
                      "minutes": summary["minutes"], "int8_mb": summary["int8_mb"]}))  # fmt: skip


if __name__ == "__main__":
    main()
