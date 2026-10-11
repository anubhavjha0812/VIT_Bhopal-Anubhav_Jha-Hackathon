"""Nine learned event classes (rules: docs/DECISIONS.md, E16).

Self-contained: runs on Kaggle as a script kernel, or anywhere with a GPU:
    python train_events.py              # both candidates
    python train_events.py --limit 300  # quick smoke test

Two candidates on the same leak-free data (E10 filter) and the E5 recipe:
  (A) events9: a 9-class head trained on the E16 classes
  (B) topic20: a 20-topic head, mapped to the E16 classes after prediction
The higher dev 9-class macro-F1 wins; within 0.01, (A). The E2 test file is read for its
text column only, to drop leaked training texts.
Outputs: <out>/events9-bge-small-en-v1.5/, <out>/topic20-bge-small-en-v1.5/ (model, tokenizer,
metrics.json) and <out>/summary.json with the choice.
"""

import argparse
import json
import random
import re
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

TOPIC = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic/resolve/main/{}"
TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News",
    "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
    "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
    "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]  # fmt: skip
# E16 mapping, fixed before training (same table as riskengine.nlp.events.TOPIC_TO_EVENT)
TOPIC_TO_EVENT = {
    "Politics": "Geopolitical",
    "Fed | Central Banks": "Macroeconomic",
    "Macro": "Macroeconomic",
    "Treasuries | Corporate Debt": "Macroeconomic",
    "Energy | Oil": "Macroeconomic",
    "Gold | Metals | Materials": "Macroeconomic",
    "Currencies": "Macroeconomic",
    "M&A | Investments": "M&A",
    "IPO": "M&A",
    "Company | Product News": "Product news",
    "Earnings": "Earnings",
    "Dividend": "Earnings",
    "Analyst Update": "Analyst rating",
    "Legal | Regulation": "Legal/Regulatory",
    "Personnel Change": "Management change",
    "General News | Opinion": "Other",
    "Markets": "Other",
    "Stock Commentary": "Other",
    "Stock Movement": "Other",
    "Financials": "Other",
}
EVENTS = [
    "Geopolitical", "Macroeconomic", "M&A", "Product news", "Earnings", "Analyst rating",
    "Legal/Regulatory", "Management change", "Other",
]  # fmt: skip

BACKBONE = "BAAI/bge-small-en-v1.5"
SEED, MAX_LEN, BATCH, EPOCHS, LR, WARMUP, DEV_FRAC = 42, 64, 32, 4, 5e-5, 0.1, 0.1
TIE = 0.01


def norm(text: str) -> str:
    text = re.sub(r"https?://\S+", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def drop_leaks(df: pd.DataFrame, test_url: str, counts: dict) -> pd.DataFrame:
    """Drop texts matching a test text, texts whose copies disagree, then duplicates (E10)."""
    test_texts = set(pd.read_csv(test_url, usecols=["text"])["text"].map(norm))
    key = df["text"].map(norm)
    leaked = key.isin(test_texts)
    counts["dropped_test_leak"] = df[leaked]["source"].value_counts().to_dict()
    df, key = df[~leaked], key[~leaked]
    conflict = df.groupby(key)["name"].transform("nunique") > 1
    counts["dropped_label_conflict"] = df[conflict]["source"].value_counts().to_dict()
    df, key = df[~conflict], key[~conflict]
    dup = key.duplicated()
    counts["dropped_duplicates"] = df[dup]["source"].value_counts().to_dict()
    return df[~dup].reset_index(drop=True)


def load_topic(limit: int | None) -> tuple[pd.DataFrame, dict]:
    """topic_train.csv with the E5 dev split, E2-test leaks removed (as E10)."""
    raw = pd.read_csv(TOPIC.format("topic_train.csv"))
    df = pd.DataFrame({"text": raw["text"], "name": raw["label"].map(dict(enumerate(TOPIC_NAMES)))})
    _, dev = train_test_split(
        np.arange(len(df)), test_size=DEV_FRAC, stratify=df["name"], random_state=SEED
    )
    df["source"] = np.where(np.isin(np.arange(len(df)), dev), "topic_dev", "topic")
    counts = {"rows_in": df["source"].value_counts().to_dict()}
    df = drop_leaks(df, TOPIC.format("topic_valid.csv"), counts)
    if limit:
        df = df.groupby("source").sample(frac=min(1.0, limit / len(df)), random_state=0)
        df = df.reset_index(drop=True)
    counts["rows_used"] = df["source"].value_counts().to_dict()
    df["event"] = df["name"].map(TOPIC_TO_EVENT)
    return df, counts


def batches(enc, labels, idx, size):
    for i in range(0, len(idx), size):
        j = idx[i : i + size]
        yield {k: v[j] for k, v in enc.items()}, labels[j]


@torch.no_grad()
def predict(model, enc, device) -> np.ndarray:
    model.eval()
    n = len(enc["input_ids"])
    out = []
    for b, _ in batches(enc, torch.zeros(n, dtype=torch.long), torch.arange(n), 256):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


def run(name: str, labels: list[str], target: str, df: pd.DataFrame, out_dir: Path) -> dict:
    """Train one candidate; keep the epoch with the best dev 9-class macro-F1."""
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = np.flatnonzero(df["source"] != "topic_dev")
    dev = np.flatnonzero(df["source"] == "topic_dev")
    y = df[target].map({n: i for i, n in enumerate(labels)}).to_numpy()
    dev_events = df["event"].iloc[dev].to_numpy()

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={n: i for i, n in enumerate(labels)},
    ).to(device)

    def encode(ix):
        return tok(df["text"].iloc[ix].tolist(), padding="max_length", truncation=True,
                   max_length=MAX_LEN, return_tensors="pt")  # fmt: skip

    enc_tr, enc_dev = encode(tr), encode(dev)
    y_tr = torch.tensor(y[tr])

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(tr) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    save_to = out_dir / f"{name}-{BACKBONE.split('/')[-1]}"
    history, best, t0 = [], -1.0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr))
        for b, lab in batches(enc_tr, y_tr, perm, BATCH):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = model(**{k: v.to(device) for k, v in b.items()}, labels=lab.to(device)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        pred = [labels[i] for i in predict(model, enc_dev, device)]
        pred_events = [TOPIC_TO_EVENT.get(p, p) for p in pred]  # identity for the 9-class head
        f1 = f1_score(dev_events, pred_events, labels=EVENTS, average="macro", zero_division=0)
        per_class = f1_score(dev_events, pred_events, labels=EVENTS, average=None, zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "dev_events9_macro_f1": round(float(f1), 4),
                "dev_events9_f1": dict(zip(EVENTS, np.round(per_class, 4).tolist(), strict=True)),
            }
        )
        print(f"{name} epoch {epoch}: dev 9-class macro-F1 {f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(save_to)
            tok.save_pretrained(save_to)

    metrics = {
        "candidate": name,
        "backbone": BACKBONE,
        "labels": labels,
        "train_rows": len(tr),
        "dev_rows": len(dev),
        "best_dev_events9_macro_f1": round(float(best), 4),
        "history": history,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "settings": {
            "seed": SEED,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "epochs": EPOCHS,
            "lr": LR,
            "warmup": WARMUP,
            "dev_frac": DEV_FRAC,
        },  # fmt: skip
    }
    (save_to / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/models"))
    p.add_argument("--limit", type=int, help="use only about N rows (smoke test)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, counts = load_topic(args.limit)
    print("data:", json.dumps(counts), flush=True)
    a = run("events9", EVENTS, "event", df, args.out)
    b = run("topic20", TOPIC_NAMES, "name", df, args.out)
    fa, fb = a["best_dev_events9_macro_f1"], b["best_dev_events9_macro_f1"]
    chosen = "events9" if fa >= fb - TIE else "topic20"
    summary = {
        "data": counts,
        "dev_events9_macro_f1": {"events9": fa, "topic20": fb},
        "rule": f"higher dev 9-class macro-F1; within {TIE}, events9",
        "chosen": chosen,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
