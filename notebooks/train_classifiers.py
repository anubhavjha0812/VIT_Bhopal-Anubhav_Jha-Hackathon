"""Fine-tune sentiment and topic classifiers on a Kaggle GPU (rules: docs/DECISIONS.md, E5).

Self-contained: runs on Kaggle as a script kernel, or anywhere with a GPU:
    python train_classifiers.py                      # all 4 runs
    python train_classifiers.py --task sentiment --backbone BAAI/bge-small-en-v1.5

Only the *train* files are downloaded. The test (valid) files are never touched here.
Outputs, per run: <out>/<task>-<backbone>/ with model, tokenizer and metrics.json.
"""

import argparse
import json
import random
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

HF = "https://huggingface.co/datasets/zeroshot/{repo}/resolve/main/{file}"
TASKS = {
    "sentiment": {
        "url": HF.format(repo="twitter-financial-news-sentiment", file="sent_train.csv"),
        "names": {0: "negative", 1: "positive", 2: "neutral"},  # Bearish, Bullish, Neutral
    },
    "topic": {
        "url": HF.format(repo="twitter-financial-news-topic", file="topic_train.csv"),
        "names": dict(
            enumerate(
                [
                    "Analyst Update",
                    "Fed | Central Banks",
                    "Company | Product News",
                    "Treasuries | Corporate Debt",
                    "Dividend",
                    "Earnings",
                    "Energy | Oil",
                    "Financials",
                    "Currencies",
                    "General News | Opinion",
                    "Gold | Metals | Materials",
                    "IPO",
                    "Legal | Regulation",
                    "M&A | Investments",
                    "Macro",
                    "Markets",
                    "Politics",
                    "Personnel Change",
                    "Stock Commentary",
                    "Stock Movement",
                ]
            )
        ),  # fmt: skip
    },
}
LR = {"ProsusAI/finbert": 3e-5, "BAAI/bge-small-en-v1.5": 5e-5}
SEED, MAX_LEN, BATCH, EPOCHS, WARMUP, DEV_FRAC = 42, 64, 32, 4, 0.1, 0.1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_order(backbone: str, names: list[str]) -> list[str]:
    """Reuse the backbone's own head order if it already has these labels (FinBERT sentiment)."""
    from transformers import AutoConfig

    id2label = AutoConfig.from_pretrained(backbone).id2label or {}
    existing = [v.lower() for _, v in sorted(id2label.items())]
    return existing if sorted(existing) == sorted(n.lower() for n in names) else names


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


def run(task: str, backbone: str, out_dir: Path, limit: int | None = None) -> dict:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(TASKS[task]["url"])
    if limit:  # quick smoke test only
        df = df.sample(limit, random_state=SEED).reset_index(drop=True)
    df["name"] = df["label"].map(TASKS[task]["names"])

    order = label_order(backbone, list(TASKS[task]["names"].values()))
    to_id = {n.lower(): i for i, n in enumerate(order)}
    y = df["name"].str.lower().map(to_id).to_numpy()
    tr, dev = train_test_split(
        np.arange(len(df)), test_size=DEV_FRAC, stratify=y, random_state=SEED
    )

    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=len(order),
        id2label=dict(enumerate(order)),
        label2id={n: i for i, n in enumerate(order)},
        ignore_mismatched_sizes=True,
    ).to(device)

    def encode(ix):
        return tok(df["text"].iloc[ix].tolist(), padding="max_length", truncation=True,
                   max_length=MAX_LEN, return_tensors="pt")  # fmt: skip

    enc_tr, enc_dev = encode(tr), encode(dev)
    y_tr, y_dev = torch.tensor(y[tr]), y[dev]

    opt = torch.optim.AdamW(model.parameters(), lr=LR[backbone], weight_decay=0.01)
    steps = EPOCHS * ((len(tr) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    save_to = out_dir / f"{task}-{backbone.split('/')[-1]}"
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
        dev_f1 = f1_score(y_dev, predict(model, enc_dev, device), average="macro")
        history.append({"epoch": epoch, "dev_macro_f1": round(float(dev_f1), 4)})
        print(f"{task} {backbone} epoch {epoch}: dev macro-F1 {dev_f1:.4f}", flush=True)
        if dev_f1 > best:
            best = dev_f1
            model.save_pretrained(save_to)
            tok.save_pretrained(save_to)

    metrics = {
        "task": task,
        "backbone": backbone,
        "train_rows": len(tr),
        "dev_rows": len(dev),
        "best_dev_macro_f1": round(float(best), 4),
        "history": history,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "settings": {
            "seed": SEED,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "epochs": EPOCHS,
            "lr": LR[backbone],
            "warmup": WARMUP,
            "dev_frac": DEV_FRAC,
        },  # fmt: skip
    }
    (save_to / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=[*TASKS, "all"], default="all")
    p.add_argument("--backbone", choices=[*LR, "all"], default="all")
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/models"))
    p.add_argument("--limit", type=int, help="use only N train rows (smoke test)")
    args = p.parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    backbones = list(LR) if args.backbone == "all" else [args.backbone]
    summary = [run(t, b, args.out, args.limit) for t in tasks for b in backbones]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    for m in summary:
        print(m["task"], m["backbone"], "best dev macro-F1", m["best_dev_macro_f1"])


if __name__ == "__main__":
    main()
