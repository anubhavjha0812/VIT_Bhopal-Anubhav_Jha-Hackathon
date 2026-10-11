"""E21: one bge-small sentiment model for tweets and news (rules: docs/DECISIONS.md, E21).

Trains on the leak-filtered rows prepared by `python -m riskengine.eval.e21 --build-training`
(train.csv, dev_tweets.csv, dev_news.csv in the attached private dataset). Two weightings
(C1: all rows weight 1; C2: TFNS rows weight 2) x seeds 42, 43, 44, E5 recipe. Each run keeps
its best epoch by dev score = mean of tweet-dev and news-dev macro-F1; the best run overall is
saved to /kaggle/working/models/sentiment-e21-bge-small-en-v1.5 with summary.json.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BACKBONE = "BAAI/bge-small-en-v1.5"
NAMES = ["negative", "positive", "neutral"]  # same order as the E5 models
MAX_LEN, BATCH, LR, EPOCHS, WARMUP = 64, 32, 5e-5, 4, 0.1
SEEDS = [42, 43, 44]
WEIGHTINGS = {"C1": 1.0, "C2": 2.0}  # loss weight of TFNS rows
OUT = Path("/kaggle/working")


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


def encode(tok, texts):
    return tok(list(texts), padding="max_length", truncation=True, max_length=MAX_LEN,
               return_tensors="pt")  # fmt: skip


@torch.no_grad()
def predict(model, enc, device) -> np.ndarray:
    model.eval()
    out = []
    n = len(enc["input_ids"])
    for i in range(0, n, 256):
        b = {k: v[i : i + 256].to(device) for k, v in enc.items()}
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**b).logits.float().argmax(-1).cpu())
    return torch.cat(out).numpy()


def macro(y, p) -> float:
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def run(name, tfns_weight, seed, data, device, tok):
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=3, id2label=dict(enumerate(NAMES)),
        label2id={n: i for i, n in enumerate(NAMES)},
    ).to(device)  # fmt: skip
    train, enc, y, w = data["train"], data["enc"], data["y"], data["w"][tfns_weight]
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    best, history = None, []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train), generator=g)
        for i in range(0, len(order), BATCH):
            j = order[i : i + BATCH]
            b = {k: v[j].to(device) for k, v in enc.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**b).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y[j].to(device), reduction="none")
            loss = (loss * w[j].to(device)).sum() / w[j].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        tw = macro(data["dev_tweets_y"], predict(model, data["dev_tweets_enc"], device))
        nw = macro(data["dev_news_y"], predict(model, data["dev_news_enc"], device))
        score = (tw + nw) / 2
        history.append({"epoch": epoch, "dev_tweets": tw, "dev_news": nw, "dev_score": score})
        print(name, seed, history[-1], flush=True)
        if best is None or score > best["dev_score"]:
            best = {"epoch": epoch, "dev_tweets": tw, "dev_news": nw, "dev_score": score}
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"weighting": name, "seed": seed, "best": best, "history": history}


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name() if device.type == "cuda" else "cpu")
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    lab = {n: i for i, n in enumerate(NAMES)}
    train = pd.read_csv(find("train.csv"))
    dev_t, dev_n = pd.read_csv(find("dev_tweets.csv")), pd.read_csv(find("dev_news.csv"))
    data = {
        "train": train,
        "enc": encode(tok, train["text"].astype(str)),
        "y": torch.tensor(train["label"].map(lab).to_numpy()),
        "w": {
            wt: torch.tensor(np.where(train["source"] == "tfns", wt, 1.0), dtype=torch.float)
            for wt in set(WEIGHTINGS.values())
        },  # fmt: skip
        "dev_tweets_enc": encode(tok, dev_t["text"].astype(str)),
        "dev_tweets_y": dev_t["label"].map(lab).to_numpy(),
        "dev_news_enc": encode(tok, dev_n["text"].astype(str)),
        "dev_news_y": dev_n["label"].map(lab).to_numpy(),
    }
    print({"train": len(train), "dev_tweets": len(dev_t), "dev_news": len(dev_n)}, flush=True)
    runs, best_model, best_run = [], None, None
    for name, wt in WEIGHTINGS.items():
        for seed in SEEDS:
            model, info = run(name, wt, seed, data, device, tok)
            runs.append(info)
            if best_run is None or info["best"]["dev_score"] > best_run["best"]["dev_score"]:
                best_run, best_model = info, model.cpu()
            else:
                del model
            torch.cuda.empty_cache()
    out = OUT / "models" / "sentiment-e21-bge-small-en-v1.5"
    out.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(out)
    tok.save_pretrained(out)
    rows = train["source"].value_counts().to_dict()
    minutes = round((time.time() - t0) / 60, 1)
    summary = {"chosen": best_run, "runs": runs, "train_rows": rows, "minutes": minutes}
    (out / "metrics.json").write_text(json.dumps(best_run, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"chosen": best_run["weighting"], "seed": best_run["seed"],
                      "best": best_run["best"], "minutes": summary["minutes"]}))  # fmt: skip


if __name__ == "__main__":
    main()
