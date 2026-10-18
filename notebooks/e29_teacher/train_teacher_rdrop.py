"""Sentiment teacher + distilled student (E7) and a shared two-head encoder (E10).

Rules: docs/DECISIONS.md. Self-contained: runs on Kaggle as a script kernel, or anywhere with a GPU:
    python train_teacher.py --stage teacher     # teacher (3 seeds) then student (3 seeds)
    python train_teacher.py --stage multitask   # needs a finished teacher (--teacher or mounted)
    python train_teacher.py --limit 300         # quick smoke test
Default stage (Kaggle runs take no arguments): multitask when a finished teacher is mounted
under /kaggle/input (notebooks/multitask/kernel-metadata.json), otherwise teacher.

The E1/E2 test files are read for their text column only, to drop leaked training texts.
Outputs: <out>/sentiment-teacher-deberta-v3-base/, <out>/sentiment-distil-bge-small-en-v1.5/,
<out>/multihead-bge-small-en-v1.5/ (each with metrics.json) and <out>/summary*.json.
"""

import io
import json
import random
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

TFNS = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/{}"
PHRASEBANK = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/"
    "FinancialPhraseBank-v1.0.zip"
)
FIQA = "https://huggingface.co/api/datasets/pauri32/fiqa-2018/parquet/default/{}/0.parquet"

NAMES = ["negative", "positive", "neutral"]  # same order as the E5 models
TFNS_NAMES = {0: "negative", 1: "positive", 2: "neutral"}  # Bearish, Bullish, Neutral
FIQA_NAMES = {0: "positive", 1: "neutral", 2: "negative"}

TOPIC = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic/resolve/main/{}"
TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News",
    "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
    "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
    "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]  # fmt: skip
MULTIHEAD = "multihead-bge-small-en-v1.5"
TEACHER_DIR = "sentiment-teacher-deberta-v3-base"

TEACHER, STUDENT = "microsoft/deberta-v3-base", "BAAI/bge-small-en-v1.5"
RECIPE = {
    TEACHER: {"lr": 2e-5, "epochs": 4},
    STUDENT: {"lr": 5e-5, "epochs": 6},
}
SEEDS, MAX_LEN, BATCH, WARMUP, DEV_FRAC, SPLIT_SEED = [42, 43, 44], 64, 32, 0.1, 0.1, 42
ALPHA, TEMP = 0.5, 2.0
RDROP_ALPHA = 1.0
EPOCHS_MULTI = 6


def norm(text: str) -> str:
    text = re.sub(r"https?://\S+", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phrasebank() -> pd.DataFrame:
    raw = urllib.request.urlopen(PHRASEBANK).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = z.read("FinancialPhraseBank-v1.0/Sentences_75Agree.txt").decode("latin-1")
    rows = [line.rsplit("@", 1) for line in lines.splitlines() if "@" in line]
    return pd.DataFrame(rows, columns=["text", "name"]).assign(source="phrasebank")


def fiqa() -> pd.DataFrame:
    df = pd.concat(pd.read_parquet(FIQA.format(s)) for s in ["train", "validation"])
    return pd.DataFrame(
        {"text": df["sentence"], "name": df["label"].map(FIQA_NAMES), "source": "fiqa"}
    )


def drop_leaks(df: pd.DataFrame, test_url: str, counts: dict) -> pd.DataFrame:
    """Drop texts matching a test text, texts whose copies disagree, then duplicates."""
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


def load_topic(limit: int | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """E10 topic data: topic_train.csv with the E5 dev split, E2-test leaks removed."""
    raw = pd.read_csv(TOPIC.format("topic_train.csv"))
    df = pd.DataFrame({"text": raw["text"], "name": raw["label"].map(dict(enumerate(TOPIC_NAMES)))})
    _, dev = train_test_split(
        np.arange(len(df)), test_size=DEV_FRAC, stratify=df["name"], random_state=SPLIT_SEED
    )
    df["source"] = np.where(np.isin(np.arange(len(df)), dev), "topic_dev", "topic")
    counts = {"rows_in": df["source"].value_counts().to_dict()}
    df = drop_leaks(df, TOPIC.format("topic_valid.csv"), counts)
    if limit:
        df = df.groupby("source").sample(frac=min(1.0, limit / len(df)), random_state=0)
        df = df.reset_index(drop=True)
    counts["rows_used"] = df["source"].value_counts().to_dict()
    df["y"] = df["name"].map({n: i for i, n in enumerate(TOPIC_NAMES)})
    return (
        df,
        np.flatnonzero(df["source"] != "topic_dev"),
        np.flatnonzero(df["source"] == "topic_dev"),
        counts,
    )


def load_data(limit: int | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Training rows (TFNS + extra) and the E5 dev split, with E1-test leaks removed."""
    tfns = pd.read_csv(TFNS.format("sent_train.csv"))
    tfns = pd.DataFrame({"text": tfns["text"], "name": tfns["label"].map(TFNS_NAMES)})
    # Same dev split as E5: drawn on the full TFNS train file before anything is dropped.
    y = tfns["name"].to_numpy()
    tr, dev = train_test_split(
        np.arange(len(tfns)), test_size=DEV_FRAC, stratify=y, random_state=SPLIT_SEED
    )
    tfns["source"] = np.where(np.isin(np.arange(len(tfns)), dev), "tfns_dev", "tfns")

    df = pd.concat([tfns, phrasebank(), fiqa()], ignore_index=True)
    counts = {"rows_in": df["source"].value_counts().to_dict()}
    # FiQA repeats a sentence once per aspect: copies that disagree are dropped too.
    df = drop_leaks(df, TFNS.format("sent_valid.csv"), counts)
    if limit:  # quick smoke test only
        df = df.groupby("source").sample(frac=min(1.0, limit / len(df)), random_state=0)
        df = df.reset_index(drop=True)
    counts["rows_used"] = df["source"].value_counts().to_dict()
    df["y"] = df["name"].map({n: i for i, n in enumerate(NAMES)})
    dev_idx = np.flatnonzero(df["source"] == "tfns_dev")
    train_idx = np.flatnonzero(df["source"] != "tfns_dev")
    return df, train_idx, dev_idx, counts


def batches(enc, idx, size):
    for i in range(0, len(idx), size):
        j = idx[i : i + size]
        yield {k: v[j] for k, v in enc.items()}, j


@torch.no_grad()
def logits_of(model, enc, device) -> torch.Tensor:
    model.eval()
    n = len(enc["input_ids"])
    out = []
    for b, _ in batches(enc, torch.arange(n), 256):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(**{k: v.to(device) for k, v in b.items()}).logits.float().cpu())
    return torch.cat(out)


def train_one(backbone, seed, df, train_idx, dev_idx, save_to, device, soft=None) -> dict:
    """One seed. `soft` = teacher probabilities at TEMP for every row of df (student only)."""
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(backbone)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=len(NAMES),
            id2label=dict(enumerate(NAMES)),
            label2id={n: i for i, n in enumerate(NAMES)},
            ignore_mismatched_sizes=True,
        )
        .float()
        .to(device)
    )  # some checkpoints load as fp16; AMP needs fp32 weights
    enc = tok(df["text"].tolist(), padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="pt")  # fmt: skip
    enc_dev = {k: v[dev_idx] for k, v in enc.items()}
    y = torch.tensor(df["y"].to_numpy())
    y_dev = df["y"].to_numpy()[dev_idx]

    lr, epochs = RECIPE[backbone]["lr"], RECIPE[backbone]["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_idx) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best, t0 = [], -1.0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.from_numpy(train_idx)[torch.randperm(len(train_idx))]
        for step, (b, j) in enumerate(batches(enc, perm, BATCH), 1):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**{k: v.to(device) for k, v in b.items()}).logits.float()
            loss = F.cross_entropy(logits, y[j].to(device))
            if soft is None and RDROP_ALPHA > 0:  # R-Drop: second dropout pass + symmetric KL
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    logits2 = model(**{k: v.to(device) for k, v in b.items()}).logits.float()
                yb = y[j].to(device)
                ce = 0.5 * (F.cross_entropy(logits, yb) + F.cross_entropy(logits2, yb))
                lp, lq = F.log_softmax(logits, -1), F.log_softmax(logits2, -1)
                kl = 0.5 * (F.kl_div(lp, lq, log_target=True, reduction="batchmean")
                            + F.kl_div(lq, lp, log_target=True, reduction="batchmean"))
                loss = ce + RDROP_ALPHA * kl
            if soft is not None:
                kd = F.kl_div(F.log_softmax(logits / TEMP, -1), soft[j].to(device),
                              reduction="batchmean") * TEMP**2  # fmt: skip
                loss = ALPHA * loss + (1 - ALPHA) * kd
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 50 == 0:
                print(f"  {backbone} seed {seed} epoch {epoch} step {step}: loss {loss.item():.4f}",
                      flush=True)  # fmt: skip
        pred = logits_of(model, enc_dev, device).argmax(-1).numpy()
        dev_f1 = f1_score(y_dev, pred, average="macro")
        history.append({"epoch": epoch, "dev_macro_f1": round(float(dev_f1), 4)})
        print(f"{backbone} seed {seed} epoch {epoch}: dev macro-F1 {dev_f1:.4f}", flush=True)
        if dev_f1 > best:
            best = dev_f1
            model.save_pretrained(save_to)
            tok.save_pretrained(save_to)
    return {
        "backbone": backbone,
        "seed": seed,
        "best_dev_macro_f1": round(float(best), 4),
        "history": history,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_minutes": round((time.time() - t0) / 60, 1),
    }


def train_seeds(backbone, name, df, train_idx, dev_idx, out, device, soft=None) -> dict:
    """Train every seed, keep the best on dev as <out>/<name>, delete the rest."""
    runs = [train_one(backbone, s, df, train_idx, dev_idx, out / f"{name}-seed{s}", device, soft)
            for s in SEEDS]  # fmt: skip
    best = max(runs, key=lambda r: r["best_dev_macro_f1"])
    final = out / name
    shutil.rmtree(final, ignore_errors=True)
    (out / f"{name}-seed{best['seed']}").rename(final)
    for s in SEEDS:
        shutil.rmtree(out / f"{name}-seed{s}", ignore_errors=True)
    metrics = {
        "backbone": backbone,
        "chosen_seed": best["seed"],
        "best_dev_macro_f1": best["best_dev_macro_f1"],
        "seeds": runs,
        "train_rows": len(train_idx),
        "dev_rows": len(dev_idx),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "settings": {"max_len": MAX_LEN, "batch": BATCH, "warmup": WARMUP,
                     **RECIPE[backbone], "distil": None if soft is None
                     else {"alpha": ALPHA, "temperature": TEMP, "teacher": TEACHER}},
    }  # fmt: skip
    (final / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def teacher_probs(path: Path, df: pd.DataFrame, device) -> torch.Tensor:
    """The teacher's probabilities at temperature TEMP for every row of df."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).float().to(device)
    enc = tok(df["text"].tolist(), padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="pt")  # fmt: skip
    soft = F.softmax(logits_of(model, enc, device) / TEMP, -1)
    del model
    torch.cuda.empty_cache()
    return soft


class MultiHead(torch.nn.Module):
    """One encoder, first-token vector, a sentiment head and a topic head (E10)."""

    def __init__(self, backbone: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone).float()
        h = self.encoder.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.sentiment = torch.nn.Linear(h, len(NAMES))
        self.topic = torch.nn.Linear(h, len(TOPIC_NAMES))

    def forward(self, **batch):
        x = self.drop(self.encoder(**batch).last_hidden_state[:, 0])
        return self.sentiment(x), self.topic(x)

    def save(self, path: Path, tok) -> None:
        self.encoder.save_pretrained(path)
        tok.save_pretrained(path)
        torch.save({"sentiment": self.sentiment.state_dict(), "topic": self.topic.state_dict()},
                   path / "heads.pt")  # fmt: skip
        (path / "heads.json").write_text(json.dumps({"sentiment": NAMES, "topic": TOPIC_NAMES}))


@torch.no_grad()
def multihead_dev_f1(model, enc, y, device) -> tuple[float, float]:
    model.eval()
    preds = [[], []]
    n = len(enc[0]["input_ids"]), len(enc[1]["input_ids"])
    for task in (0, 1):
        for b, _ in batches(enc[task], torch.arange(n[task]), 256):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(**{k: v.to(device) for k, v in b.items()})[task]
            preds[task].append(out.float().argmax(-1).cpu())
    return tuple(
        float(f1_score(y[t], torch.cat(preds[t]).numpy(), average="macro")) for t in (0, 1)
    )


def train_multihead(seed, sent, topic, soft, save_to, device) -> dict:
    """sent/topic = (df, train_idx, dev_idx). One sentiment + one topic batch per step."""
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = MultiHead(STUDENT).to(device)

    def encode(df):
        return tok(df["text"].tolist(), padding="max_length", truncation=True,
                   max_length=MAX_LEN, return_tensors="pt")  # fmt: skip

    (sdf, s_tr, s_dev), (tdf, t_tr, t_dev) = sent, topic
    s_enc, t_enc = encode(sdf), encode(tdf)
    s_y, t_y = torch.tensor(sdf["y"].to_numpy()), torch.tensor(tdf["y"].to_numpy())
    dev_enc = ({k: v[s_dev] for k, v in s_enc.items()}, {k: v[t_dev] for k, v in t_enc.items()})
    dev_y = (sdf["y"].to_numpy()[s_dev], tdf["y"].to_numpy()[t_dev])

    epochs = EPOCHS_MULTI
    steps_per_epoch = (max(len(s_tr), len(t_tr)) + BATCH - 1) // BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * epochs * steps_per_epoch),
                                            epochs * steps_per_epoch)  # fmt: skip
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    def cycle(idx):
        while True:
            perm = torch.from_numpy(idx)[torch.randperm(len(idx))]
            yield from (perm[i : i + BATCH] for i in range(0, len(perm), BATCH))

    history, best, t0 = [], -1.0, time.time()
    s_iter, t_iter = cycle(s_tr), cycle(t_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        for step in range(1, steps_per_epoch + 1):
            sj, tj = next(s_iter), next(t_iter)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                s_logits = model(**{k: v[sj].to(device) for k, v in s_enc.items()})[0].float()
                t_logits = model(**{k: v[tj].to(device) for k, v in t_enc.items()})[1].float()
            s_loss = F.cross_entropy(s_logits, s_y[sj].to(device))
            kd = F.kl_div(F.log_softmax(s_logits / TEMP, -1), soft[sj].to(device),
                          reduction="batchmean") * TEMP**2  # fmt: skip
            loss = ALPHA * s_loss + (1 - ALPHA) * kd + F.cross_entropy(t_logits, t_y[tj].to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 50 == 0:
                print(f"  multihead seed {seed} epoch {epoch} step {step}: loss {loss.item():.4f}",
                      flush=True)  # fmt: skip
        s_f1, t_f1 = multihead_dev_f1(model, dev_enc, dev_y, device)
        mean = (s_f1 + t_f1) / 2
        history.append({"epoch": epoch, "dev_sentiment_f1": round(s_f1, 4),
                        "dev_topic_f1": round(t_f1, 4), "dev_mean_f1": round(mean, 4)})  # fmt: skip
        print(f"multihead seed {seed} epoch {epoch}: dev sentiment {s_f1:.4f} topic {t_f1:.4f}",
              flush=True)  # fmt: skip
        if mean > best:
            best = mean
            save_to.mkdir(parents=True, exist_ok=True)
            model.save(save_to, tok)
    return {"seed": seed, "best_dev_mean_f1": round(best, 4), "history": history,
            "train_minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip


def find_teacher(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    found = sorted(Path("/kaggle/input").glob(f"**/{TEACHER_DIR}/config.json"))
    return found[0].parent if found else None


def run_multitask(out: Path, limit: int | None, teacher: Path, device) -> dict:
    sdf, s_tr, s_dev, s_counts = load_data(limit)
    tdf, t_tr, t_dev, t_counts = load_topic(limit)
    print(json.dumps({"sentiment": s_counts, "topic": t_counts}, indent=2), flush=True)
    soft = teacher_probs(teacher, sdf, device)
    runs = [train_multihead(s, (sdf, s_tr, s_dev), (tdf, t_tr, t_dev), soft,
                            out / f"{MULTIHEAD}-seed{s}", device) for s in SEEDS]  # fmt: skip
    best = max(runs, key=lambda r: r["best_dev_mean_f1"])
    final = out / MULTIHEAD
    shutil.rmtree(final, ignore_errors=True)
    (out / f"{MULTIHEAD}-seed{best['seed']}").rename(final)
    for s in SEEDS:
        shutil.rmtree(out / f"{MULTIHEAD}-seed{s}", ignore_errors=True)
    metrics = {
        "backbone": STUDENT, "chosen_seed": best["seed"],
        "best_dev_mean_f1": best["best_dev_mean_f1"], "seeds": runs,
        "data": {"sentiment": s_counts, "topic": t_counts},
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "settings": {"max_len": MAX_LEN, "batch": BATCH, "lr": 5e-5, "epochs": 6, "warmup": WARMUP,
                     "distil": {"alpha": ALPHA, "temperature": TEMP, "teacher": str(teacher)}},
    }  # fmt: skip
    (final / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_multitask.json").write_text(json.dumps(metrics, indent=2))
    print("multihead seed", best["seed"], "dev mean macro-F1", best["best_dev_mean_f1"])
    return metrics


def main() -> None:
    """E29: R-Drop teacher, 3 seeds; choose single-vs-ensemble on dev; score the test once."""
    out = Path("/kaggle/working/models")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, train_idx, dev_idx, counts = load_data(None)
    print(json.dumps(counts, indent=2), flush=True)
    runs = [train_one(TEACHER, s, df, train_idx, dev_idx, out / f"rdrop-seed{s}", device)
            for s in SEEDS]  # fmt: skip
    tok = AutoTokenizer.from_pretrained(TEACHER)
    enc_all = tok(df["text"].tolist(), padding="max_length", truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")  # fmt: skip
    enc_dev = {k: v[dev_idx] for k, v in enc_all.items()}
    y_dev = df["y"].to_numpy()[dev_idx]
    test = pd.read_csv(TFNS.format("sent_valid.csv"))
    y_test = test["label"].map(TFNS_NAMES).map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    enc_test = tok(test["text"].tolist(), padding="max_length", truncation=True,
                   max_length=MAX_LEN, return_tensors="pt")  # fmt: skip
    dev_p, test_p = {}, {}
    for r in runs:
        path = out / f"rdrop-seed{r['seed']}"
        model = AutoModelForSequenceClassification.from_pretrained(path).float().to(device)
        dev_p[r["seed"]] = F.softmax(logits_of(model, enc_dev, device), -1).numpy()
        test_p[r["seed"]] = F.softmax(logits_of(model, enc_test, device), -1).numpy()
        del model
        torch.cuda.empty_cache()
    dev_f1 = {f"seed{s}": float(f1_score(y_dev, p.argmax(1), average="macro"))
              for s, p in dev_p.items()}  # fmt: skip
    ens_dev = np.mean(list(dev_p.values()), axis=0)
    dev_f1["ensemble3"] = float(f1_score(y_dev, ens_dev.argmax(1), average="macro"))
    chosen = max(dev_f1, key=dev_f1.get)
    if chosen == "ensemble3":
        p_test = np.mean(list(test_p.values()), axis=0)
    else:
        p_test = test_p[int(chosen.replace("seed", ""))]
    pred = p_test.argmax(1)
    rng = np.random.default_rng(0)
    accs = [float((pred[i] == y_test[i]).mean())
            for i in rng.integers(0, len(y_test), (1000, len(y_test)))]  # fmt: skip
    summary = {
        "data": counts, "seeds": runs, "dev_macro_f1": dev_f1, "chosen": chosen,
        "test": {"n": int(len(y_test)), "accuracy": float((pred == y_test).mean()),
                 "accuracy_ci95": [float(np.percentile(accs, q)) for q in (2.5, 97.5)],
                 "macro_f1": float(f1_score(y_test, pred, average="macro"))},
        "rdrop_alpha": RDROP_ALPHA, "reference_e7_teacher_accuracy": 0.905,
    }  # fmt: skip
    np.savez_compressed(out / "chosen_test_probs.npz", probs=p_test, y=y_test)
    (out / "summary_e29.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("dev_macro_f1", "chosen", "test")}, indent=2))


if __name__ == "__main__":
    main()
