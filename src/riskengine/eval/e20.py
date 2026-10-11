"""E20: choose the engine sentiment model on tweet and news dev sets, then confirm once on a
fresh test (rules: docs/DECISIONS.md, E20).

uv run python -m riskengine.eval.e20 --dev       # export B to int8, score both dev sets, decide
uv run python -m riskengine.eval.e20 --confirm   # SEntFiN once (needs the --dev decision)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from riskengine.eval.datasets import (
    EVAL_DIR,
    HF,
    TFNS_LABELS,
    _rows,
    fiqa_split,
    phrasebank_lines,
    tfns_test,
)
from riskengine.eval.leaks import corpora, flags, norm
from riskengine.paths import DATA, MODELS

OUT = DATA / "outputs" / "e20_results.json"
PREDS = DATA / "outputs" / "e20_predictions.csv.gz"
LABELS = ["negative", "neutral", "positive"]
A = "sentiment-bge-small-en-v1.5-int8"  # current engine (E5, E9)
B = "sentiment-distil-bge-small-en-v1.5-int8"  # E7 student, int8 by the E9 recipe
B_SRC = "sentiment-distil-bge-small-en-v1.5"
NEWS_DEV_N, MARGIN = 2000, 0.01
SENTFIN = "sentfin_v1.1.csv"  # Kaggle ankurzing/aspect-based-sentiment-analysis-for-financial-news


def model_dir(name: str) -> Path:
    return next(p.parent for p in MODELS.glob(f"**/{name}/config.json"))


def ensure_b_int8() -> Path:
    src = model_dir(B_SRC)
    out = src.parent / B
    if not (out / "model.onnx").exists():
        from riskengine.nlp.onnx_int8 import export_int8

        export_int8(src, out)
    return out


def tfns_dev() -> pd.DataFrame:
    """The E7 dev split: 10% of sent_train.csv (stratified, seed 42, drawn on the full file),
    minus rows E7 dropped (match an E1 test text, copies with different labels, repeats)."""
    tf = pd.read_csv(HF.format(repo="twitter-financial-news-sentiment", file="sent_train.csv"))
    tf = pd.DataFrame({"text": tf["text"], "label": tf["label"].map(TFNS_LABELS)})
    _, dev = train_test_split(
        np.arange(len(tf)), test_size=0.1, stratify=tf["label"], random_state=42
    )
    tf["dev"] = np.isin(np.arange(len(tf)), dev)
    extra = pd.concat(
        [phrasebank_lines("75Agree"), fiqa_split("train"), fiqa_split("validation")],
        ignore_index=True,
    ).assign(dev=False)
    df = pd.concat([tf, extra], ignore_index=True)  # the E7 order: TFNS, PhraseBank, FiQA
    key = df["text"].map(norm)
    keep = ~key.isin(set(tfns_test()["text"].map(norm)))  # 1. matches an E1 test text
    df, key = df[keep], key[keep]
    keep = df.groupby(key)["label"].transform("nunique") <= 1  # 2. copies that disagree
    df, key = df[keep], key[keep]
    keep = ~key.duplicated()  # 3. repeats keep the first copy
    return df[keep & df["dev"]][["text", "label"]].reset_index(drop=True)


def nwgi_dev() -> pd.DataFrame:
    path = EVAL_DIR / "nwgi_train.csv"
    if not path.exists():
        _rows("oliverwang15/news_with_gpt_instructions", "train")[["news", "label"]].to_csv(
            path, index=False
        )
    raw = pd.read_csv(path)
    df = pd.DataFrame({"text": raw["news"], "label": raw["label"].str.split().str[-1]})
    seen = np.zeros(len(df), bool)
    for name in ("E5 models (TFNS train)",
                 "E7/E10 models (TFNS + PhraseBank 75Agree + FiQA train/val)"):  # fmt: skip
        seen |= flags(df["text"].tolist(), corpora()[name])["seen"].to_numpy()
    return df[~seen].sample(NEWS_DEV_N, random_state=0).reset_index(drop=True)


def sentfin_test() -> pd.DataFrame:
    raw = pd.read_csv(EVAL_DIR / SENTFIN)
    rows = []
    for title, dec in zip(raw["Title"], raw["Decisions"], strict=True):
        d = json.loads(dec)
        if len(d) == 1:
            rows.append({"text": title, "label": next(iter(d.values()))})
    return pd.DataFrame(rows)


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def predict(name: str, texts: list[str]) -> list[str]:
    from riskengine.nlp.sentiment import TransformerSentiment

    return TransformerSentiment(model_dir(name), name=name).classify(texts)


def dev() -> dict:
    ensure_b_int8()
    sets = {"tweets (TFNS dev)": tfns_dev(), "news (NWGI train sample)": nwgi_dev()}
    scores = {m: {} for m in (A, B)}
    for m in (A, B):
        for s, df in sets.items():
            scores[m][s] = macro_f1(df["label"], predict(m, df["text"].tolist()))
    mean = {m: float(np.mean(list(v.values()))) for m, v in scores.items()}
    tweets = "tweets (TFNS dev)"
    pick_b = mean[B] > mean[A] + MARGIN and scores[B][tweets] >= scores[A][tweets] - MARGIN
    res = {
        "dev_rows": {s: len(df) for s, df in sets.items()},
        "dev_macro_f1": scores,
        "dev_mean": mean,
        "choice": B if pick_b else A,
    }
    OUT.write_text(json.dumps({"dev": res}, indent=2))
    return res


def paired(y, pa, pb, n: int = 1000, seed: int = 0) -> dict:
    y, pa, pb = map(np.asarray, (y, pa, pb))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = [macro_f1(y[i], pb[i]) - macro_f1(y[i], pa[i]) for i in idx]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"b_minus_a": macro_f1(y, pb) - macro_f1(y, pa), "ci95": [float(lo), float(hi)],
            "verdict": "higher" if lo > 0 else "lower" if hi < 0 else "tie"}  # fmt: skip


def confirm() -> dict:
    from riskengine.eval.__main__ import bootstrap_ci

    res = json.loads(OUT.read_text())
    if "dev" not in res:
        raise SystemExit("run --dev first: the choice is made before the test is scored")
    if "confirm" in res:
        raise SystemExit("SEntFiN was already scored once (E20)")
    test = sentfin_test()
    seen = np.zeros(len(test), bool)
    for name in ("E5 models (TFNS train)",
                 "E7/E10 models (TFNS + PhraseBank 75Agree + FiQA train/val)"):  # fmt: skip
        seen |= flags(test["text"].tolist(), corpora()[name])["seen"].to_numpy()
    test = test[~seen].reset_index(drop=True)
    y = test["label"].to_numpy()
    pred = {m: predict(m, test["text"].tolist()) for m in (A, B)}
    rows = {
        m: {"macro_f1": macro_f1(y, p), **bootstrap_ci(y, p, LABELS),
            "accuracy": float(np.mean(np.asarray(p) == y))}
        for m, p in pred.items()
    }  # fmt: skip
    pd.DataFrame({"row": range(len(y)), "label": y, A: pred[A], B: pred[B]}).to_csv(
        PREDS, index=False
    )
    diff = paired(y, pred[A], pred[B])
    choice = res["dev"]["choice"]
    switch = choice == B and diff["verdict"] != "lower"
    res["confirm"] = {
        "test": "SEntFiN 1.1, single-entity headlines",
        "rows": len(y),
        "dropped_seen": int(seen.sum()),
        "labels": pd.Series(y).value_counts().to_dict(),
        "scores": rows,
        "paired_b_minus_a": diff,
        "engine": B if switch else A,
    }
    OUT.write_text(json.dumps(res, indent=2))
    return res["confirm"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()
    if args.dev:
        print(json.dumps(dev(), indent=2))
    if args.confirm:
        print(json.dumps(confirm(), indent=2))


if __name__ == "__main__":
    main()
