"""E21 fresh tests: FinEntity news paragraphs and a blind hand-labelled tweet sample
(rules: docs/DECISIONS.md, E21). Nothing here scores a model.

uv run python -m riskengine.eval.e21 --sample   # draw the 300 tweets, write the blind CSV

Writes data/annotation/e21_tweets_to_label.csv (text, kept out of git: tweet terms of use) and
data/outputs/e21_sample.json (row ids, text hashes, the overlap check and file hashes).
"""

import argparse
import ast
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd

from riskengine.eval.datasets import EVAL_DIR, fiqa_split, nwgi_test, phrasebank_lines, tfns_test
from riskengine.eval.e20 import sentfin_test
from riskengine.eval.leaks import NEAR, flags, tfns_train
from riskengine.paths import DATA

FINENTITY_REV = "e86e5451a1fce4731fe07c710649a229d92e5cab"
FINENTITY_URL = (
    f"https://huggingface.co/datasets/yixuantt/FinEntity/resolve/{FINENTITY_REV}/FinEntity.json"
)
FINENTITY = EVAL_DIR / "finentity.json"
ANNOTATION = DATA / "annotation" / "e21_tweets_to_label.csv"
SAMPLE_META = DATA / "outputs" / "e21_sample.json"
PER_COMPANY, MIN_ELIGIBLE, MIN_WORDS, SEED = 30, 50, 5, 21


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def finentity() -> pd.DataFrame:
    """The 851 paragraphs whose entities all carry one label (E21)."""
    if not FINENTITY.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        FINENTITY.write_bytes(urllib.request.urlopen(FINENTITY_URL).read())
    rows = []
    for r in json.loads(FINENTITY.read_text()):
        ann = r["annotations"]
        ann = ast.literal_eval(ann) if isinstance(ann, str) else ann
        labels = {a["label"].lower() for a in ann}
        if len(labels) == 1:
            rows.append({"text": r["content"], "label": labels.pop()})
    return pd.DataFrame(rows)


def other_texts() -> list[str]:
    """Every text from the sets E21 checks overlap against (training, dev and tests)."""
    nwgi_train = pd.read_csv(EVAL_DIR / "nwgi_train.csv")["news"]
    sets = [
        tfns_train(),
        tfns_test()["text"],
        phrasebank_lines("50Agree")["text"],
        *(fiqa_split(s)["text"] for s in ("train", "validation", "test")),
        nwgi_train,
        nwgi_test()["text"],
        sentfin_test()["text"],
        finentity()["text"],
    ]
    return [str(t) for s in sets for t in s]


def eligible() -> pd.DataFrame:
    from riskengine.ingest.tweets import load_equinxx

    df = load_equinxx()[0]
    df = df[df["tickers"].map(len) == 1].copy()
    df["ticker"] = df["tickers"].map(lambda t: t[0])
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    return df[["published_at", "ticker", "text"]].reset_index(drop=True)


def sample(df: pd.DataFrame, others: list[str]) -> tuple[pd.DataFrame, dict]:
    counts = df["ticker"].value_counts()
    companies = sorted(counts[counts >= MIN_ELIGIBLE].index)
    picked, skipped = [], {}
    rng = np.random.default_rng(SEED)
    for tk in companies:
        pool = df[df["ticker"] == tk].iloc[rng.permutation(int(counts[tk]))]
        taken, n_skip, start = [], 0, 0
        while len(taken) < PER_COMPANY and start < len(pool):
            chunk = pool.iloc[start : start + 4 * PER_COMPANY]
            start += len(chunk)
            outside = flags(chunk["text"].tolist(), others)["seen"].to_numpy()
            for (_, row), seen in zip(chunk.iterrows(), outside, strict=True):
                if len(taken) == PER_COMPANY:
                    break
                prior = [t["text"] for t in taken] + [p["text"] for p in picked]
                dup = bool(prior) and bool(flags([row["text"]], prior)["seen"].iloc[0])
                if seen or dup:
                    n_skip += 1
                    continue
                taken.append(row.to_dict())
        picked += taken
        skipped[tk] = n_skip
    out = pd.DataFrame(picked)
    return out, {"companies": companies, "skipped_overlap": skipped}


def write_sample() -> dict:
    df = eligible()
    out, info = sample(df, other_texts())
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # mix companies
    out.insert(0, "row_id", [f"t{i:03d}" for i in range(len(out))])
    blind = out[["row_id", "ticker", "text"]].assign(label="", notes="")
    ANNOTATION.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(ANNOTATION, index=False)
    meta = {
        "rule": "docs/DECISIONS.md E21",
        "eligible": len(df),
        "eligible_by_company": df["ticker"].value_counts().to_dict(),
        **info,
        "rows": len(out),
        "near_threshold": NEAR,
        "row_ids": {
            r: {"ticker": t, "published_at": str(p), "text_sha256": sha(x)}
            for r, t, p, x in zip(
                out["row_id"], out["ticker"], out["published_at"], out["text"], strict=True
            )
        },  # fmt: skip
        "annotation_file_sha256": hashlib.sha256(ANNOTATION.read_bytes()).hexdigest(),
        "finentity": {
            "revision": FINENTITY_REV,
            "file_sha256": hashlib.sha256(finentity_bytes()).hexdigest(),
            "rows": len(finentity()),
            "labels": finentity()["label"].value_counts().to_dict(),
        },
    }
    SAMPLE_META.write_text(json.dumps(meta, indent=2))
    return meta


def finentity_bytes() -> bytes:
    finentity()
    return FINENTITY.read_bytes()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--build-training", action="store_true")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()
    if args.confirm:
        print(json.dumps(confirm(), indent=2, default=float))
    if args.dev:
        print(json.dumps(dev_choice(), indent=2))
    if args.build_training:
        print(json.dumps(build_training(), indent=2))
    if args.sample:
        meta = write_sample()
        print(json.dumps({k: v for k, v in meta.items() if k != "row_ids"}, indent=2))


# --- E21 training data (prepared here, trained on Kaggle: notebooks/e21_sentiment) ---
TRAIN_DIR = DATA / "cache" / "e21_train"


def build_training() -> dict:
    """Human-labelled training rows and the two dev sets, leak-filtered (E21)."""
    from sklearn.model_selection import train_test_split

    from riskengine.eval.e20 import tfns_dev
    from riskengine.eval.leaks import norm

    sentfin = sentfin_test()
    tr_idx, dev_idx = train_test_split(
        np.arange(len(sentfin)), test_size=0.1, stratify=sentfin["label"], random_state=SEED
    )
    dev_tweets = tfns_dev()
    dev_news = sentfin.iloc[dev_idx].reset_index(drop=True)
    from riskengine.eval.datasets import HF, TFNS_LABELS

    tfns = pd.read_csv(HF.format(repo="twitter-financial-news-sentiment", file="sent_train.csv"))
    tfns = pd.DataFrame({"text": tfns["text"], "label": tfns["label"].map(TFNS_LABELS)})
    parts = [
        tfns.assign(source="tfns"),
        phrasebank_lines("75Agree").assign(source="phrasebank"),
        *(fiqa_split(s).assign(source="fiqa") for s in ("train", "validation", "test")),
        sentfin.iloc[tr_idx].assign(source="sentfin"),
    ]
    train = pd.concat(parts, ignore_index=True)
    counts = {"rows_in": train["source"].value_counts().to_dict()}

    held = pd.concat(
        [dev_tweets["text"], dev_news["text"], tfns_test()["text"], finentity()["text"],
         pd.read_csv(ANNOTATION)["text"]],
        ignore_index=True,
    ).astype(str).tolist()  # fmt: skip
    key = train["text"].map(norm)
    exact = key.isin({norm(t) for t in held})
    near = flags(train["text"].astype(str).tolist(), held)["seen"].to_numpy()
    drop = exact.to_numpy() | near
    counts["dropped_overlap_with_dev_or_test"] = train[drop]["source"].value_counts().to_dict()
    train, key = train[~drop], key[~drop]
    conflict = train.groupby(key)["label"].transform("nunique") > 1
    counts["dropped_label_conflict"] = train[conflict]["source"].value_counts().to_dict()
    train, key = train[~conflict], key[~conflict]
    dup = key.duplicated()
    counts["dropped_duplicates"] = train[dup]["source"].value_counts().to_dict()
    train = train[~dup].reset_index(drop=True)
    counts["rows_used"] = train["source"].value_counts().to_dict()
    counts["dev_rows"] = {"tweets": len(dev_tweets), "news": len(dev_news)}

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_DIR / "train.csv", index=False)
    dev_tweets.to_csv(TRAIN_DIR / "dev_tweets.csv", index=False)
    dev_news.to_csv(TRAIN_DIR / "dev_news.csv", index=False)
    counts["files_sha256"] = {
        f: hashlib.sha256((TRAIN_DIR / f).read_bytes()).hexdigest()
        for f in ("train.csv", "dev_tweets.csv", "dev_news.csv")
    }
    (DATA / "outputs" / "e21_training_data.json").write_text(json.dumps(counts, indent=2))
    return counts


# --- E21 choice on dev (int8 for both; no fresh test is touched here) ---
A_NAME = "sentiment-bge-small-en-v1.5-int8"
C_SRC = "sentiment-e21-bge-small-en-v1.5"
C_NAME = C_SRC + "-int8"
RESULTS = DATA / "outputs" / "e21_results.json"
MARGIN = 0.01


def dev_choice() -> dict:
    from riskengine.eval.e20 import macro_f1, model_dir, predict
    from riskengine.nlp.onnx_int8 import export_int8

    src = model_dir(C_SRC)
    if not (src.parent / C_NAME / "model.onnx").exists():
        export_int8(src, src.parent / C_NAME)
    sets = {
        "tweets": pd.read_csv(TRAIN_DIR / "dev_tweets.csv"),
        "news": pd.read_csv(TRAIN_DIR / "dev_news.csv"),
    }
    scores = {
        m: {s: macro_f1(df["label"], predict(m, df["text"].astype(str).tolist()))
            for s, df in sets.items()}
        for m in (A_NAME, C_NAME)
    }  # fmt: skip
    mean = {m: (v["tweets"] + v["news"]) / 2 for m, v in scores.items()}
    switch = (
        mean[C_NAME] > mean[A_NAME] + MARGIN
        and scores[C_NAME]["tweets"] >= scores[A_NAME]["tweets"] - MARGIN
    )
    res = {
        "dev": {
            "rows": {s: len(df) for s, df in sets.items()},
            "macro_f1": scores,
            "dev_score": mean,
            "choice": C_NAME if switch else A_NAME,
            "rule": "C if dev score > A + 0.01 and tweet dev >= A - 0.01 (E21)",
        }
    }
    RESULTS.write_text(json.dumps(res, indent=2))
    return res["dev"]


LABELLED = DATA / "annotation" / "e21_tweets_final_labels.csv"
PREDS = DATA / "outputs" / "e21_predictions.csv.gz"


def confirm() -> dict:
    """Once, after the dev choice: FinEntity (primary), the AI-assisted tweets and TFNS test
    (secondary). Per-row predictions and all metrics are stored (E21 + amendment)."""
    from riskengine.eval.__main__ import bootstrap_ci
    from riskengine.eval.e20 import LABELS, macro_f1, paired, predict

    res = json.loads(RESULTS.read_text())
    if "dev" not in res:
        raise SystemExit("run --dev first: the choice is made before the test is scored")
    if "confirm" in res:
        raise SystemExit("E21 confirmation was already run once")
    tw = pd.read_csv(LABELLED)
    meta = json.loads(SAMPLE_META.read_text())["row_ids"]
    if [meta[r]["text_sha256"] for r in tw["row_id"]] != [sha(t) for t in tw["text"]]:
        raise SystemExit("labelled file does not match the frozen sample")
    skipped = int((tw["label"] == "skip").sum())
    tw = tw[tw["label"].isin(LABELS)].reset_index(drop=True)
    sets = {
        "FinEntity (primary, human)": finentity(),
        "Tweets 2021-22 (secondary, AI-assisted, human-reviewed)": tw[["text", "label"]],
        "TFNS test (secondary, observed before)": tfns_test(),
    }
    out, preds = {}, []
    for name, df in sets.items():
        y = df["label"].to_numpy()
        p = {m: predict(m, df["text"].astype(str).tolist()) for m in (A_NAME, C_NAME)}
        out[name] = {
            "rows": len(y),
            "labels": pd.Series(y).value_counts().to_dict(),
            **{m: {"macro_f1": macro_f1(y, p[m]), **bootstrap_ci(y, p[m], LABELS, per_class=True),
                   "accuracy": float(np.mean(np.asarray(p[m]) == y))} for m in p},
            "paired_c_minus_a": paired(y, p[A_NAME], p[C_NAME]),
        }  # fmt: skip
        preds.append(pd.DataFrame({"test_set": name, "row": range(len(y)), "label": y,
                                   A_NAME: p[A_NAME], C_NAME: p[C_NAME]}))  # fmt: skip
    pd.concat(preds, ignore_index=True).to_csv(PREDS, index=False)
    primary = out["FinEntity (primary, human)"]["paired_c_minus_a"]["verdict"]
    switch = res["dev"]["choice"] == C_NAME and primary != "lower"
    res["confirm"] = {
        "sets": out,
        "tweets_skipped_by_annotator": skipped,
        "labelled_file_sha256": hashlib.sha256(LABELLED.read_bytes()).hexdigest(),
        "engine": C_NAME if switch else A_NAME,
    }
    RESULTS.write_text(json.dumps(res, indent=2, default=float))
    return res["confirm"]


if __name__ == "__main__":
    main()
