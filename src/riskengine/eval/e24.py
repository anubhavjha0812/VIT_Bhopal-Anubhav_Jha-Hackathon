"""E24: freeze the final samples and the dev split (docs/DECISIONS.md, E24). No labels here.

uv run python -m riskengine.eval.e24 --dry-run   # print the blind-file hashes, write nothing
uv run python -m riskengine.eval.e24 --freeze    # write the blind files, split, manifest, backups

Final news: 200 FNSPID headlines 2017-2018 (earlier-period evaluation). Final tweets: 100 equinxx
tweets 2021-22 (later-period, cross-source). Simple random samples (seed 24) from stably sorted
frames after the E24 exclusions. Dev: E3 split into D1/D2, prediction-stratified.
"""

import argparse
import hashlib
import json
import shutil
import stat

import numpy as np
import pandas as pd

from riskengine.eval.leaks import flags
from riskengine.paths import DATA, ROOT

SEED, N_NEWS, N_TWEETS = 24, 200, 100
NEWS_FROM, NEWS_TO = "2017-01-01", "2019-01-01"  # end exclusive
NEWS_RULE = "FNSPID 2017-2018, E17 relevance kept, role subject, unique text"
TWEET_RULE = "equinxx 2021-22, cleaned and linked (load_equinxx), unique text"
OUT = DATA / "gold" / "e24"
MANIFEST = DATA / "outputs" / "e24_final_sample.json"
BACKUP = ROOT / "private_notes" / "e24_frozen"
OTHER = ROOT.parent / "iiit-dharwad-arnav-gupta-hackathon-main"
HF = "https://huggingface.co/datasets/zeroshot/{}/resolve/main/{}"
CREDIT_WORD = (
    r"rating|moody|fitch|downgrad|default|bankrupt|chapter 11|covenant|lender|debt|bond|credit|loan"
)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(open(path, "rb").read())


def exclusion_corpus() -> dict[str, list[str]]:
    """Every text E24 must not test on (names as in DECISIONS E24)."""
    from riskengine.eval.e21 import finentity

    def col(path, name):
        return pd.read_csv(path)[name].astype(str).tolist()

    train = []
    for repo, files in (("twitter-financial-news-topic", ["topic_train.csv", "topic_valid.csv"]),
                        ("twitter-financial-news-sentiment",
                         ["sent_train.csv", "sent_valid.csv"])):  # fmt: skip
        for f in files:
            train += pd.read_csv(HF.format(repo, f))["text"].astype(str).tolist()
    rel = DATA / "relevance"
    other = [*col(OTHER / "data/gold/labels.csv", "text"),
             *col(OTHER / "data/gold/labels_2.csv", "text"),
             *col(OTHER / "data/gold/entity_check.csv", "headline")]  # fmt: skip
    return {
        "training (TFN topic train+valid, TFNS train+valid)": train,
        "E3": col(DATA / "gold/gold_to_label.csv", "headline"),
        "E23 sample": col(DATA / "gold/credit/credit_to_label.csv", "headline"),
        "E22b alerts": col(DATA / "outputs/e22b_alerts.csv", "text"),
        "E17 relevance samples": [
            *col(rel / "to_label.csv", "headline"),
            *col(rel / "audit_to_label.csv", "headline"),
        ],  # fmt: skip
        "E21 tweet sample": col(DATA / "annotation/e21_tweets_to_label.csv", "text"),
        "FinEntity": finentity()["text"].astype(str).tolist(),
        "other project's gold-1, gold-2, entity check": other,
    }


def news_frame() -> pd.DataFrame:
    from riskengine.history import HISTORY_PATH, load_history

    h = load_history()
    h = h[h["why"].isna() & (h["role"] == "subject")]
    h = h[(h["published_at"] >= NEWS_FROM) & (h["published_at"] < NEWS_TO)]
    h = h.sort_values(["published_at", "text", "ticker"], kind="stable")
    tickers = ("ticker", lambda s: " ".join(sorted(set(s))))
    f = h.groupby("text", sort=False).agg(published_at=("published_at", "first"), tickers=tickers)
    f = f.reset_index().sort_values(["published_at", "text"], kind="stable").reset_index(drop=True)
    return f.assign(source="FNSPID (E17 history)", source_file=str(HISTORY_PATH.relative_to(ROOT)),
                    source_sha256=sha_file(HISTORY_PATH),
                    frame_rule=NEWS_RULE)  # fmt: skip


def tweet_frame() -> pd.DataFrame:
    from riskengine.ingest.tweets import EQUINXX_PATH, load_equinxx

    t = load_equinxx()[0]
    t = t.assign(tickers=t["tickers"].map(lambda x: " ".join(sorted(x))))
    t = t.sort_values(["published_at", "text"], kind="stable").drop_duplicates("text")
    t = t[["published_at", "text", "tickers"]].reset_index(drop=True)
    return t.assign(source="equinxx (Kaggle)", source_file=str(EQUINXX_PATH.relative_to(ROOT)),
                    source_sha256=sha_file(EQUINXX_PATH),
                    frame_rule=TWEET_RULE)  # fmt: skip


def sample(frame: pd.DataFrame, n: int, corpus: dict, prefix: str) -> tuple[pd.DataFrame, dict]:
    texts = frame["text"].astype(str).tolist()
    seen = np.zeros(len(frame), bool)
    counts = {}
    for name, other in corpus.items():
        hit = flags(texts, other)["seen"].to_numpy()
        counts[name] = int(hit.sum())
        seen |= hit
    pool = frame[~seen].reset_index(drop=True)
    order = np.random.default_rng(SEED).permutation(len(pool))
    picked = []
    for i in order:  # take the next row unless it near-duplicates one already taken
        t = str(pool.loc[i, "text"])
        if picked and flags([t], [str(pool.loc[j, "text"]) for j in picked])["seen"].iloc[0]:
            continue
        picked.append(i)
        if len(picked) == n:
            break
    s = pool.loc[picked].reset_index(drop=True)
    s.insert(0, "item_id", [f"{prefix}{k:03d}" for k in range(len(s))])
    info = {"frame": len(frame), "excluded_by_set": counts, "excluded_total": int(seen.sum()),
            "pool_after_exclusions": len(pool), "sampled": len(s)}  # fmt: skip
    return s, info


def blind(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"item_id": s["item_id"], "text": s["text"],
                         "date": pd.to_datetime(s["published_at"], utc=True).dt.date.astype(str),
                         "companies": s["tickers"]})  # fmt: skip


def e3_split() -> pd.DataFrame:
    """D1/D2, prediction-stratified (by E3 design half and engine-predicted class), seed 24."""
    from riskengine.nlp.events import get_event_classifier

    e3 = pd.read_csv(DATA / "gold/gold_to_label.csv")
    half = np.where(e3["headline"].str.contains(CREDIT_WORD, case=False), "credit-word", "random")
    pred = get_event_classifier("model")(e3["headline"].astype(str).tolist())
    e3 = e3.assign(half=half, pred=pred, split="")
    rng = np.random.default_rng(SEED)
    for _, g in e3.groupby(["half", "pred"], sort=True):
        idx = rng.permutation(g.index.to_numpy())
        k = (len(idx) + 1) // 2
        e3.loc[idx[:k], "split"], e3.loc[idx[k:], "split"] = "D1", "D2"
    return e3[["doc_id", "split"]]  # predictions are not kept: annotators never see them


def build() -> dict:
    corpus = exclusion_corpus()
    news, n_info = sample(news_frame(), N_NEWS, corpus, "N")
    tweets, t_info = sample(tweet_frame(), N_TWEETS, corpus, "T")
    files = {
        "final_news_to_label.csv": blind(news),
        "final_tweets_to_label.csv": blind(tweets),
        "final_provenance.csv": pd.concat([news, tweets]).drop(columns=["text"]),
        "e3_split.csv": e3_split(),
    }
    blobs = {k: v.to_csv(index=False).encode() for k, v in files.items()}
    return {"blobs": blobs, "news": n_info, "tweets": t_info,
            "hashes": {k: sha_bytes(b) for k, b in blobs.items()}}  # fmt: skip


def freeze(res: dict, dry_hashes: list[dict]) -> dict:
    if any(h != res["hashes"] for h in dry_hashes):
        raise SystemExit("not deterministic: hashes differ between runs; nothing frozen")
    if (OUT / "final_news_to_label.csv").exists():
        raise SystemExit("E24 final sample already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, b in res["blobs"].items():
        (OUT / name).write_bytes(b)
    manifest = {
        "rule": "docs/DECISIONS.md E24",
        "seed": SEED,
        "news": res["news"] | {"label": "earlier-period evaluation (FNSPID 2017-2018)"},
        "tweets": res["tweets"]
        | {"label": "later-period, cross-source evaluation (equinxx 2021-22)"},
        "files_sha256": res["hashes"],
        "determinism": f"{len(dry_hashes) + 1} runs, identical hashes",
        "labels": "none made at freeze time",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    BACKUP.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in [*(OUT / n for n in res["blobs"]), MANIFEST]:
        dst = BACKUP / p.name
        shutil.copy2(p, dst)
        dst.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        lines.append(f"{sha_file(dst)}  {dst.name}")
    (BACKUP / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    (BACKUP / "MANIFEST.sha256").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--expect", nargs="*", default=[], help="hash JSON files from dry runs")
    args = p.parse_args()
    res = build()
    if args.dry_run:
        print(json.dumps(res["hashes"]))
        return
    if args.freeze:
        dry = [json.load(open(f)) for f in args.expect]
        print(json.dumps(freeze(res, dry), indent=2))


if __name__ == "__main__":
    main()
