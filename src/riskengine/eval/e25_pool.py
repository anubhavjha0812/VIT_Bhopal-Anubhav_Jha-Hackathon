"""E25 training pool for the teacher–student event model (docs/E25_TEACHER_STUDENT.md). No labels.

uv run python -m riskengine.eval.e25_pool --dry-run   # counts and hashes, writes nothing
uv run python -m riskengine.eval.e25_pool --build     # writes the pool and its manifest

Unlabelled FNSPID headlines published before the earliest test text (2010 – 2016; the E24 final
news is 2017 – 2018, the dev set 2019 – Jun 2020). Same frame rule as the E24 final news: E17
relevance kept, role subject, unique text. Then: every text that is an exact or near duplicate
(cosine >= 0.9, E18 leak rule) of a test or training-excluded text is removed, and each company
is capped so one company cannot dominate. Nothing here reads a label or a model output.
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from riskengine.eval.leaks import flags
from riskengine.paths import DATA, ROOT

SEED = 25
START, END = "2010-01-01", "2017-01-01"  # end exclusive
PER_COMPANY = 3000
RULE = "FNSPID 2010-2016, E17 relevance kept, role subject, unique text, per-company cap"
POOL_PATH = DATA / "cache" / "e25" / "pool.csv.gz"  # git-ignored (FNSPID licence)
MANIFEST = DATA / "outputs" / "e25_pool_manifest.json"
FINAL_FILES = {
    "E24 final news": DATA / "gold/e24/final_news_to_label.csv",
    "E24 final tweets": DATA / "gold/e24/final_tweets_to_label.csv",
}


def sha_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def frame(history: pd.DataFrame, start: str = START, end: str = END) -> pd.DataFrame:
    """One row per unique headline text, with the companies it is linked to (E24 news rule)."""
    h = history[history["why"].isna() & (history["role"] == "subject")]
    h = h[(h["published_at"] >= start) & (h["published_at"] < end)]
    h = h.sort_values(["published_at", "text", "ticker"], kind="stable")
    tickers = ("ticker", lambda s: " ".join(sorted(set(s))))
    f = h.groupby("text", sort=False).agg(published_at=("published_at", "first"), tickers=tickers)
    return (
        f.reset_index().sort_values(["published_at", "text"], kind="stable").reset_index(drop=True)
    )


def exclusions(
    corpus: dict[str, list[str]], final_files: dict | None = None
) -> dict[str, list[str]]:
    """The E24 exclusion corpus plus the final sample texts (they are not in it)."""
    out = dict(corpus)
    for name, path in (final_files or FINAL_FILES).items():
        out[name] = pd.read_csv(path, dtype=str)["text"].tolist()
    return out


def remove_seen(f: pd.DataFrame, corpus: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    """Drop rows that match any excluded text exactly or by cosine >= 0.9."""
    texts = f["text"].astype(str).tolist()
    seen = np.zeros(len(f), bool)
    counts = {}
    for name, other in corpus.items():
        hit = flags(texts, other)["seen"].to_numpy()
        counts[name] = int(hit.sum())
        seen |= hit
    kept = f[~seen].reset_index(drop=True)
    return kept, {"frame": len(f), "excluded_by_set": counts, "excluded_total": int(seen.sum()),
                  "after_exclusions": len(kept)}  # fmt: skip


def cap_per_company(f: pd.DataFrame, cap: int = PER_COMPANY, seed: int = SEED) -> pd.DataFrame:
    """Keep at most `cap` headlines per primary company (the first ticker), chosen at random."""
    primary = f["tickers"].str.split().str[0]
    order = np.random.default_rng(seed).permutation(len(f))
    taken: dict[str, int] = {}
    keep = []
    for i in order:
        t = primary.iloc[i]
        if taken.get(t, 0) < cap:
            taken[t] = taken.get(t, 0) + 1
            keep.append(i)
    out = f.iloc[sorted(keep)].reset_index(drop=True)
    out.insert(0, "pool_id", [f"p{k:06d}" for k in range(len(out))])
    return out


def settle(pool: pd.DataFrame, corpus: dict[str, list[str]], rounds: int = 5):
    """Repeat the leak filter on the final pool until a pass drops nothing.

    The cosine uses a TF-IDF fitted on the texts being compared, so a text right at the 0.9
    border can flip once the pool shrinks. Returns the pool (ids renumbered) and rows dropped.
    """
    dropped = 0
    for _ in range(rounds):
        smaller, _ = remove_seen(pool, corpus)
        dropped += len(pool) - len(smaller)
        if len(smaller) == len(pool):
            break
        pool = smaller
    pool = pool.drop(columns="pool_id").reset_index(drop=True)
    pool.insert(0, "pool_id", [f"p{k:06d}" for k in range(len(pool))])
    return pool, dropped


def assert_clean(pool: pd.DataFrame, corpus: dict[str, list[str]]) -> None:
    """Fail if any pool text is an exact or near duplicate of an excluded text."""
    texts = pool["text"].astype(str).tolist()
    for name, other in corpus.items():
        n = int(flags(texts, other)["seen"].sum())
        if n:
            raise RuntimeError(f"{n} pool texts match {name}")


def build(write: bool) -> dict:
    from riskengine.eval.e24 import exclusion_corpus
    from riskengine.history import HISTORY_PATH, load_history

    corpus = exclusions(exclusion_corpus())
    f = frame(load_history())
    kept, info = remove_seen(f, corpus)
    pool, extra = settle(cap_per_company(kept), corpus)
    info["dropped_in_final_passes"] = extra
    assert_clean(pool, corpus)
    primary = pool["tickers"].str.split().str[0]
    year = pd.to_datetime(pool["published_at"], utc=True).dt.year
    info |= {
        "rule": RULE, "seed": SEED, "cap_per_company": PER_COMPANY, "window": [START, END],
        "pool": len(pool), "by_company": primary.value_counts().to_dict(),
        "by_year": year.value_counts().sort_index().to_dict(),
        "pool_sha256": hashlib.sha256(
            "\n".join(pool["pool_id"] + "\t" + pool["text"]).encode()
        ).hexdigest(),
        "source": str(HISTORY_PATH.relative_to(ROOT)), "source_sha256": sha_file(HISTORY_PATH),
        "cleared_against": sorted(corpus),
    }  # fmt: skip
    if write:
        POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(POOL_PATH, index=False)
        MANIFEST.write_text(json.dumps(info, indent=2, default=int))
    return info


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--build", action="store_true")
    args = a.parse_args()
    print(json.dumps(build(write=args.build), indent=2, default=int))


if __name__ == "__main__":
    main()
