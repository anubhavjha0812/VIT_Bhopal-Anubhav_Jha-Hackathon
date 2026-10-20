"""E25 extra rows for thin classes (docs/experiments/E25_TEACHER_STUDENT.md). No labels.

uv run python -m riskengine.eval.e25_extra --dry-run   # counts, writes nothing
uv run python -m riskengine.eval.e25_extra --build     # writes data/cache/e25/extra.csv.gz

The pool keeps at most 3,000 headlines per company, so about 37,000 clean headlines were left out.
Macroeconomic and Geopolitical are thin in the pool after the teachers' agreement filter. This
picks left-out headlines that mention words typical of those two classes. Keywords only choose
which headlines the teachers read; they are not labels. The teachers label them and the same
"both agree" filter applies. Same frame, same leak filter as the pool; nothing here reads a label.
"""

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

from riskengine.eval.e25_pool import (
    FINAL_FILES,
    POOL_PATH,
    assert_clean,
    exclusions,
    frame,
    remove_seen,
    settle,
)
from riskengine.paths import DATA, ROOT

SEED = 25
PER_GROUP = 2500
EXTRA_PATH = DATA / "cache" / "e25" / "extra.csv.gz"  # git-ignored (FNSPID licence)
MANIFEST = DATA / "outputs" / "e25_extra_manifest.json"
GROUPS = {
    "Macroeconomic": (
        r"inflation|deflation|interest rates?|rate (?:cut|hike|decision)|federal reserve|\bfed\b"
        r"|central bank|\becb\b|bank of (?:japan|england)|\bgdp\b|unemployment|jobless|payrolls"
        r"|jobs report|recession|stimulus|treasury yields?|bond yields?"
        r"|consumer (?:prices|confidence|spending)|retail sales|housing starts|\bpmi\b"
        r"|trade deficit|budget deficit|oil prices?"
        r"|crude|commodity|commodities|currency|\bdollar\b|\beuro\b|\byen\b|economy|economic"
    ),
    "Geopolitical": (
        r"\bwar\b|military|troops|sanctions?|tariffs?|trade (?:war|dispute|talks)|embargo"
        r"|elections?|parliament|prime minister|diplomat\w*|treaty|summit|protest\w*|\bcoup\b"
        r"|terror\w*|missile|nuclear|\biran\b|\biraq\b|russia\w*|ukrain\w*|north korea|israel\w*"
        r"|greece|greek|euro zone|eurozone|brexit|border"
    ),
}  # keyword lists only shortlist; the teachers decide the class


def hits(texts: pd.Series, pattern: str) -> pd.Series:
    """Number of keyword matches per text (more matches = a clearer candidate)."""
    rx = re.compile(pattern, re.IGNORECASE)
    return texts.map(lambda t: len(rx.findall(str(t))))


def shortlist(
    f: pd.DataFrame, groups: dict[str, str] = GROUPS, per_group: int = PER_GROUP, seed: int = SEED
) -> pd.DataFrame:
    """Per group: the headlines with most keyword hits (ties broken at random), up to per_group.
    A headline picked by two groups is kept once. Ids x000000... follow the sorted order."""
    rng = np.random.default_rng(seed)
    noise = pd.Series(rng.random(len(f)), index=f.index)
    chosen: dict[int, str] = {}
    for name, pattern in groups.items():
        score = hits(f["text"], pattern)
        order = pd.DataFrame({"score": score, "noise": noise})
        order = order[order["score"] > 0].sort_values(["score", "noise"], ascending=[False, True])
        for i in order.index[:per_group]:
            chosen.setdefault(i, name)
    out = f.loc[sorted(chosen)].assign(shortlist=lambda d: [chosen[i] for i in d.index])
    out = out.reset_index(drop=True)
    out.insert(0, "extra_id", [f"x{k:06d}" for k in range(len(out))])
    return out


def without_pool(f: pd.DataFrame, pool_texts: set[str]) -> pd.DataFrame:
    return f[~f["text"].isin(pool_texts)].reset_index(drop=True)


def build(write: bool) -> dict:
    from riskengine.eval.e24 import exclusion_corpus
    from riskengine.history import HISTORY_PATH, load_history

    corpus = exclusions(exclusion_corpus(), FINAL_FILES)
    pool_texts = set(pd.read_csv(POOL_PATH)["text"].astype(str))
    kept, info = remove_seen(frame(load_history()), corpus)
    left_out = without_pool(kept, pool_texts)
    picked = shortlist(left_out)
    extra, dropped = settle(picked.drop(columns="extra_id").assign(pool_id=""), corpus)
    extra = extra.drop(columns="pool_id")
    extra.insert(0, "extra_id", [f"x{k:06d}" for k in range(len(extra))])
    assert_clean(extra, corpus)
    if set(extra["text"]) & pool_texts:
        raise RuntimeError("extra rows overlap the pool")
    info |= {
        "rule": "left-out FNSPID 2010-2016 rows, same frame and leak filter as the pool, "
                "keyword shortlist for thin classes, both teachers label",  # fmt: skip
        "seed": SEED, "per_group": PER_GROUP, "left_out_after_pool": len(left_out),
        "dropped_in_final_passes": dropped, "extra": len(extra),
        "by_group": extra["shortlist"].value_counts().to_dict(),
        "extra_sha256": hashlib.sha256(
            "\n".join(extra["extra_id"] + "\t" + extra["text"]).encode()
        ).hexdigest(),
        "source": str(HISTORY_PATH.relative_to(ROOT)), "cleared_against": sorted(corpus),
    }  # fmt: skip
    if write:
        EXTRA_PATH.parent.mkdir(parents=True, exist_ok=True)
        extra.to_csv(EXTRA_PATH, index=False)
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
