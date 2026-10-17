"""Fresh event test sample for the E vs C re-test (docs/E25_RETEST.md). No labels, no model output.

uv run python -m riskengine.eval.e25_retest --dry-run   # counts, writes nothing
uv run python -m riskengine.eval.e25_retest --build     # writes the blind label sheet

400 headlines from the dev window (2019-01-01 to 2020-06-30): 200 drawn at random (the natural
mix) and 25 for each of 8 keyword strata so rare classes appear. Same frame rule and leak filter
as the E25 pool, so no text matches the D1/D2 dev rows, the final holdouts, E23 or any training
set. Keywords only choose which headlines are drawn; they are not labels. The label sheet holds
id and text only; the strata go to a separate file the labeller does not read.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

from riskengine.eval.e25_pool import FINAL_FILES, assert_clean, exclusions, frame, remove_seen
from riskengine.paths import DATA, ROOT

SEED = 26
START, END = "2019-01-01", "2020-07-01"  # end exclusive
RANDOM_ROWS, PER_STRATUM = 200, 25
OUT_DIR = ROOT / "private_notes/annotation_uploads/task1/retest_set"
MANIFEST = DATA / "outputs" / "e25_retest_manifest.json"
STRATA = {  # sampling cues fixed before any row was read
    "M&A": r"acqui\w+|merger|takeover|buyout|\bto buy\b|\bstake\b|spin-?off|\bipo\b",
    "Macroeconomic": (
        r"inflation|interest rates?|\bfed\b|federal reserve|central bank|\bgdp\b|unemployment"
        r"|jobs report|recession|oil prices?|treasury yields?|stimulus|currency|\bdollar\b"
    ),
    "Geopolitical": (
        r"\bwar\b|sanctions?|tariffs?|trade (?:war|deal|talks)|election|military|china|russia"
        r"|\biran\b|brexit|north korea"
    ),
    "Management change": r"\bceo\b|\bcfo\b|chairman|resigns?|steps? down|appoint\w*|names? new",
    "Earnings": r"earnings|quarterly|revenue|profit|\beps\b|guidance|dividend",
    "Analyst rating": r"upgrad\w+|downgrad\w+|price target|initiates|\brating\b|analyst",
    "Legal/Regulatory": (
        r"lawsuit|\bsues?\b|\bfine[sd]?\b|\bprobe\b|investigat\w+|settle\w*|\bcourt\b"
        r"|regulator|antitrust|\bsec\b"
    ),
    "Product news": (
        r"launch\w*|unveil\w*|releas\w+|introduc\w+|\brecall\w*|new (?:iphone|model|feature)"
    ),
}


def hit(texts: pd.Series, pattern: str) -> pd.Series:
    rx = re.compile(pattern, re.IGNORECASE)
    return texts.map(lambda t: bool(rx.search(str(t))))


def draw(
    f: pd.DataFrame,
    strata: dict[str, str] = STRATA,
    n_random: int = RANDOM_ROWS,
    per_stratum: int = PER_STRATUM,
    seed: int = SEED,
) -> pd.DataFrame:
    """Random rows first, then per_stratum keyword rows per stratum (never a row twice)."""
    rng = np.random.default_rng(seed)
    pick: dict[int, str] = {}
    for i in rng.permutation(f.index)[:n_random]:
        pick[i] = "random"
    for name, pattern in strata.items():
        cand = f.index[hit(f["text"], pattern).to_numpy() & ~f.index.isin(list(pick))]
        for i in rng.permutation(cand)[:per_stratum]:
            pick[i] = name
    out = f.loc[list(pick)].assign(stratum=list(pick.values()))
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "item_id", [f"r3_{k:04d}" for k in range(len(out))])
    return out


def build(write: bool) -> dict:
    from riskengine.eval.e24 import exclusion_corpus
    from riskengine.history import load_history

    corpus = exclusions(exclusion_corpus(), FINAL_FILES)
    kept, info = remove_seen(frame(load_history(), START, END), corpus)
    sample = draw(kept)
    assert_clean(sample, corpus)
    info |= {
        "window": [START, END], "seed": SEED, "rows": len(sample),
        "by_stratum": sample["stratum"].value_counts().to_dict(),
        "sha256": __import__("hashlib").sha256(
            "\n".join(sample["item_id"] + "\t" + sample["text"]).encode()
        ).hexdigest(),
    }  # fmt: skip
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sample[["item_id", "text", "published_at", "tickers"]].to_csv(
            OUT_DIR / "to_label.csv", index=False
        )
        sample[["item_id", "stratum"]].to_csv(OUT_DIR / "sampling_strata.csv", index=False)
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
