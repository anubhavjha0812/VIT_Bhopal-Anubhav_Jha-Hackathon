"""E25 student training data. No test row is read.
Plan: docs/experiments/E25_TEACHER_STUDENT.md section 5

uv run python -m riskengine.eval.e25_student_data --dry-run   # counts, writes nothing
uv run python -m riskengine.eval.e25_student_data --build     # writes train.csv / valid.csv

Keeps the pool and extra headlines where both teachers pick the same class, drops the three credit
classes (the E13 rules own them), cuts "Other" to a fixed size, and splits a 5% validation slice
(stratified). Soft label = mean of the two teachers' probabilities over the 9 learned classes.
The Macro and Geo rows from Twitter Financial News Topic are added inside the Kaggle script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e25_agreement import top_class
from riskengine.eval.e25_pool import FINAL_FILES, assert_clean, exclusions
from riskengine.nlp.events import LEARNED_CLASSES
from riskengine.paths import DATA, ROOT

SEED = 25
OTHER_CAP = 5000
VALID_FRAC = 0.05
KAGGLE_OUT = ROOT / "private_notes/annotation_uploads/task1/e25_kaggle_output"
SETS = {  # name -> (teacher folder, file tag, text file, id column)
    "pool": ("pool", "pool", DATA / "cache/e25/pool.csv.gz", "pool_id"),
    "extra": ("extra", "extra", DATA / "cache/e25/extra.csv.gz", "extra_id"),
}
TEACHERS = ("qwen3_8b", "mistral_nemo")
OUT_DIR = DATA / "cache" / "e25" / "student"  # git-ignored
MANIFEST = DATA / "outputs" / "e25_student_data_manifest.json"
PCOLS = [f"p_{c}" for c in LEARNED_CLASSES]


def agreed_rows(q: pd.DataFrame, m: pd.DataFrame, texts: pd.Series, source: str) -> pd.DataFrame:
    """Rows where both teachers pick the same learned class; soft label over the 9 classes."""
    m = m.reindex(q.index)
    pq, pm = top_class(q), top_class(m)
    same = (pq == pm) & pq.isin(LEARNED_CLASSES)
    ids = pq.index[same]
    soft = (q.loc[ids, PCOLS].to_numpy() + m.loc[ids, PCOLS].to_numpy()) / 2
    soft = soft / soft.sum(axis=1, keepdims=True)
    out = pd.DataFrame(soft, columns=PCOLS).round(5)
    out.insert(0, "label", pq.loc[ids].to_numpy())
    out.insert(0, "text", texts.loc[ids].to_numpy())
    out.insert(0, "id", ids)
    out.insert(3, "source", source)
    return out


def cut_other(df: pd.DataFrame, cap: int = OTHER_CAP, seed: int = SEED) -> pd.DataFrame:
    other = df.index[df["label"] == "Other"]
    drop = np.random.default_rng(seed).permutation(other)[max(cap, 0) :]
    return df.drop(index=drop).sort_values("id").reset_index(drop=True)


def split_valid(df: pd.DataFrame, frac: float = VALID_FRAC, seed: int = SEED):
    """Stratified validation slice: each class gives round(frac * n) rows, at least 1 if n >= 2."""
    rng = np.random.default_rng(seed)
    valid = []
    for _, rows in df.groupby("label"):
        k = round(frac * len(rows)) if len(rows) >= 2 else 0
        k = max(k, 1) if len(rows) >= 2 else 0
        valid += list(rng.permutation(rows.index)[:k])
    mask = df.index.isin(valid)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def load_set(name: str, root: Path = KAGGLE_OUT) -> pd.DataFrame:
    folder, tag, text_file, id_col = SETS[name]
    teachers = [pd.read_csv(root / folder / f"{tag}_{t}.csv.gz").set_index("id") for t in TEACHERS]
    texts = pd.read_csv(text_file).set_index(id_col)["text"].astype(str)
    return agreed_rows(teachers[0], teachers[1], texts, name)


def digest(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(df["id"] + "\t" + df["label"]).encode()).hexdigest()


def build(write: bool, check_leaks: bool = True) -> dict:
    rows = pd.concat([load_set(n) for n in SETS], ignore_index=True)
    info = {"agreed_rows": len(rows), "agreed_by_class": rows["label"].value_counts().to_dict()}
    rows = cut_other(rows)
    train, valid = split_valid(rows)
    if check_leaks:
        from riskengine.eval.e24 import exclusion_corpus

        assert_clean(rows, exclusions(exclusion_corpus(), FINAL_FILES))
    info |= {
        "seed": SEED, "other_cap": OTHER_CAP, "valid_frac": VALID_FRAC,
        "classes": LEARNED_CLASSES, "leak_check": check_leaks,
        "train": len(train), "valid": len(valid),
        "train_by_class": train["label"].value_counts().to_dict(),
        "valid_by_class": valid["label"].value_counts().to_dict(),
        "train_by_source": train["source"].value_counts().to_dict(),
        "train_sha256": digest(train), "valid_sha256": digest(valid),
        "overlap_train_valid": len(set(train["id"]) & set(valid["id"])),
    }  # fmt: skip
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        train.to_csv(OUT_DIR / "train.csv", index=False)
        valid.to_csv(OUT_DIR / "valid.csv", index=False)
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
