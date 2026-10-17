"""E25 student data v3: v2 training file + hard-label rows resolved by two AI labellers.

uv run python -m riskengine.eval.e25_student_data_v3 --dry-run   # counts, writes nothing
uv run python -m riskengine.eval.e25_student_data_v3 --build     # writes train_v3 / valid_v3

Added rows (hard labels, no teacher probabilities):
- `resolved_ai`: the 963 teacher-disagreement news rows, final label from two AI labellers, only
  rows the second labeller did not flag as ambiguous; the 3 credit-class labels are dropped.
- `official_geo`: UK foreign-office titles, labelled blind by one AI labeller (second check
  pending), Geo capped at OFFICIAL_GEO_CAP so formal title style is not a hint for Geopolitical;
  all Other titles kept.
Validation gets 5% of the added rows (stratified, same split code as v2).
"""

import argparse
import json

import pandas as pd

from riskengine.eval.e25_pool import FINAL_FILES, assert_clean, exclusions
from riskengine.eval.e25_student_data import OUT_DIR, SEED, split_valid
from riskengine.nlp.events import LEARNED_CLASSES
from riskengine.paths import DATA, ROOT

TASK = ROOT / "private_notes/annotation_uploads/task1"
RESOLVED = TASK / "macrogeo_adjudication/macrogeo_labels_final.csv"
OFFICIAL = TASK / "official_geo/official_geo_labels_claude.csv"
OFFICIAL_GEO_CAP = 150
MANIFEST = DATA / "outputs" / "e25_student_data_v3_manifest.json"


def resolved_rows(path=RESOLVED) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["use_for_training"] & d["label_final"].isin(LEARNED_CLASSES)]
    return pd.DataFrame(
        {"id": d["id"], "text": d["text"], "label": d["label_final"], "source": "resolved_ai"}
    )


def official_rows(path=OFFICIAL, cap: int = OFFICIAL_GEO_CAP, seed: int = SEED) -> pd.DataFrame:
    d = pd.read_csv(path)
    geo = d[d["label"] == "Geopolitical"].sample(n=min(cap, (d["label"] == "Geopolitical").sum()),
                                                 random_state=seed)  # fmt: skip
    d = pd.concat([geo, d[d["label"] != "Geopolitical"]])
    return pd.DataFrame(
        {"id": d["id"], "text": d["text"], "label": d["label"], "source": "official_geo"}
    )


def add_rows(train: pd.DataFrame, valid: pd.DataFrame, extra: pd.DataFrame):
    """Drop added texts already in train/valid, split the rest 5% to validation, append."""
    seen = set(train["text"].str.lower().str.strip()) | set(valid["text"].str.lower().str.strip())
    extra = extra[~extra["text"].str.lower().str.strip().isin(seen)]
    extra = extra.drop_duplicates("text").reset_index(drop=True)
    add_train, add_valid = split_valid(extra)
    return (
        pd.concat([train, add_train], ignore_index=True),
        pd.concat([valid, add_valid], ignore_index=True),
        len(extra),
    )


def build(write: bool, check_leaks: bool = True) -> dict:
    train, valid = pd.read_csv(OUT_DIR / "train.csv"), pd.read_csv(OUT_DIR / "valid.csv")
    extra = pd.concat([resolved_rows(), official_rows()], ignore_index=True)
    if check_leaks:
        from riskengine.eval.e24 import exclusion_corpus

        assert_clean(extra, exclusions(exclusion_corpus(), FINAL_FILES))
    train3, valid3, added = add_rows(train, valid, extra)
    info = {
        "v2_train": len(train), "v2_valid": len(valid), "added_rows": added,
        "train": len(train3), "valid": len(valid3), "leak_check": check_leaks,
        "train_by_class": train3["label"].value_counts().to_dict(),
        "train_by_source": train3["source"].value_counts().to_dict(),
        "added_by_source_class": {
            f"{k[0]}/{k[1]}": int(v) for k, v in extra.groupby(["source", "label"]).size().items()
        },
        "overlap_train_valid": len(set(train3["text"]) & set(valid3["text"])),
    }  # fmt: skip
    if write:
        train3.to_csv(OUT_DIR / "train_v3.csv", index=False)
        valid3.to_csv(OUT_DIR / "valid_v3.csv", index=False)
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
