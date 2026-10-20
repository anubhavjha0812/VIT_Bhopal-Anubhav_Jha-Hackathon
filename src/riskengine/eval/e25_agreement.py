"""E25: how well do the teachers agree with the D1 reference, and with each other?

uv run python -m riskengine.eval.e25_agreement --dir <folder with d1_<teacher>.csv.gz files>

Reads the teacher probabilities (Kaggle output) and the v6 reference labels of D1 only (never D2
or the final sets), and writes data/outputs/e25_d1_agreement.json. The reference stays on this
machine. Also reads each teacher's <tag>_<teacher>_meta.json (if present) and applies the speed
rule of docs/experiments/E25_TEACHER_STUDENT.md to the second teacher.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from riskengine.eval.e24_gate import CLASSES
from riskengine.paths import DATA, ROOT

REFERENCE = ROOT / "private_notes/annotation_uploads/task1/reference_v6/development/D1_labels.csv"
OUT = DATA / "outputs" / "e25_d1_agreement.json"
MAX_POOL_HOURS = 8.0  # E25: a second teacher slower than this is replaced by the backup


def top_class(probs: pd.DataFrame) -> pd.Series:
    """Most likely class per row from the p_<class> columns."""
    cols = [f"p_{c}" for c in CLASSES]
    return pd.Series(np.array(CLASSES)[probs[cols].to_numpy().argmax(axis=1)], index=probs.index)


def load_teacher(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("id")
    missing = [c for c in CLASSES if f"p_{c}" not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing probability columns {missing}")
    return df


def score_vs_reference(pred: pd.Series, ref: pd.Series) -> dict:
    """Agreement with the reference: accuracy, fixed 12-class macro-F1, supported-class macro-F1."""
    ids = ref.index.intersection(pred.index)
    y, p = ref.loc[ids], pred.loc[ids]
    supported = [c for c in CLASSES if (y == c).any()]
    return {
        "rows": len(ids),
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1_fixed_12": round(float(f1_score(y, p, labels=CLASSES, average="macro",
                                                  zero_division=0)), 4),
        "macro_f1_supported": round(float(f1_score(y, p, labels=supported, average="macro",
                                                   zero_division=0)), 4),
        "kappa": round(float(cohen_kappa_score(y, p)), 4),
        "predicted_counts": p.value_counts().to_dict(),
        "reference_counts": y.value_counts().to_dict(),
    }  # fmt: skip


def pair_report(a: pd.Series, b: pd.Series, ref: pd.Series) -> dict:
    """Two teachers: how often they agree, and how right the agreed rows are (the E25 filter)."""
    ids = a.index.intersection(b.index).intersection(ref.index)
    same = (a.loc[ids] == b.loc[ids]).to_numpy()
    kept, dropped = ids[same], ids[~same]

    def right(rows):
        return round(float((a.loc[rows] == ref.loc[rows]).mean()), 4) if len(rows) else None

    return {
        "rows": len(ids),
        "agree_rate": round(float(same.mean()), 4),
        "kept_rows": len(kept),
        "accuracy_on_kept": right(kept),
        "accuracy_on_dropped": right(dropped),
        "kept_by_class": a.loc[kept].value_counts().to_dict(),
        "kappa_between_teachers": round(float(cohen_kappa_score(a.loc[ids], b.loc[ids])), 4),
    }


def speed_decision(metas: dict[str, dict], second: str = "mistral_nemo") -> dict:
    """E25 rule for the second teacher: keep it unless it failed or the projected pool time is
    above MAX_POOL_HOURS. (A clearly worse D1 agreement is judged separately, from the report.)"""
    m = metas.get(second)
    if m is None:
        return {"second_teacher": second, "decision": "no meta found: it failed or has not run"}
    hours = m.get("projected_pool_hours")
    if hours is None:
        return {"second_teacher": second, "decision": "no speed measured (resumed run)"}
    keep = hours <= MAX_POOL_HOURS
    return {
        "second_teacher": second, "projected_pool_hours": hours,
        "rows_per_second": m.get("rows_per_second"), "peak_gpu_gib": m.get("peak_gpu_gib"),
        "limit_hours": MAX_POOL_HOURS,
        "decision": "keep" if keep else "too slow: use the backup (llama31_8b)",
    }  # fmt: skip


def build(folder: Path, tag: str = "d1", reference: Path = REFERENCE) -> dict:
    files = sorted(folder.glob(f"{tag}_*.csv.gz"))
    teachers = {f.name[len(tag) + 1 : -len(".csv.gz")]: load_teacher(f) for f in files}
    if not teachers:
        raise SystemExit(f"no {tag}_<teacher>.csv.gz in {folder}")
    preds = {name: top_class(df) for name, df in teachers.items()}
    metas = {
        name: json.loads(p.read_text())
        for name in teachers
        if (p := folder / f"{tag}_{name}_meta.json").exists()
    }
    if tag == "dry":  # pool rows have no reference: speed and class mix only
        return {
            "teachers": {
                n: {"predicted_counts": p.value_counts().to_dict()} for n, p in preds.items()
            },
            "second_teacher_speed": speed_decision(metas),
            "note": "dry run: pool rows, no reference label used.",
        }
    ref = pd.read_csv(reference).set_index("item_id")["label"]
    return {
        "reference": str(reference.relative_to(ROOT)),
        "teachers": {n: score_vs_reference(p, ref) for n, p in preds.items()},
        "pairs": {
            f"{a}+{b}": pair_report(preds[a], preds[b], ref) for a, b in combinations(preds, 2)
        },
        "second_teacher_speed": speed_decision(metas),
        "note": "AI-derived reference; D1 only; no D2 or final row used.",
    }  # fmt: skip


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--dir", type=Path, required=True)
    a.add_argument("--tag", default="d1", choices=["d1", "dry"])
    args = a.parse_args()
    report = build(args.dir, args.tag)
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
