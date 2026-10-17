"""Score A, C, D and E on the fresh R3 set with the rule in docs/E25_RETEST.md.

uv run python -m riskengine.eval.e25_retest_score
uv run python -m riskengine.eval.e25_retest_score --student <folder> --output-suffix _v3

Reads the R3 sheet and its AI-derived labels only (no D2, final or E23 row). Per-row inference,
no threshold or temperature search (D uses the fixed 0.50 of the D1 report). Writes
data/outputs/e25_retest_{scores.json,per_class.csv,predictions.csv}. A comparison suffix
keeps alternate-student outputs separate; the default student and file names are unchanged.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from riskengine.eval.e24_dev import Config, build_predictor, engine_probs
from riskengine.eval.e24_gate import CLASSES
from riskengine.eval.e25_candidate import student_probs
from riskengine.paths import DATA, ROOT

SET = ROOT / "private_notes/annotation_uploads/task1/retest_set"
MIN_WIN, MIN_ROWS, MAX_DROP, SEED = 0.02, 10, 0.05, 26
CANDIDATES = {"A": Config("A"), "C": Config("C"), "D": Config("D", 0.5), "E": Config("E")}


def fixed12(y, p) -> float:
    return float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0))


def per_class(y, p) -> pd.DataFrame:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    pred = pd.Series(p).value_counts().reindex(CLASSES).fillna(0).astype(int)
    return pd.DataFrame({"class": CLASSES, "precision": pr, "recall": rc, "f1": f1,
                         "support": sup, "predicted": pred.to_numpy()})  # fmt: skip


def decide(c_f1: float, e_f1: float, c_pc: pd.DataFrame, e_pc: pd.DataFrame) -> dict:
    """The registered rule: E beats C by MIN_WIN and no class with MIN_ROWS rows loses MAX_DROP."""
    big = c_pc.loc[c_pc["support"] >= MIN_ROWS, "class"].tolist()
    drops = {
        c: round(
            float(e_pc.set_index("class").at[c, "f1"] - c_pc.set_index("class").at[c, "f1"]), 4
        )
        for c in big
    }
    ok_gain = e_f1 - c_f1 >= MIN_WIN
    ok_drop = all(d >= -MAX_DROP for d in drops.values())
    return {"gain": round(e_f1 - c_f1, 4), "gain_ok": ok_gain, "class_f1_change": drops,
            "no_big_drop": ok_drop, "e_replaces_c": ok_gain and ok_drop}  # fmt: skip


def bootstrap_gain(y, pc, pe, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    y, pc, pe = np.array(y), np.array(pc), np.array(pe)
    g = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        g.append(fixed12(y[i], pe[i]) - fixed12(y[i], pc[i]))
    return [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student", type=Path, help="local student folder; default is the v2 student"
    )
    parser.add_argument("--output-suffix", default="", help="separate comparison files, e.g. _v3")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:_[A-Za-z0-9_-]+)?", args.output_suffix):
        parser.error("--output-suffix must be empty or an underscore followed by letters/digits")
    targets = {
        name: DATA / f"outputs/e25_retest_{name}{args.output_suffix}.{extension}"
        for name, extension in (("scores", "json"), ("per_class", "csv"), ("predictions", "csv"))
    }
    if (args.student is not None or args.output_suffix) and any(
        p.exists() for p in targets.values()
    ):
        parser.error("comparison outputs already exist; choose a new --output-suffix")
    sheet = pd.read_csv(SET / "to_label.csv", dtype=str)
    lab = pd.read_csv(SET / "R3_labels_claude.csv", dtype=str)
    assert sheet["item_id"].tolist() == lab["item_id"].tolist() and len(sheet) == 400
    texts, y = sheet["text"].tolist(), lab["label"].tolist()
    base = engine_probs()
    student = student_probs(args.student) if args.student is not None else None
    preds = {k: build_predictor(cfg, student if k == "E" else base)(texts)[0]
             for k, cfg in CANDIDATES.items()}  # fmt: skip
    scores, pcs = {}, {}
    for k, p in preds.items():
        pcs[k] = per_class(y, p)
        scores[k] = {
            "macro_f1_fixed_12": round(fixed12(y, p), 4),
            "macro_f1_supported": round(float(f1_score(
                y, p, labels=[c for c in CLASSES if c in set(y)], average="macro",
                zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, p)), 4),
        }  # fmt: skip
    rule = decide(scores["C"]["macro_f1_fixed_12"], scores["E"]["macro_f1_fixed_12"],
                  pcs["C"], pcs["E"])  # fmt: skip
    rule["gain_ci95_bootstrap"] = bootstrap_gain(y, preds["C"], preds["E"])
    out = {"rows": len(y), "reference_counts": pd.Series(y).value_counts().to_dict(),
           "scores": scores, "rule": rule,
           "note": "AI-derived labels (Claude, blind); R3 is a development set."}  # fmt: skip
    if args.student is not None:
        out["student"] = str(args.student)
        out["student_inference"] = student.inference_metadata
    targets["scores"].parent.mkdir(parents=True, exist_ok=True)
    targets["scores"].write_text(json.dumps(out, indent=2))
    pd.concat({k: v for k, v in pcs.items()}, names=["candidate"]).reset_index(level=0).to_csv(
        targets["per_class"], index=False
    )
    pd.DataFrame({"item_id": sheet["item_id"], "text": texts, "label": y, **preds}).to_csv(
        targets["predictions"], index=False
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
