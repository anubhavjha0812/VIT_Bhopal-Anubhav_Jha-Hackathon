"""E28 candidate G on D1, D2 and R3 (dev sets only): C's routing with an encoder + TF-IDF model.

uv run python -m riskengine.eval.e28_g_eval --probs <folder with *_eval_probs.npz and summary.json>

No E24 final file is read. G is exploratory; the D2 rule decides whether it can be tested on fresh
rows (a gain of at least 0.02 macro-F1 over the running candidate C on D2, fixed 12 classes).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.eval.e24_dev import Config, build_predictor, engine_probs
from riskengine.eval.e25_retest_score import bootstrap_gain, fixed12, per_class
from riskengine.paths import DATA, ROOT

TASK = ROOT / "private_notes/annotation_uploads/task1"
LABELS = {
    "D1": TASK / "reference_v7/development/D1_labels.csv",
    "D2": TASK / "reference_v7/development/D2_labels.csv",
    "R3": TASK / "retest_set/R3_labels_claude.csv",
}
MIN_WIN = 0.02


def lookup_probs(npz_path: Path, classes: list[str]):
    z = np.load(npz_path, allow_pickle=True)
    table = {t: p for t, p in zip(z["texts"], z["probs"], strict=True)}

    def probs(texts):
        return classes, np.stack([table[t] for t in texts])

    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", type=Path, required=True)
    args = ap.parse_args()
    classes = json.loads((args.probs / "summary.json").read_text())["classes"]
    base = build_predictor(Config("C"), engine_probs())
    ev = pd.read_csv(TASK / "e28_encoder_input/eval_texts.csv", dtype=str)
    texts_by_set, y_by_set = {}, {}
    for name, path in LABELS.items():
        lab = pd.read_csv(path, dtype=str).set_index("item_id")["label"]
        sub = ev[ev["set"] == name]
        texts_by_set[name] = sub["text"].tolist()
        y_by_set[name] = [lab[i] for i in sub["id"]]
    c_pred = {s: base(t)[0] for s, t in texts_by_set.items()}
    out = {"rule": f"gain over C on D2 >= {MIN_WIN}", "sets": {}}
    out["C"] = {s: round(fixed12(y_by_set[s], c_pred[s]), 4) for s in texts_by_set}
    for npz in sorted(args.probs.glob("*_eval_probs.npz")):
        tag = npz.name.replace("_eval_probs.npz", "")
        pred = build_predictor(Config("C"), lookup_probs(npz, classes))
        row = {}
        for s, t in texts_by_set.items():
            g = pred(t)[0]
            f = fixed12(y_by_set[s], g)
            row[s] = {
                "macro_f1": round(f, 4),
                "gain_vs_C": round(f - out["C"][s], 4),
                "gain_ci95": bootstrap_gain(y_by_set[s], c_pred[s], g),
                "per_class": per_class(y_by_set[s], g).round(3).to_dict("records"),
            }
        row["passes_D2_rule"] = bool(row["D2"]["gain_vs_C"] >= MIN_WIN)
        out["sets"][tag] = row
        print(
            tag,
            {s: (row[s]["macro_f1"], row[s]["gain_vs_C"]) for s in texts_by_set},
            "D2 rule:",
            row["passes_D2_rule"],
        )
    (DATA / "outputs" / "e28_g_eval.json").write_text(json.dumps(out, indent=2))
    print("C:", out["C"])


if __name__ == "__main__":
    main()
